"""Layered ML Defense & Hardening Suite for Cybersecurity Flow Classifiers.

Implements three complementary defense layers:
  1. Defense 1 — Min-Max Adversarial Training (PGD-AT):
       Madry-style saddle-point optimization generating domain-constrained PGD
       evasion samples inside each mini-batch and optimizing a clean/adversarial
       loss mixture while isolating BatchNorm running statistics.
  2. Defense 2 — Feature Squeezing & Input Quantization:
       Inference-time bit-depth precision quantization and continuous median/Gaussian
       feature smoothing strictly restricted to mutable flow features (M_j = 1).
  3. Defense 3 — Manifold & Entropy Anomaly Detector:
       Fits class-conditional Gaussian distributions (centroids mu_c and shrinkage-
       regularized tied precision matrix Sigma^{-1}) on penultimate-layer embeddings
       (plus optional input telemetry manifold) alongside predictive Shannon entropy
       to reject off-manifold adversarial and zero-day out-of-distribution flows.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d, median_filter
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from data.dataset_loader import DomainFeatureSchema
from models.ids_classifier import (
    RobustIDSNet,
    compute_balanced_class_weights,
    evaluate_classifier,
    set_deterministic_seed,
)
from robustness.gradient_eval import evaluate_whitebox_robustness, pgd_attack

REJECT_LABEL: int = -1


@dataclass
class PGDATConfig:
    """Configuration for Madry-style Min-Max PGD Adversarial Training."""

    epochs: int = 45
    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    train_eps: float = 0.11
    pgd_steps: int = 10
    adv_mix_weight: float = 0.65
    warmup_epochs: int = 5
    patience: int = 12
    hidden_dims: Tuple[int, ...] = (128, 128, 64)
    dropout: float = 0.12
    seed: int = 42
    device: str = "cpu"
    checkpoint_path: Optional[Union[str, Path]] = None


@dataclass
class PGDATHistory:
    """Epoch-level clean and adversarial training telemetry for PGD-AT."""

    train_total_loss: List[float] = field(default_factory=list)
    val_clean_acc: List[float] = field(default_factory=list)
    val_robust_acc: List[float] = field(default_factory=list)
    composite_score: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_composite_score: float = -float("inf")


def train_pgd_adversarial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    schema: DomainFeatureSchema,
    config: Optional[PGDATConfig] = None,
) -> Tuple[RobustIDSNet, PGDATHistory]:
    """Train `RobustIDSNet` via domain-constrained Min-Max PGD Adversarial Training.

    Solves:
        min_theta E_{(x, y)} [ (1 - lambda) L(theta, x, y)
                               + lambda * max_{delta in S(x, eps)} L(theta, x + delta, y) ]
    """
    cfg = config if config is not None else PGDATConfig()
    set_deterministic_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = RobustIDSNet(
        input_dim=schema.num_features,
        hidden_dims=cfg.hidden_dims,
        num_classes=schema.num_classes,
        dropout=cfg.dropout,
    ).to(device)

    mask_t, xmin_t, xmax_t = schema.to_torch(device)
    class_weights = compute_balanced_class_weights(
        y_train, num_classes=schema.num_classes, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.learning_rate * 0.05
    )

    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    drop_last = (X_train.shape[0] % cfg.batch_size) == 1
    train_ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=gen,
        drop_last=drop_last,
    )

    history = PGDATHistory()
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(cfg.epochs):
        # Linear epsilon curriculum over warmup_epochs stabilizes BatchNorm and representation learning
        if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
            curr_eps = cfg.train_eps * float(epoch + 1) / float(cfg.warmup_epochs)
        else:
            curr_eps = cfg.train_eps

        model.train()
        epoch_loss = 0.0
        seen = 0

        for step_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            # Inner maximization: pgd_attack automatically runs in eval() mode
            # so intermediate gradient-ascent trajectories do not pollute BatchNorm stats
            x_adv = pgd_attack(
                model=model,
                x=xb,
                y=yb,
                eps=curr_eps,
                mutability_mask=mask_t,
                x_min=xmin_t,
                x_max=xmax_t,
                steps=cfg.pgd_steps,
                random_start=True,
                criterion=criterion,
                seed=cfg.seed + epoch * 1000 + step_idx,
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)

            logits_clean = model(xb)
            loss_clean = criterion(logits_clean, yb)

            # Evaluate adversarial forward pass without overwriting clean BN running stats
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.track_running_stats = False
            logits_adv = model(x_adv)
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.track_running_stats = True

            loss_adv = criterion(logits_adv, yb)
            total_loss = (
                (1.0 - cfg.adv_mix_weight) * loss_clean
                + cfg.adv_mix_weight * loss_adv
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(total_loss.item()) * xb.shape[0]
            seen += int(xb.shape[0])

        scheduler.step()

        clean_val = evaluate_classifier(
            model, X_val, y_val, criterion=criterion, device=device
        )
        rob_val = evaluate_whitebox_robustness(
            model=model,
            X=X_val,
            y=y_val,
            schema=schema,
            epsilons=[cfg.train_eps],
            attack="pgd",
            pgd_steps=max(7, cfg.pgd_steps),
            device=device,
            seed=cfg.seed,
        )[0]

        comp_score = 0.42 * clean_val["accuracy"] + 0.58 * rob_val.accuracy
        history.train_total_loss.append(epoch_loss / max(seen, 1))
        history.val_clean_acc.append(clean_val["accuracy"])
        history.val_robust_acc.append(rob_val.accuracy)
        history.composite_score.append(comp_score)

        # Require warmup completion before locking best robust checkpoint
        if epoch >= max(cfg.warmup_epochs - 1, 0):
            if comp_score > (history.best_composite_score + 1e-4):
                history.best_composite_score = comp_score
                history.best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= cfg.patience:
                    break

    model.load_state_dict(best_state)
    model.eval()

    if cfg.checkpoint_path is not None:
        ckpt_path = Path(cfg.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": model.input_dim,
                "hidden_dims": model.hidden_dims,
                "num_classes": model.num_classes,
                "dropout": model.dropout,
                "train_eps": cfg.train_eps,
                "best_epoch": history.best_epoch,
                "best_composite_score": history.best_composite_score,
            },
            ckpt_path,
        )

    return model, history


class FeatureSqueezer:
    """Inference-time Feature Squeezing & Input Quantization defense.

    Neutralizes small-norm gradient perturbations on continuous flow features (M_j = 1)
    without retraining via:
      1. Optional continuous feature smoothing (`gaussian` Nadaraya-Watson kernel
         smoothing, `median` local k-NN / filter smoothing, or `none`) restricted
         strictly to mutable continuous telemetry coordinates.
      2. Uniform b-bit precision quantization into 2^b - 1 discrete levels in [x_min, x_max].
      3. Strict preservation of all immutable discrete protocol features (M_j = 0).
    """

    def __init__(
        self,
        schema: DomainFeatureSchema,
        bit_depth: int = 4,
        smoothing: str = "gaussian",
        gaussian_sigma: float = 0.24,
        smooth_blend: float = 0.48,
        median_k: int = 9,
        max_ref_samples: int = 1024,
    ) -> None:
        if bit_depth < 1 or bit_depth > 16:
            raise ValueError(f"bit_depth must be in [1, 16], got {bit_depth}")
        if smoothing not in {"none", "gaussian", "median"}:
            raise ValueError(f"smoothing must be 'none', 'gaussian', or 'median', got {smoothing}")

        self.schema = schema
        self.bit_depth = int(bit_depth)
        self.levels = float((1 << self.bit_depth) - 1)
        self.smoothing = smoothing
        self.gaussian_sigma = float(gaussian_sigma)
        self.smooth_blend = float(smooth_blend)
        self.median_k = int(median_k)
        self.max_ref_samples = int(max_ref_samples)
        self._ref_all: Optional[np.ndarray] = None
        self._ref_mut_u: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, seed: int = 42) -> "FeatureSqueezer":
        """Store a subsampled clean reference bank for metric-space Gaussian/median smoothing."""
        x_tr = np.asarray(X_train, dtype=np.float32)
        if x_tr.shape[0] > self.max_ref_samples:
            rng = np.random.Generator(np.random.PCG64(seed))
            idx = rng.choice(x_tr.shape[0], size=self.max_ref_samples, replace=False)
            x_ref = x_tr[idx].copy()
        else:
            x_ref = x_tr.copy()

        mut_idx = self.schema.mutable_indices
        xmin_m = self.schema.x_min[mut_idx]
        xmax_m = self.schema.x_max[mut_idx]
        span_m = np.maximum(xmax_m - xmin_m, 1e-8)

        self._ref_all = x_ref
        self._ref_mut_u = np.clip(
            (x_ref[:, mut_idx] - xmin_m[None, :]) / span_m[None, :], 0.0, 1.0
        )
        return self

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Apply continuous smoothing + b-bit quantization to a NumPy feature matrix."""
        x_arr = np.asarray(X, dtype=np.float32).copy()
        mut_idx = self.schema.mutable_indices
        imm_idx = self.schema.immutable_indices

        if mut_idx.size == 0:
            return x_arr

        xmin_m = self.schema.x_min[mut_idx]
        xmax_m = self.schema.x_max[mut_idx]
        span_m = np.maximum(xmax_m - xmin_m, 1e-8)

        u_mut = np.clip((x_arr[:, mut_idx] - xmin_m[None, :]) / span_m[None, :], 0.0, 1.0)

        if self.smoothing == "gaussian" and self.gaussian_sigma > 0.0:
            if self._ref_all is not None and self._ref_mut_u is not None:
                # Nadaraya-Watson isotropic Gaussian kernel smoothing in joint flow space
                sq_norms_x = np.sum(x_arr ** 2, axis=1, keepdims=True)
                sq_norms_r = np.sum(self._ref_all ** 2, axis=1, keepdims=True).T
                sq_dists = np.maximum(
                    sq_norms_x + sq_norms_r - 2.0 * (x_arr @ self._ref_all.T), 0.0
                )
                logits_k = -sq_dists / (2.0 * (self.gaussian_sigma ** 2))
                logits_k -= np.max(logits_k, axis=1, keepdims=True)
                weights = np.exp(logits_k)
                weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
                u_smooth = weights @ self._ref_mut_u
                u_mut = (1.0 - self.smooth_blend) * u_mut + self.smooth_blend * u_smooth
            elif u_mut.shape[1] > 1:
                smoothed = gaussian_filter1d(
                    u_mut, sigma=self.gaussian_sigma, axis=1, mode="nearest"
                )
                u_mut = (1.0 - 0.15) * u_mut + 0.15 * smoothed
        elif self.smoothing == "median":
            if self._ref_all is not None and self._ref_mut_u is not None:
                sq_norms_x = np.sum(x_arr ** 2, axis=1, keepdims=True)
                sq_norms_r = np.sum(self._ref_all ** 2, axis=1, keepdims=True).T
                sq_dists = np.maximum(
                    sq_norms_x + sq_norms_r - 2.0 * (x_arr @ self._ref_all.T), 0.0
                )
                k = min(self.median_k, self._ref_all.shape[0])
                nn_idx = np.argpartition(sq_dists, kth=k - 1, axis=1)[:, :k]
                u_med = np.median(self._ref_mut_u[nn_idx], axis=1)
                u_mut = (1.0 - self.smooth_blend) * u_mut + self.smooth_blend * u_med
            elif u_mut.shape[1] >= 3:
                med_filtered = median_filter(u_mut, size=(1, 3), mode="nearest")
                u_mut = 0.85 * u_mut + 0.15 * med_filtered

        # Uniform b-bit rounding in normalized [0, 1] coordinates
        u_quant = np.round(u_mut * self.levels) / self.levels
        x_quant_mut = xmin_m[None, :] + np.clip(u_quant, 0.0, 1.0) * span_m[None, :]

        x_out = x_arr.copy()
        x_out[:, mut_idx] = np.clip(x_quant_mut, xmin_m[None, :], xmax_m[None, :])
        if imm_idx.size > 0:
            x_out[:, imm_idx] = x_arr[:, imm_idx]
        return x_out.astype(np.float32)

    def transform(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply feature squeezing while preserving input container type and device."""
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            squeezed_np = self.transform_numpy(x_np)
            return torch.as_tensor(squeezed_np, dtype=x.dtype, device=x.device)
        return self.transform_numpy(x)

    @torch.no_grad()
    def predict(self, model: RobustIDSNet, x: torch.Tensor) -> torch.Tensor:
        """Run inference through `model` after squeezing continuous flow inputs."""
        x_squeezed = self.transform(x)
        assert isinstance(x_squeezed, torch.Tensor)
        return model.predict(x_squeezed)

    @torch.no_grad()
    def disagreement_score(self, model: RobustIDSNet, x: torch.Tensor) -> torch.Tensor:
        """Compute Xu et al. (2018) L_1 softmax divergence ||p(x) - p(squeeze(x))||_1."""
        p_raw = model.predict_proba(x)
        x_squeezed = self.transform(x)
        assert isinstance(x_squeezed, torch.Tensor)
        p_sq = model.predict_proba(x_squeezed)
        return torch.sum(torch.abs(p_raw - p_sq), dim=-1)


class MahalanobisOODDetector:
    """Manifold & Predictive Entropy Out-of-Distribution / Evasion Detector.

    Fits class-conditional Gaussian distributions (mean vectors mu_c and shrinkage-
    regularized tied precision matrix Sigma^{-1}) on penultimate-layer embeddings z
    of clean training data (plus optional input-space covariance manifold), scoring
    test inputs via calibrated Mahalanobis distance + normalized Shannon entropy.
    """

    def __init__(
        self,
        target_fpr: float = 0.04,
        penultimate_weight: float = 0.50,
        input_manifold_weight: float = 0.25,
        entropy_weight: float = 0.25,
        ridge_eps: float = 1e-4,
    ) -> None:
        if not (0.0 < target_fpr < 0.5):
            raise ValueError(f"target_fpr must be in (0, 0.5), got {target_fpr}")
        self.target_fpr = float(target_fpr)
        self.penultimate_weight = float(penultimate_weight)
        self.input_manifold_weight = float(input_manifold_weight)
        self.entropy_weight = float(entropy_weight)
        self.ridge_eps = float(ridge_eps)

        self.class_means_: Optional[np.ndarray] = None        # shape (C, m)
        self.precision_: Optional[np.ndarray] = None          # shape (m, m)
        self.input_means_: Optional[np.ndarray] = None        # shape (C, d)
        self.input_precision_: Optional[np.ndarray] = None    # shape (d, d)

        # Clean validation standardization parameters
        self._z_stats: Dict[str, Tuple[float, float]] = {}
        self.threshold_: float = float("inf")
        self._fitted: bool = False

    @staticmethod
    @torch.no_grad()
    def _extract_embeddings_and_probs(
        model: RobustIDSNet,
        X: Union[np.ndarray, torch.Tensor],
        batch_size: int = 512,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        was_training = model.training
        model.eval()
        dev = torch.device(device)

        x_t = torch.as_tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x_t), batch_size=batch_size, shuffle=False)

        emb_list: List[np.ndarray] = []
        prob_list: List[np.ndarray] = []
        for (xb,) in loader:
            xb = xb.to(dev)
            logits, z = model(xb, return_features=True)
            probs = F.softmax(logits, dim=-1)
            emb_list.append(z.cpu().numpy())
            prob_list.append(probs.cpu().numpy())

        if was_training:
            model.train()

        return np.concatenate(emb_list, axis=0), np.concatenate(prob_list, axis=0)

    def _fit_tied_gaussian(
        self, feats: np.ndarray, y: np.ndarray, num_classes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        dim = feats.shape[1]
        means = np.zeros((num_classes, dim), dtype=np.float64)
        residuals = np.zeros_like(feats, dtype=np.float64)

        for c in range(num_classes):
            mask_c = (y == c)
            if np.any(mask_c):
                means[c] = np.mean(feats[mask_c], axis=0)
                residuals[mask_c] = feats[mask_c] - means[c][None, :]

        # Shrinkage covariance estimation ensures strictly positive-definite precision
        lw = LedoitWolf(assume_centered=True)
        lw.fit(residuals)
        cov = lw.covariance_ + self.ridge_eps * np.eye(dim, dtype=np.float64)
        precision = np.linalg.inv(cov)
        # Symmetrize to remove floating-point asymmetry
        precision = 0.5 * (precision + precision.T)
        return means, precision

    @staticmethod
    def _class_mahalanobis_matrix(
        feats: np.ndarray, means: np.ndarray, precision: np.ndarray
    ) -> np.ndarray:
        """Compute squared-root Mahalanobis distances D_M(z, mu_c) of shape (N, C)."""
        n_samples = feats.shape[0]
        num_classes = means.shape[0]
        dists = np.zeros((n_samples, num_classes), dtype=np.float64)
        for c in range(num_classes):
            diff = feats.astype(np.float64) - means[c][None, :]
            quad = np.einsum("ni,ij,nj->n", diff, precision, diff)
            dists[:, c] = np.sqrt(np.maximum(quad, 0.0))
        return dists

    @staticmethod
    def _normalized_entropy(probs: np.ndarray) -> np.ndarray:
        num_classes = probs.shape[1]
        p_safe = np.clip(probs.astype(np.float64), 1e-12, 1.0)
        ent = -np.sum(p_safe * np.log(p_safe), axis=1)
        return ent / max(np.log(float(num_classes)), 1e-8)

    def fit(
        self,
        model: RobustIDSNet,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> "MahalanobisOODDetector":
        """Fit class-conditional Gaussian manifold parameters and calibrate threshold."""
        x_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.int64)
        num_classes = model.num_classes

        z_tr, _ = self._extract_embeddings_and_probs(model, x_tr, device=device)
        self.class_means_, self.precision_ = self._fit_tied_gaussian(z_tr, y_tr, num_classes)
        self.input_means_, self.input_precision_ = self._fit_tied_gaussian(x_tr, y_tr, num_classes)

        cal_x = np.asarray(X_val, dtype=np.float32) if X_val is not None else x_tr
        z_cal, probs_cal = self._extract_embeddings_and_probs(model, cal_x, device=device)

        d_penult = np.min(
            self._class_mahalanobis_matrix(z_cal, self.class_means_, self.precision_), axis=1
        )
        pred_cls = np.argmax(probs_cal, axis=1)
        d_input_all = self._class_mahalanobis_matrix(
            cal_x, self.input_means_, self.input_precision_
        )
        d_input = d_input_all[np.arange(cal_x.shape[0]), pred_cls]
        ent_cal = self._normalized_entropy(probs_cal)

        self._z_stats = {
            "penult": (float(np.mean(d_penult)), float(max(np.std(d_penult), 1e-6))),
            "input": (float(np.mean(d_input)), float(max(np.std(d_input), 1e-6))),
            "entropy": (float(np.mean(ent_cal)), float(max(np.std(ent_cal), 1e-6))),
        }

        self._fitted = True
        val_scores = self.anomaly_score(model, cal_x, device=device)
        self.threshold_ = float(np.quantile(val_scores, 1.0 - self.target_fpr))
        return self

    def mahalanobis_distance(
        self,
        model: RobustIDSNet,
        X: Union[np.ndarray, torch.Tensor],
        device: Union[str, torch.device] = "cpu",
    ) -> np.ndarray:
        """Compute minimum penultimate-layer class-conditional Mahalanobis distance M(x)."""
        if not self._fitted or self.class_means_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisOODDetector must be fitted before scoring.")
        z, _ = self._extract_embeddings_and_probs(model, X, device=device)
        d_mat = self._class_mahalanobis_matrix(z, self.class_means_, self.precision_)
        return np.min(d_mat, axis=1).astype(np.float32)

    def score_components(
        self,
        model: RobustIDSNet,
        X: Union[np.ndarray, torch.Tensor],
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, np.ndarray]:
        """Return penultimate Mahalanobis, input Mahalanobis, predictive entropy, and composite score."""
        if (
            not self._fitted
            or self.class_means_ is None
            or self.precision_ is None
            or self.input_means_ is None
            or self.input_precision_ is None
        ):
            raise RuntimeError("MahalanobisOODDetector must be fitted before scoring.")

        x_np = (
            X.detach().cpu().numpy().astype(np.float32)
            if isinstance(X, torch.Tensor)
            else np.asarray(X, dtype=np.float32)
        )
        z, probs = self._extract_embeddings_and_probs(model, x_np, device=device)

        d_penult = np.min(
            self._class_mahalanobis_matrix(z, self.class_means_, self.precision_), axis=1
        )
        pred_cls = np.argmax(probs, axis=1)
        d_input_all = self._class_mahalanobis_matrix(
            x_np, self.input_means_, self.input_precision_
        )
        d_input = d_input_all[np.arange(x_np.shape[0]), pred_cls]
        ent = self._normalized_entropy(probs)

        mu_p, sd_p = self._z_stats["penult"]
        mu_i, sd_i = self._z_stats["input"]
        mu_e, sd_e = self._z_stats["entropy"]

        z_p = (d_penult - mu_p) / sd_p
        z_i = (d_input - mu_i) / sd_i
        z_e = (ent - mu_e) / sd_e

        composite = (
            self.penultimate_weight * z_p
            + self.input_manifold_weight * z_i
            + self.entropy_weight * z_e
        )

        return {
            "mahalanobis_penult": d_penult.astype(np.float32),
            "mahalanobis_input": d_input.astype(np.float32),
            "predictive_entropy": ent.astype(np.float32),
            "composite_score": composite.astype(np.float32),
            "probabilities": probs.astype(np.float32),
        }

    def anomaly_score(
        self,
        model: RobustIDSNet,
        X: Union[np.ndarray, torch.Tensor],
        device: Union[str, torch.device] = "cpu",
    ) -> np.ndarray:
        """Compute composite manifold + entropy anomaly score S(x)."""
        return self.score_components(model, X, device=device)["composite_score"]

    def is_rejected(
        self,
        model: RobustIDSNet,
        X: Union[np.ndarray, torch.Tensor],
        device: Union[str, torch.device] = "cpu",
    ) -> np.ndarray:
        """Boolean mask indicating inputs rejected as anomalous/off-manifold (S(x) > threshold)."""
        scores = self.anomaly_score(model, X, device=device)
        return scores > self.threshold_


@dataclass
class HardenedPrediction:
    """Inference output of the layered defense stack with reject option."""

    predictions_with_reject: np.ndarray  # -1 (REJECT_LABEL) where rejected, else class index
    raw_predictions: np.ndarray          # underlying argmax class index in {0, ..., C-1}
    is_rejected: np.ndarray              # bool mask (N,)
    anomaly_scores: np.ndarray           # composite anomaly score S(x)
    mahalanobis_distances: np.ndarray    # penultimate Mahalanobis distance M(x)
    predictive_entropy: np.ndarray       # normalized Shannon entropy H(x)
    probabilities: np.ndarray            # softmax probabilities (N, C)


class HardenedIDSStack:
    """Full Layered Defense Stack combining PGD-AT, optional Feature Squeezing, and Mahalanobis OOD."""

    def __init__(
        self,
        model: RobustIDSNet,
        detector: MahalanobisOODDetector,
        squeezer: Optional[FeatureSqueezer] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.model = model
        self.detector = detector
        self.squeezer = squeezer
        self.device = torch.device(device)

    def predict_detailed(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> HardenedPrediction:
        """Evaluate anomaly detector on raw inputs and classify (optionally squeezed) flows."""
        x_np = (
            X.detach().cpu().numpy().astype(np.float32)
            if isinstance(X, torch.Tensor)
            else np.asarray(X, dtype=np.float32)
        )

        # Score off-manifold deviation prior to quantization so subtle covariance violations are caught
        comps = self.detector.score_components(self.model, x_np, device=self.device)
        scores = comps["composite_score"]
        rejected = scores > self.detector.threshold_

        if self.squeezer is not None:
            x_cls = self.squeezer.transform_numpy(x_np)
            probs = self.detector._extract_embeddings_and_probs(
                self.model, x_cls, device=self.device
            )[1]
        else:
            probs = comps["probabilities"]

        raw_preds = np.argmax(probs, axis=1).astype(np.int64)
        preds_with_reject = np.where(rejected, REJECT_LABEL, raw_preds).astype(np.int64)

        return HardenedPrediction(
            predictions_with_reject=preds_with_reject,
            raw_predictions=raw_preds,
            is_rejected=rejected,
            anomaly_scores=scores,
            mahalanobis_distances=comps["mahalanobis_penult"],
            predictive_entropy=comps["predictive_entropy"],
            probabilities=probs,
        )


def compute_ood_auroc(
    in_dist_scores: np.ndarray,
    out_dist_scores: np.ndarray,
) -> float:
    """Compute AUROC for distinguishing clean in-distribution flows (0) from OOD/adversarial flows (1)."""
    y_true = np.concatenate(
        [
            np.zeros(len(in_dist_scores), dtype=np.int64),
            np.ones(len(out_dist_scores), dtype=np.int64),
        ]
    )
    y_scores = np.concatenate([in_dist_scores, out_dist_scores]).astype(np.float64)
    return float(roc_auc_score(y_true, y_scores))
