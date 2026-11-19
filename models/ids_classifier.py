"""Modular PyTorch Intrusion Detection System (IDS) Classifier & Baseline Trainer.

Implements `RobustIDSNet`, a tabular Multi-Layer Perceptron composed of stacked
`Linear -> BatchNorm1d -> LeakyReLU -> Dropout` feature blocks and an accessible
penultimate representation space for manifold anomaly detection.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


def set_deterministic_seed(seed: int = 42) -> None:
    """Configure deterministic PRNG seeds across Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _TabularBlock(nn.Module):
    """Single modular tabular hidden block: Linear -> BatchNorm1d -> LeakyReLU -> Dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        negative_slope: float = 0.1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        if self.training and h.shape[0] == 1:
            # Safeguard singleton trailing batch during training
            h = F.batch_norm(
                h,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
                eps=self.bn.eps,
            )
        else:
            h = self.bn(h)
        h = self.act(h)
        return self.drop(h)


class RobustIDSNet(nn.Module):
    """Modular Multi-Layer Perceptron for Tabular Network-Flow Classification.

    Architecture:
        x -> [Linear -> BatchNorm1d -> LeakyReLU -> Dropout] x L -> z (penultimate) -> Linear -> logits
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: Sequence[int] = (128, 128, 64),
        num_classes: int = 4,
        dropout: float = 0.15,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must specify at least one hidden layer.")

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)
        self.negative_slope = float(negative_slope)

        blocks: List[nn.Module] = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            blocks.append(
                _TabularBlock(
                    in_features=prev_dim,
                    out_features=h_dim,
                    negative_slope=self.negative_slope,
                    dropout=self.dropout,
                )
            )
            prev_dim = h_dim

        self.encoder = nn.Sequential(*blocks)
        self.embedding_dim = prev_dim
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, a=self.negative_slope, nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate-layer latent embeddings z in R^{B x embedding_dim}."""
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute unnormalized class logits, optionally returning penultimate embeddings."""
        z = self.encoder(x)
        logits = self.classifier(z)
        if return_features:
            return logits, z
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated softmax class probabilities in eval mode."""
        was_training = self.training
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        if was_training:
            self.train()
        return probs

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices in eval mode."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)


def compute_balanced_class_weights(
    y: np.ndarray,
    num_classes: int = 4,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Compute inverse-frequency balanced class weights w_c = N / (C * N_c)."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes).astype(np.float64)
    total = float(np.sum(counts))
    weights = np.where(counts > 0, total / (num_classes * np.maximum(counts, 1.0)), 1.0)
    # Normalize mean weight to 1.0 for stable learning rate scaling
    weights = weights / np.mean(weights)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


@dataclass
class TrainingConfig:
    """Hyperparameter configuration for clean baseline IDS training."""

    epochs: int = 45
    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-4
    hidden_dims: Tuple[int, ...] = (128, 128, 64)
    dropout: float = 0.15
    seed: int = 42
    device: str = "cpu"
    checkpoint_path: Optional[Union[str, Path]] = None


@dataclass
class TrainingHistory:
    """Epoch-level telemetry recorded during model optimization."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    val_macro_f1: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


@torch.no_grad()
def evaluate_classifier(
    model: RobustIDSNet,
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    criterion: Optional[nn.Module] = None,
    batch_size: int = 512,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, float]:
    """Evaluate clean loss, accuracy, and macro F1-score of a classifier."""
    was_training = model.training
    model.eval()

    x_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    ds = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_count = 0
    preds_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []

    loss_fn = criterion if criterion is not None else nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += float(loss.item()) * xb.shape[0]
        total_count += int(xb.shape[0])
        preds_all.append(torch.argmax(logits, dim=-1).cpu().numpy())
        targets_all.append(yb.cpu().numpy())

    if was_training:
        model.train()

    y_pred = np.concatenate(preds_all, axis=0)
    y_true = np.concatenate(targets_all, axis=0)

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_baseline_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int = 4,
    config: Optional[TrainingConfig] = None,
) -> Tuple[RobustIDSNet, TrainingHistory]:
    """Train a clean baseline `RobustIDSNet` with class weighting, LR scheduling, and early stopping."""
    cfg = config if config is not None else TrainingConfig()
    set_deterministic_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = RobustIDSNet(
        input_dim=int(X_train.shape[1]),
        hidden_dims=cfg.hidden_dims,
        num_classes=num_classes,
        dropout=cfg.dropout,
    ).to(device)

    class_weights = compute_balanced_class_weights(y_train, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, cfg.patience // 3),
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

    history = TrainingHistory()
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.shape[0]
            seen += int(xb.shape[0])

        train_epoch_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_classifier(
            model, X_val, y_val, criterion=criterion, device=device
        )
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.train_loss.append(train_epoch_loss)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_metrics["accuracy"])
        history.val_macro_f1.append(val_metrics["macro_f1"])
        history.learning_rates.append(current_lr)

        if val_loss < (history.best_val_loss - cfg.min_delta):
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
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
                "best_epoch": history.best_epoch,
                "best_val_loss": history.best_val_loss,
            },
            ckpt_path,
        )

    return model, history
