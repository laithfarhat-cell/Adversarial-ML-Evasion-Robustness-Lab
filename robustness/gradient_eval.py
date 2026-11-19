"""Domain-Aware White-Box Gradient Robustness Evaluation Harness.

Implements mathematically transparent first-order evasion algorithms from scratch in PyTorch:
  1. Masked & Bounded Fast Gradient Sign Method (FGSM):
       x_adv = Clip_{[x_min, x_max]}( x + eps * (M (*) sign(grad_x L(theta, x, y))) )
  2. Masked & Bounded Projected Gradient Descent (PGD):
       Iterative multi-step gradient ascent with uniform initialization inside the
       masked L_inf epsilon-ball and exact coordinate-wise projection onto valid
       physical telemetry bounds [x_min, x_max] and protocol mutability mask M in {0, 1}^d.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from data.dataset_loader import DomainFeatureSchema


def _prepare_bounds_and_mask(
    x: torch.Tensor,
    mutability_mask: Union[np.ndarray, torch.Tensor],
    x_min: Union[float, np.ndarray, torch.Tensor],
    x_max: Union[float, np.ndarray, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x.device
    dtype = x.dtype
    mask_t = torch.as_tensor(mutability_mask, dtype=dtype, device=device).view(1, -1)
    if isinstance(x_min, (int, float)):
        xmin_t = torch.full((1, x.shape[1]), float(x_min), dtype=dtype, device=device)
    else:
        xmin_t = torch.as_tensor(x_min, dtype=dtype, device=device).view(1, -1)
    if isinstance(x_max, (int, float)):
        xmax_t = torch.full((1, x.shape[1]), float(x_max), dtype=dtype, device=device)
    else:
        xmax_t = torch.as_tensor(x_max, dtype=dtype, device=device).view(1, -1)
    immutable_cols = (mask_t.squeeze(0) == 0.0)
    return mask_t, xmin_t, xmax_t, immutable_cols


def project_domain_perturbation(
    x_candidate: torch.Tensor,
    x_orig: torch.Tensor,
    eps: float,
    mutability_mask: Union[np.ndarray, torch.Tensor],
    x_min: Union[float, np.ndarray, torch.Tensor] = 0.0,
    x_max: Union[float, np.ndarray, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """Euclidean/L_inf exact coordinate-wise projection onto the feasible evasion polytope:
        S(x, eps) = { x' in [x_min, x_max] : ||x' - x||_inf <= eps and x'_{M=0} == x_{M=0} }.
    """
    mask_t, xmin_t, xmax_t, immutable_cols = _prepare_bounds_and_mask(
        x_orig, mutability_mask, x_min, x_max
    )
    delta = torch.clamp(x_candidate - x_orig, min=-float(eps), max=float(eps))
    delta = delta * mask_t
    x_proj = torch.max(torch.min(x_orig + delta, xmax_t), xmin_t)
    if torch.any(immutable_cols):
        x_proj = x_proj.clone()
        x_proj[:, immutable_cols] = x_orig[:, immutable_cols]
    return x_proj


def verify_perturbation_constraints(
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    eps: float,
    mutability_mask: Union[np.ndarray, torch.Tensor],
    x_min: Union[float, np.ndarray, torch.Tensor] = 0.0,
    x_max: Union[float, np.ndarray, torch.Tensor] = 1.0,
    atol: float = 1e-6,
    raise_on_violation: bool = True,
) -> Dict[str, Union[bool, float]]:
    """Mathematically verify mutability mask invariance, L_inf budget, and domain bounds."""
    _, xmin_t, xmax_t, immutable_cols = _prepare_bounds_and_mask(
        x_orig, mutability_mask, x_min, x_max
    )

    with torch.no_grad():
        delta = x_adv - x_orig
        max_imm_delta = (
            float(torch.max(torch.abs(delta[:, immutable_cols])).item())
            if torch.any(immutable_cols)
            else 0.0
        )
        max_linf_delta = float(torch.max(torch.abs(delta)).item()) if delta.numel() > 0 else 0.0

        low_viol = float(torch.max(torch.clamp(xmin_t - x_adv, min=0.0)).item())
        high_viol = float(torch.max(torch.clamp(x_adv - xmax_t, min=0.0)).item())
        max_bound_viol = max(low_viol, high_viol)
        finite_ok = bool(torch.all(torch.isfinite(x_adv)).item())

    immutable_ok = max_imm_delta == 0.0
    linf_ok = max_linf_delta <= (float(eps) + atol)
    bounds_ok = max_bound_viol <= atol
    all_ok = bool(immutable_ok and linf_ok and bounds_ok and finite_ok)

    report: Dict[str, Union[bool, float]] = {
        "valid": all_ok,
        "immutable_preserved": immutable_ok,
        "within_linf_ball": linf_ok,
        "within_domain_bounds": bounds_ok,
        "is_finite": finite_ok,
        "max_immutable_delta": max_imm_delta,
        "max_linf_delta": max_linf_delta,
        "max_bound_violation": max_bound_viol,
    }

    if raise_on_violation and not all_ok:
        raise ValueError(f"Domain perturbation constraint violation detected: {report}")

    return report


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    mutability_mask: Union[np.ndarray, torch.Tensor],
    x_min: Union[float, np.ndarray, torch.Tensor] = 0.0,
    x_max: Union[float, np.ndarray, torch.Tensor] = 1.0,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Domain-constrained Fast Gradient Sign Method (FGSM):
        x_adv = Clip_{[x_min, x_max]}( x + eps * (M (*) sign(grad_x L(theta, x, y))) )
    """
    if eps < 0.0:
        raise ValueError(f"Perturbation budget eps must be >= 0.0, got {eps}")

    mask_t, xmin_t, xmax_t, immutable_cols = _prepare_bounds_and_mask(
        x, mutability_mask, x_min, x_max
    )

    if eps == 0.0:
        x_clean = torch.max(torch.min(x.detach().clone(), xmax_t), xmin_t)
        if torch.any(immutable_cols):
            x_clean[:, immutable_cols] = x[:, immutable_cols]
        return x_clean

    loss_fn = criterion if criterion is not None else nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()

    try:
        with torch.enable_grad():
            x_in = x.detach().clone().requires_grad_(True)
            logits = model(x_in)
            loss = loss_fn(logits, y)
            grad_x = torch.autograd.grad(loss, x_in, retain_graph=False, create_graph=False)[0]

        signed_step = float(eps) * (mask_t * torch.sign(grad_x.detach()))
        x_adv = torch.max(torch.min(x.detach() + signed_step, xmax_t), xmin_t)
        if torch.any(immutable_cols):
            x_adv[:, immutable_cols] = x.detach()[:, immutable_cols]
    finally:
        if was_training:
            model.train()

    return x_adv.detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    mutability_mask: Union[np.ndarray, torch.Tensor],
    x_min: Union[float, np.ndarray, torch.Tensor] = 0.0,
    x_max: Union[float, np.ndarray, torch.Tensor] = 1.0,
    steps: int = 20,
    alpha: Optional[float] = None,
    random_start: bool = True,
    restarts: int = 1,
    criterion: Optional[nn.Module] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Domain-constrained Projected Gradient Descent (PGD) under L_inf budget.

    Performs `steps` iterations of masked sign-gradient ascent with uniform random
    initialization inside the feasible epsilon-ball and exact projection back onto
    [x_min, x_max] and protocol mutability mask M at every iteration.
    """
    if eps < 0.0:
        raise ValueError(f"Perturbation budget eps must be >= 0.0, got {eps}")
    if steps < 1:
        raise ValueError(f"PGD steps must be >= 1, got {steps}")
    if restarts < 1:
        raise ValueError(f"PGD restarts must be >= 1, got {restarts}")

    mask_t, xmin_t, xmax_t, immutable_cols = _prepare_bounds_and_mask(
        x, mutability_mask, x_min, x_max
    )
    x_orig = x.detach()

    if eps == 0.0:
        x_clean = torch.max(torch.min(x_orig.clone(), xmax_t), xmin_t)
        if torch.any(immutable_cols):
            x_clean[:, immutable_cols] = x_orig[:, immutable_cols]
        return x_clean

    step_size = float(alpha) if alpha is not None else max(eps / 4.0, 2.5 * eps / float(steps))
    per_sample_ce = nn.CrossEntropyLoss(reduction="none")
    scalar_loss_fn = criterion if criterion is not None else nn.CrossEntropyLoss()

    gen: Optional[torch.Generator] = None
    if seed is not None:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))

    was_training = model.training
    model.eval()

    best_adv = x_orig.clone()
    best_loss = torch.full((x_orig.shape[0],), -float("inf"), device=x.device, dtype=x.dtype)

    try:
        for _ in range(restarts):
            if random_start:
                noise = torch.empty_like(x_orig).uniform_(-float(eps), float(eps), generator=gen)
                x_curr = project_domain_perturbation(
                    x_orig + noise * mask_t, x_orig, eps, mask_t, xmin_t, xmax_t
                )
            else:
                x_curr = x_orig.clone()

            for _ in range(steps):
                with torch.enable_grad():
                    x_in = x_curr.detach().requires_grad_(True)
                    logits = model(x_in)
                    loss = scalar_loss_fn(logits, y)
                    grad_x = torch.autograd.grad(
                        loss, x_in, retain_graph=False, create_graph=False
                    )[0]

                x_next = x_curr + step_size * (mask_t * torch.sign(grad_x.detach()))
                x_curr = project_domain_perturbation(
                    x_next, x_orig, eps, mask_t, xmin_t, xmax_t
                )

            with torch.no_grad():
                final_logits = model(x_curr)
                sample_losses = per_sample_ce(final_logits, y)
                improved = sample_losses > best_loss
                best_loss = torch.where(improved, sample_losses, best_loss)
                best_adv[improved] = x_curr[improved]

        if torch.any(immutable_cols):
            best_adv[:, immutable_cols] = x_orig[:, immutable_cols]
    finally:
        if was_training:
            model.train()

    return best_adv.detach()


@dataclass
class RobustnessEvalResult:
    """Structured metrics at a single perturbation budget epsilon."""

    epsilon: float
    attack: str
    accuracy: float
    macro_f1: float
    mean_linf_delta: float
    max_immutable_delta: float
    constraints_valid: bool


def evaluate_whitebox_robustness(
    model: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    schema: DomainFeatureSchema,
    epsilons: List[float],
    attack: str = "pgd",
    pgd_steps: int = 20,
    batch_size: int = 256,
    predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: Union[str, torch.device] = "cpu",
    seed: int = 42,
) -> List[RobustnessEvalResult]:
    """Evaluate classifier or defense stack across L_inf perturbation budgets."""
    attack_lower = attack.lower()
    if attack_lower not in {"fgsm", "pgd"}:
        raise ValueError(f"Unsupported attack='{attack}'. Expected 'fgsm' or 'pgd'.")

    dev = torch.device(device)
    model.eval()
    mask_t, xmin_t, xmax_t = schema.to_torch(dev)

    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    results: List[RobustnessEvalResult] = []
    for eps in epsilons:
        preds_list: List[np.ndarray] = []
        targets_list: List[np.ndarray] = []
        linf_vals: List[float] = []
        max_imm = 0.0
        all_valid = True

        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(dev)
            yb = yb.to(dev)

            if attack_lower == "fgsm":
                x_adv = fgsm_attack(
                    model=model,
                    x=xb,
                    y=yb,
                    eps=eps,
                    mutability_mask=mask_t,
                    x_min=xmin_t,
                    x_max=xmax_t,
                )
            else:
                x_adv = pgd_attack(
                    model=model,
                    x=xb,
                    y=yb,
                    eps=eps,
                    mutability_mask=mask_t,
                    x_min=xmin_t,
                    x_max=xmax_t,
                    steps=pgd_steps,
                    seed=seed + batch_idx,
                )

            check = verify_perturbation_constraints(
                x_orig=xb,
                x_adv=x_adv,
                eps=eps,
                mutability_mask=mask_t,
                x_min=xmin_t,
                x_max=xmax_t,
                raise_on_violation=True,
            )
            all_valid = all_valid and bool(check["valid"])
            max_imm = max(max_imm, float(check["max_immutable_delta"]))
            per_sample_linf = torch.max(torch.abs(x_adv - xb), dim=1).values
            linf_vals.extend(per_sample_linf.cpu().tolist())

            with torch.no_grad():
                if predict_fn is not None:
                    batch_preds = predict_fn(x_adv)
                else:
                    batch_preds = torch.argmax(model(x_adv), dim=-1)

            preds_list.append(batch_preds.detach().cpu().numpy())
            targets_list.append(yb.detach().cpu().numpy())

        y_pred = np.concatenate(preds_list, axis=0)
        y_true = np.concatenate(targets_list, axis=0)

        results.append(
            RobustnessEvalResult(
                epsilon=float(eps),
                attack=attack_lower,
                accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                mean_linf_delta=float(np.mean(linf_vals)) if linf_vals else 0.0,
                max_immutable_delta=max_imm,
                constraints_valid=all_valid,
            )
        )

    return results


@dataclass
class FeatureVulnerabilityReport:
    """Per-feature white-box input Jacobian sensitivity and perturbation diagnostics."""

    feature_names: List[str]
    mutability_mask: np.ndarray
    raw_grad_sensitivity: np.ndarray
    effective_grad_sensitivity: np.ndarray
    mean_pgd_perturbation: np.ndarray

    def ranked_indices(self) -> np.ndarray:
        """Indices sorted descending by effective (masked) gradient sensitivity."""
        return np.argsort(-self.effective_grad_sensitivity)


def analyze_feature_vulnerability(
    model: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    schema: DomainFeatureSchema,
    eps: float = 0.10,
    pgd_steps: int = 15,
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
) -> FeatureVulnerabilityReport:
    """Quantify per-feature input gradient sensitivity E[|dL/dx_j|] and PGD distortion."""
    dev = torch.device(device)
    model.eval()
    mask_t, xmin_t, xmax_t = schema.to_torch(dev)
    per_sample_ce = nn.CrossEntropyLoss(reduction="sum")

    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    grad_abs_sum = np.zeros(schema.num_features, dtype=np.float64)
    pgd_abs_sum = np.zeros(schema.num_features, dtype=np.float64)
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(dev)
        yb = yb.to(dev)

        with torch.enable_grad():
            x_in = xb.detach().clone().requires_grad_(True)
            logits = model(x_in)
            loss_sum = per_sample_ce(logits, yb)
            grad_x = torch.autograd.grad(loss_sum, x_in)[0].detach()

        x_adv = pgd_attack(
            model=model,
            x=xb,
            y=yb,
            eps=eps,
            mutability_mask=mask_t,
            x_min=xmin_t,
            x_max=xmax_t,
            steps=pgd_steps,
            seed=101,
        )

        grad_abs_sum += torch.sum(torch.abs(grad_x), dim=0).cpu().numpy()
        pgd_abs_sum += torch.sum(torch.abs(x_adv - xb), dim=0).cpu().numpy()
        total_n += int(xb.shape[0])

    raw_sens = (grad_abs_sum / max(total_n, 1)).astype(np.float32)
    eff_sens = (raw_sens * schema.mutability_mask).astype(np.float32)
    mean_pgd = (pgd_abs_sum / max(total_n, 1)).astype(np.float32)

    return FeatureVulnerabilityReport(
        feature_names=list(schema.feature_names),
        mutability_mask=schema.mutability_mask.copy(),
        raw_grad_sensitivity=raw_sens,
        effective_grad_sensitivity=eff_sens,
        mean_pgd_perturbation=mean_pgd,
    )
