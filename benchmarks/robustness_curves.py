"""Automated White-Box Robustness Benchmark & Visualization Suite.

CLI usage:
    python -m benchmarks.robustness_curves

Evaluates four defense configurations across L_inf budgets
eps in {0.0, 0.02, 0.05, 0.10, 0.15, 0.20}:
  1. Unhardened Baseline Model
  2. Baseline + Feature Squeezing
  3. Adversarially Trained Model (PGD-AT)
  4. Full Defense Stack (PGD-AT + Mahalanobis Reject Option)

Generates:
  - `results/accuracy_vs_epsilon.png` (Clean vs. Robust Accuracy tradeoff curves)
  - `results/feature_vulnerability.png` (Ranked network feature gradient sensitivity)
  - `results/benchmark_metrics.json` (Reproducible structured telemetry)
  - Formatted `rich` terminal summary tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from data.dataset_loader import (
    DomainFeatureSchema,
    FlowDatasetBundle,
    load_or_generate_dataset,
)
from defenses.hardening import (
    FeatureSqueezer,
    HardenedIDSStack,
    MahalanobisOODDetector,
    PGDATConfig,
    compute_ood_auroc,
    train_pgd_adversarial,
)
from models.ids_classifier import (
    RobustIDSNet,
    TrainingConfig,
    set_deterministic_seed,
    train_baseline_classifier,
)
from robustness.gradient_eval import (
    FeatureVulnerabilityReport,
    analyze_feature_vulnerability,
    fgsm_attack,
    pgd_attack,
    verify_perturbation_constraints,
)

DEFAULT_EPSILONS: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)


@dataclass
class BenchmarkConfig:
    """Configuration parameters for the end-to-end robustness benchmark."""

    n_samples: int = 12_000
    csv_path: Optional[str] = None
    baseline_epochs: int = 40
    pgdat_epochs: int = 40
    batch_size: int = 128
    eval_batch_size: int = 256
    pgd_eval_steps: int = 20
    pgd_train_steps: int = 10
    train_eps: float = 0.11
    adv_mix_weight: float = 0.65
    squeeze_bit_depth: int = 4
    mahalanobis_fpr: float = 0.04
    epsilons: Tuple[float, ...] = DEFAULT_EPSILONS
    seed: int = 42
    output_dir: str = "results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ConfigBenchmarkMetrics:
    """Complete evaluation telemetry for a single defense configuration."""

    name: str
    epsilons: List[float]
    pgd_accuracies: List[float]
    fgsm_accuracies: List[float]
    pgd_macro_f1s: List[float]
    clean_accuracy: float
    robust_acc_pgd_010: float
    robust_acc_fgsm_010: float
    clean_macro_f1: float
    robust_macro_f1_010: float
    ood_auroc: float
    rejection_rates_pgd: Optional[List[float]] = None
    selective_accuracies_pgd: Optional[List[float]] = None


@dataclass
class BenchmarkSuiteOutput:
    """Aggregated outputs of `run_full_benchmark`."""

    configs: List[ConfigBenchmarkMetrics]
    baseline_vulnerability: FeatureVulnerabilityReport
    pgdat_vulnerability: FeatureVulnerabilityReport
    constraint_summary: Dict[str, Union[bool, float, int]]
    accuracy_plot_path: Path
    vulnerability_plot_path: Path
    metrics_json_path: Path


def _generate_attack_batches(
    model: RobustIDSNet,
    X: np.ndarray,
    y: np.ndarray,
    schema: DomainFeatureSchema,
    eps: float,
    attack: str,
    pgd_steps: int,
    batch_size: int,
    device: Union[str, torch.device],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Union[bool, float]]]:
    """Generate domain-constrained FGSM or PGD samples across X and verify constraints."""
    dev = torch.device(device)
    mask_t, xmin_t, xmax_t = schema.to_torch(dev)
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    adv_chunks: List[np.ndarray] = []
    max_imm_delta = 0.0
    max_linf_delta = 0.0
    max_bound_viol = 0.0
    all_valid = True

    for b_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(dev)
        yb = yb.to(dev)
        if attack == "fgsm":
            xa = fgsm_attack(
                model=model,
                x=xb,
                y=yb,
                eps=eps,
                mutability_mask=mask_t,
                x_min=xmin_t,
                x_max=xmax_t,
            )
        elif attack == "pgd":
            xa = pgd_attack(
                model=model,
                x=xb,
                y=yb,
                eps=eps,
                mutability_mask=mask_t,
                x_min=xmin_t,
                x_max=xmax_t,
                steps=pgd_steps,
                seed=seed + b_idx,
            )
        else:
            raise ValueError(f"Unknown attack type: {attack}")

        chk = verify_perturbation_constraints(
            x_orig=xb,
            x_adv=xa,
            eps=eps,
            mutability_mask=mask_t,
            x_min=xmin_t,
            x_max=xmax_t,
            raise_on_violation=True,
        )
        all_valid = all_valid and bool(chk["valid"])
        max_imm_delta = max(max_imm_delta, float(chk["max_immutable_delta"]))
        max_linf_delta = max(max_linf_delta, float(chk["max_linf_delta"]))
        max_bound_viol = max(max_bound_viol, float(chk["max_bound_violation"]))
        adv_chunks.append(xa.cpu().numpy())

    return np.concatenate(adv_chunks, axis=0), {
        "valid": all_valid,
        "max_immutable_delta": max_imm_delta,
        "max_linf_delta": max_linf_delta,
        "max_bound_violation": max_bound_viol,
    }


def _predictive_entropy_numpy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    return (-np.sum(p * np.log(p), axis=1) / np.log(float(probs.shape[1]))).astype(np.float32)


@torch.no_grad()
def _batched_probs(
    model: RobustIDSNet, X: np.ndarray, device: Union[str, torch.device], batch_size: int = 512
) -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks: List[np.ndarray] = []
    for (xb,) in loader:
        probs = model.predict_proba(xb.to(dev))
        chunks.append(probs.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _lookup_eps_metric(epsilons: Sequence[float], values: Sequence[float], target: float = 0.10) -> float:
    for e, v in zip(epsilons, values):
        if abs(float(e) - target) < 1e-6:
            return float(v)
    # Closest available epsilon fallback
    idx = int(np.argmin(np.abs(np.asarray(epsilons, dtype=np.float64) - target)))
    return float(values[idx])


def plot_accuracy_vs_epsilon(
    configs: Sequence[ConfigBenchmarkMetrics],
    output_path: Union[str, Path],
) -> Path:
    """Render publication-quality Clean vs. Robust Accuracy tradeoff curves."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    palette = [
        "#D55E00",  # Vermilion: Unhardened Baseline
        "#E69F00",  # Amber: Baseline + Feature Squeezing
        "#0072B2",  # Deep Blue: PGD-AT
        "#009E73",  # Bluish Green: Full Defense Stack
    ]
    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", "-.", "-"]

    fig, (ax_pgd, ax_fgsm) = plt.subplots(1, 2, figsize=(14.0, 5.8), dpi=300)
    fig.patch.set_facecolor("white")

    # --- Panel A: Multi-Step PGD White-Box Tradeoff Curve ---
    for idx, cfg in enumerate(configs):
        eps_arr = np.asarray(cfg.epsilons, dtype=np.float64)
        acc_pct = np.asarray(cfg.pgd_accuracies, dtype=np.float64) * 100.0
        ax_pgd.plot(
            eps_arr,
            acc_pct,
            label=cfg.name,
            color=palette[idx % len(palette)],
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2.4,
            markersize=7.0,
        )

    # Shade resilience gain between Unhardened Baseline and Full Defense Stack
    if len(configs) >= 4:
        eps_arr = np.asarray(configs[0].epsilons, dtype=np.float64)
        base_pct = np.asarray(configs[0].pgd_accuracies, dtype=np.float64) * 100.0
        full_pct = np.asarray(configs[3].pgd_accuracies, dtype=np.float64) * 100.0
        ax_pgd.fill_between(
            eps_arr,
            base_pct,
            full_pct,
            where=(full_pct >= base_pct),
            color="#009E73",
            alpha=0.10,
            label="Layered Resilience Gain",
        )
        # Callout annotation at epsilon = 0.10
        b_010 = configs[0].robust_acc_pgd_010 * 100.0
        f_010 = configs[3].robust_acc_pgd_010 * 100.0
        gain_pp = f_010 - b_010
        ax_pgd.axvline(0.10, color="#555555", linestyle=":", linewidth=1.3, alpha=0.85)
        ax_pgd.annotate(
            f"+{gain_pp:.1f} pp @ ε=0.10",
            xy=(0.10, (b_010 + f_010) / 2.0),
            xytext=(0.115, max(22.0, (b_010 + f_010) / 2.0 - 12.0)),
            fontsize=9.5,
            fontweight="bold",
            color="#0B5345",
            arrowprops=dict(arrowstyle="->", color="#0B5345", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F8F5", edgecolor="#009E73", lw=0.9),
        )

    ax_pgd.set_title(
        "(a) Iterative White-Box PGD-20 Robustness Curve",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_pgd.set_xlabel(r"$L_\infty$ Perturbation Budget ($\epsilon$)", fontsize=11)
    ax_pgd.set_ylabel("Classification / Defense Accuracy (%)", fontsize=11)
    ax_pgd.set_xlim(-0.005, max(configs[0].epsilons) + 0.008)
    ax_pgd.set_ylim(0.0, 102.0)
    ax_pgd.grid(True, linestyle="--", alpha=0.35)
    ax_pgd.spines["top"].set_visible(False)
    ax_pgd.spines["right"].set_visible(False)
    ax_pgd.legend(loc="lower left", fontsize=8.8, frameon=True, framealpha=0.94)

    # --- Panel B: Single-Step FGSM Curve + Mahalanobis Reject Dynamics ---
    for idx, cfg in enumerate(configs):
        eps_arr = np.asarray(cfg.epsilons, dtype=np.float64)
        fgsm_pct = np.asarray(cfg.fgsm_accuracies, dtype=np.float64) * 100.0
        ax_fgsm.plot(
            eps_arr,
            fgsm_pct,
            label=cfg.name,
            color=palette[idx % len(palette)],
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2.3,
            markersize=6.8,
        )

    # Overlay Mahalanobis Quarantine / Rejection Rate on twin axis or dashed accent
    full_cfg = configs[-1]
    if full_cfg.rejection_rates_pgd is not None:
        eps_arr = np.asarray(full_cfg.epsilons, dtype=np.float64)
        rej_pct = np.asarray(full_cfg.rejection_rates_pgd, dtype=np.float64) * 100.0
        ax_fgsm.plot(
            eps_arr,
            rej_pct,
            label="Mahalanobis Quarantine Rate (PGD)",
            color="#6A3D9A",
            marker="P",
            linestyle=":",
            linewidth=2.0,
            markersize=6.5,
        )

    ax_fgsm.axvline(0.10, color="#555555", linestyle=":", linewidth=1.3, alpha=0.85)
    ax_fgsm.set_title(
        "(b) Single-Step FGSM Robustness & OOD Quarantine Rate",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_fgsm.set_xlabel(r"$L_\infty$ Perturbation Budget ($\epsilon$)", fontsize=11)
    ax_fgsm.set_ylabel("Accuracy / Quarantine Rate (%)", fontsize=11)
    ax_fgsm.set_xlim(-0.005, max(configs[0].epsilons) + 0.008)
    ax_fgsm.set_ylim(0.0, 102.0)
    ax_fgsm.grid(True, linestyle="--", alpha=0.35)
    ax_fgsm.spines["top"].set_visible(False)
    ax_fgsm.spines["right"].set_visible(False)
    ax_fgsm.legend(
        loc="center left",
        bbox_to_anchor=(0.01, 0.43),
        fontsize=8.3,
        frameon=True,
        framealpha=0.94,
    )

    fig.suptitle(
        "Adversarial ML Evasion & Layered Defense Tradeoff on Tabular IDS Network Flows",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_feature_vulnerability(
    baseline_report: FeatureVulnerabilityReport,
    pgdat_report: FeatureVulnerabilityReport,
    output_path: Union[str, Path],
    eps: float = 0.10,
) -> Path:
    """Render ranked network feature gradient sensitivity and domain-mask enforcement chart."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Sort features ascending on y-axis so highest effective baseline vulnerability appears at top
    # Place immutable features cleanly at bottom sorted by raw gradient magnitude
    mut_mask = baseline_report.mutability_mask
    eff_base = baseline_report.effective_grad_sensitivity
    raw_base = baseline_report.raw_grad_sensitivity

    sort_keys = np.lexsort((raw_base, eff_base, mut_mask))
    names_sorted = [
        f"{baseline_report.feature_names[i]}" + ("  [M=0 LOCKED]" if mut_mask[i] == 0.0 else "")
        for i in sort_keys
    ]

    base_eff_sorted = baseline_report.effective_grad_sensitivity[sort_keys]
    pgd_eff_sorted = pgdat_report.effective_grad_sensitivity[sort_keys]
    base_pert_sorted = baseline_report.mean_pgd_perturbation[sort_keys]
    pgd_pert_sorted = pgdat_report.mean_pgd_perturbation[sort_keys]
    mask_sorted = mut_mask[sort_keys]

    y_pos = np.arange(len(names_sorted), dtype=np.float64)
    bar_h = 0.37

    fig, (ax_grad, ax_pert) = plt.subplots(
        1, 2, figsize=(14.5, 7.6), dpi=300, sharey=True
    )
    fig.patch.set_facecolor("white")

    # Shade locked immutable protocol region at bottom
    n_imm = int(np.sum(mask_sorted == 0.0))
    if n_imm > 0:
        for ax in (ax_grad, ax_pert):
            ax.axhspan(
                -0.6,
                n_imm - 0.5,
                color="#EBEDEF",
                alpha=0.75,
                zorder=0,
            )
            ax.axhline(n_imm - 0.5, color="#7F8C8D", linestyle="--", linewidth=1.1, zorder=2)

    # --- Panel A: Effective Masked Input Gradient Sensitivity ---
    ax_grad.barh(
        y_pos + bar_h / 2.0,
        base_eff_sorted,
        height=bar_h,
        color="#D55E00",
        alpha=0.88,
        label="Unhardened Baseline  $M_j \\cdot \\mathbb{E}[|\\partial \\mathcal{L}/\\partial x_j|]$",
        zorder=3,
    )
    ax_grad.barh(
        y_pos - bar_h / 2.0,
        pgd_eff_sorted,
        height=bar_h,
        color="#0072B2",
        alpha=0.88,
        label="PGD-AT Hardened  $M_j \\cdot \\mathbb{E}[|\\partial \\mathcal{L}/\\partial x_j|]$",
        zorder=3,
    )

    ax_grad.set_yticks(y_pos)
    ax_grad.set_yticklabels(names_sorted, fontsize=9.3)
    ax_grad.set_xlabel(r"Effective Masked Gradient Sensitivity  $M_j \cdot \mathbb{E}[|\nabla_{x_j}\mathcal{L}|]$", fontsize=10.5)
    ax_grad.set_title(
        "(a) Input Loss-Gradient Vulnerability Ranking",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_grad.grid(True, axis="x", linestyle="--", alpha=0.35, zorder=1)
    ax_grad.spines["top"].set_visible(False)
    ax_grad.spines["right"].set_visible(False)
    ax_grad.legend(loc="lower right", fontsize=8.7, frameon=True, framealpha=0.94)

    if n_imm > 0:
        max_x_grad = max(float(np.max(base_eff_sorted)), 1e-3)
        ax_grad.text(
            max_x_grad * 0.36,
            max(0.0, (n_imm - 1) / 2.0),
            "Immutable Protocol Headers (M = 0)\nZero Allowed Evasion Perturbation",
            fontsize=8.8,
            fontweight="bold",
            color="#34495E",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#95A5A6", alpha=0.9),
        )

    # --- Panel B: Empirical Mean |x_adv - x| Distortion under PGD ---
    ax_pert.barh(
        y_pos + bar_h / 2.0,
        base_pert_sorted,
        height=bar_h,
        color="#E69F00",
        alpha=0.90,
        label=f"Baseline PGD Mean $|\\delta_j|$ ($\\epsilon={eps:.2f}$)",
        zorder=3,
    )
    ax_pert.barh(
        y_pos - bar_h / 2.0,
        pgd_pert_sorted,
        height=bar_h,
        color="#009E73",
        alpha=0.90,
        label=f"PGD-AT PGD Mean $|\\delta_j|$ ($\\epsilon={eps:.2f}$)",
        zorder=3,
    )
    ax_pert.axvline(eps, color="#C0392B", linestyle=":", linewidth=1.4, label=f"$L_\\infty$ Budget Cap ($\\epsilon={eps:.2f}$)")
    ax_pert.set_xlim(0.0, eps * 1.18)
    ax_pert.set_xlabel(r"Empirical Mean Perturbation Magnitude  $\mathbb{E}[|x_{\mathrm{adv}, j} - x_j|]$", fontsize=10.5)
    ax_pert.set_title(
        f"(b) Realized Feature Distortion & Protocol Invariance (ε={eps:.2f})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_pert.grid(True, axis="x", linestyle="--", alpha=0.35, zorder=1)
    ax_pert.spines["top"].set_visible(False)
    ax_pert.spines["right"].set_visible(False)
    ax_pert.legend(loc="lower right", fontsize=8.7, frameon=True, framealpha=0.94)

    fig.suptitle(
        "Domain-Constrained Feature Vulnerability & Jacobian Regularization Analysis",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _default_console() -> Console:
    if not sys.stdout.isatty():
        return Console(width=116)
    return Console()


def render_rich_summary_table(
    suite: BenchmarkSuiteOutput,
    console: Optional[Console] = None,
) -> Table:
    """Display formatted `rich` summary table and domain-constraint verification panel."""
    con = console if console is not None else _default_console()

    table = Table(
        title="Adversarial ML Evasion & Layered Defense Benchmark (White-Box L_inf Threat Model)",
        show_lines=True,
    )
    table.add_column("Defense Configuration", style="bold white", min_width=32)
    table.add_column("Clean Acc\n(ε = 0.00)", justify="right", style="cyan")
    table.add_column("FGSM Acc\n(ε = 0.10)", justify="right", style="yellow")
    table.add_column("PGD-20 Acc\n(ε = 0.10)", justify="right", style="bold green")
    table.add_column("Macro F1\n(Clean / ε=0.10)", justify="right", style="magenta")
    table.add_column("OOD Detection\nAUROC", justify="right", style="bright_blue")

    for cfg in suite.configs:
        table.add_row(
            cfg.name,
            f"{cfg.clean_accuracy * 100:.2f}%",
            f"{cfg.robust_acc_fgsm_010 * 100:.2f}%",
            f"{cfg.robust_acc_pgd_010 * 100:.2f}%",
            f"{cfg.clean_macro_f1:.4f} / {cfg.robust_macro_f1_010:.4f}",
            f"{cfg.ood_auroc:.4f}",
        )

    con.print(table)

    cs = suite.constraint_summary
    status_badge = "[bold green]PASS[/bold green]" if cs["all_valid"] else "[bold red]FAIL[/bold red]"
    con.print(
        Panel(
            f"Domain Protocol Constraint Verification: {status_badge}  |  "
            f"Evaluated Batches: {cs['batches_checked']}  |  "
            f"Max Immutable Header Drift (M=0): {cs['max_immutable_delta']:.8f}  |  "
            f"Max Physical Bound Violation: {cs['max_bound_violation']:.8f}",
            title="Problem-Space Tabular Invariants",
            border_style="green" if cs["all_valid"] else "red",
        )
    )
    return table


def run_full_benchmark(
    config: Optional[BenchmarkConfig] = None,
    bundle: Optional[FlowDatasetBundle] = None,
    console: Optional[Console] = None,
) -> BenchmarkSuiteOutput:
    """Execute the complete 4-configuration white-box robustness benchmark and save plots."""
    cfg = config if config is not None else BenchmarkConfig()
    con = console if console is not None else _default_console()
    set_deterministic_seed(cfg.seed)

    if bundle is None:
        con.print("[bold cyan][1/5] Loading domain-constrained network-flow dataset...[/bold cyan]")
        bundle = load_or_generate_dataset(
            csv_path=cfg.csv_path,
            n_samples=cfg.n_samples,
            seed=cfg.seed,
        )

    schema = bundle.schema
    eps_list = [float(e) for e in cfg.epsilons]

    # 1. Train Unhardened Baseline Classifier
    con.print("[bold cyan][2/5] Training Unhardened Baseline RobustIDSNet...[/bold cyan]")
    base_train_cfg = TrainingConfig(
        epochs=cfg.baseline_epochs,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        device=cfg.device,
    )
    baseline_model, _ = train_baseline_classifier(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_val=bundle.X_val,
        y_val=bundle.y_val,
        num_classes=schema.num_classes,
        config=base_train_cfg,
    )

    # 2. Configure Feature Squeezer (fitted strictly on clean training split)
    squeezer = FeatureSqueezer(
        schema=schema,
        bit_depth=cfg.squeeze_bit_depth,
        smoothing="gaussian",
    ).fit(bundle.X_train)

    # 3. Train PGD-AT Hardened Classifier
    con.print("[bold cyan][3/5] Training Min-Max PGD Adversarially Hardened Model (PGD-AT)...[/bold cyan]")
    pgdat_cfg = PGDATConfig(
        epochs=cfg.pgdat_epochs,
        batch_size=cfg.batch_size,
        train_eps=cfg.train_eps,
        pgd_steps=cfg.pgd_train_steps,
        adv_mix_weight=cfg.adv_mix_weight,
        seed=cfg.seed,
        device=cfg.device,
    )
    pgdat_model, _ = train_pgd_adversarial(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_val=bundle.X_val,
        y_val=bundle.y_val,
        schema=schema,
        config=pgdat_cfg,
    )

    # 4. Fit Mahalanobis + Entropy OOD Detector & Full Defense Stack
    con.print("[bold cyan][4/5] Fitting Class-Conditional Mahalanobis & Entropy OOD Detector...[/bold cyan]")
    detector = MahalanobisOODDetector(target_fpr=cfg.mahalanobis_fpr).fit(
        model=pgdat_model,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_val=bundle.X_val,
        device=cfg.device,
    )
    light_squeezer = FeatureSqueezer(
        schema=schema,
        bit_depth=5,
        smoothing="none",
    ).fit(bundle.X_train)
    full_stack = HardenedIDSStack(
        model=pgdat_model,
        detector=detector,
        squeezer=light_squeezer,
        device=cfg.device,
    )

    # 5. Evaluate all 4 configurations across epsilon budgets under both PGD and FGSM
    con.print("[bold cyan][5/5] Running white-box FGSM & PGD evaluations and rendering plots...[/bold cyan]")

    base_pgd_acc, base_fgsm_acc, base_pgd_f1 = [], [], []
    sq_pgd_acc, sq_fgsm_acc, sq_pgd_f1 = [], [], []
    at_pgd_acc, at_fgsm_acc, at_pgd_f1 = [], [], []
    full_pgd_acc, full_fgsm_acc, full_pgd_f1 = [], [], []
    full_rej_rates, full_sel_accs = [], []

    global_valid = True
    global_max_imm = 0.0
    global_max_bound = 0.0
    checks_count = 0

    adv_base_010: Optional[np.ndarray] = None
    adv_pgdat_010: Optional[np.ndarray] = None

    y_true = bundle.y_test

    for eps in eps_list:
        # Baseline white-box perturbations (PGD & FGSM)
        x_base_pgd, chk1 = _generate_attack_batches(
            baseline_model, bundle.X_test, y_true, schema, eps, "pgd",
            cfg.pgd_eval_steps, cfg.eval_batch_size, cfg.device, cfg.seed
        )
        x_base_fgsm, chk2 = _generate_attack_batches(
            baseline_model, bundle.X_test, y_true, schema, eps, "fgsm",
            1, cfg.eval_batch_size, cfg.device, cfg.seed
        )
        # PGD-AT white-box perturbations (PGD & FGSM)
        x_at_pgd, chk3 = _generate_attack_batches(
            pgdat_model, bundle.X_test, y_true, schema, eps, "pgd",
            cfg.pgd_eval_steps, cfg.eval_batch_size, cfg.device, cfg.seed
        )
        x_at_fgsm, chk4 = _generate_attack_batches(
            pgdat_model, bundle.X_test, y_true, schema, eps, "fgsm",
            1, cfg.eval_batch_size, cfg.device, cfg.seed
        )

        for chk in (chk1, chk2, chk3, chk4):
            global_valid = global_valid and bool(chk["valid"])
            global_max_imm = max(global_max_imm, float(chk["max_immutable_delta"]))
            global_max_bound = max(global_max_bound, float(chk["max_bound_violation"]))
            checks_count += 1

        if abs(eps - 0.10) < 1e-6:
            adv_base_010 = x_base_pgd
            adv_pgdat_010 = x_at_pgd

        # Config 1: Unhardened Baseline
        p1_pgd = np.argmax(_batched_probs(baseline_model, x_base_pgd, cfg.device), axis=1)
        p1_fgsm = np.argmax(_batched_probs(baseline_model, x_base_fgsm, cfg.device), axis=1)
        base_pgd_acc.append(float(accuracy_score(y_true, p1_pgd)))
        base_fgsm_acc.append(float(accuracy_score(y_true, p1_fgsm)))
        base_pgd_f1.append(float(f1_score(y_true, p1_pgd, average="macro", zero_division=0)))

        # Config 2: Baseline + Feature Squeezing
        x_sq_pgd = squeezer.transform_numpy(x_base_pgd)
        x_sq_fgsm = squeezer.transform_numpy(x_base_fgsm)
        p2_pgd = np.argmax(_batched_probs(baseline_model, x_sq_pgd, cfg.device), axis=1)
        p2_fgsm = np.argmax(_batched_probs(baseline_model, x_sq_fgsm, cfg.device), axis=1)
        sq_pgd_acc.append(float(accuracy_score(y_true, p2_pgd)))
        sq_fgsm_acc.append(float(accuracy_score(y_true, p2_fgsm)))
        sq_pgd_f1.append(float(f1_score(y_true, p2_pgd, average="macro", zero_division=0)))

        # Config 3: Adversarially Trained Model (PGD-AT)
        p3_pgd = np.argmax(_batched_probs(pgdat_model, x_at_pgd, cfg.device), axis=1)
        p3_fgsm = np.argmax(_batched_probs(pgdat_model, x_at_fgsm, cfg.device), axis=1)
        at_pgd_acc.append(float(accuracy_score(y_true, p3_pgd)))
        at_fgsm_acc.append(float(accuracy_score(y_true, p3_fgsm)))
        at_pgd_f1.append(float(f1_score(y_true, p3_pgd, average="macro", zero_division=0)))

        # Config 4: Full Defense Stack (PGD-AT + Mahalanobis Reject Option)
        out4_pgd = full_stack.predict_detailed(x_at_pgd)
        out4_fgsm = full_stack.predict_detailed(x_at_fgsm)

        rej_pgd = out4_pgd.is_rejected
        rej_fgsm = out4_fgsm.is_rejected
        accepted_pgd = ~rej_pgd

        full_rej_rates.append(float(np.mean(rej_pgd)))
        sel_acc = (
            float(accuracy_score(y_true[accepted_pgd], out4_pgd.raw_predictions[accepted_pgd]))
            if np.any(accepted_pgd)
            else 1.0
        )
        full_sel_accs.append(sel_acc)

        if eps == 0.0:
            # Clean evaluation: false rejections count strictly as errors (honest FPR tradeoff)
            eff_pgd = float(np.mean((~rej_pgd) & (out4_pgd.raw_predictions == y_true)))
            eff_fgsm = float(np.mean((~rej_fgsm) & (out4_fgsm.raw_predictions == y_true)))
            effective_preds = np.where(
                rej_pgd, (y_true + 1) % schema.num_classes, out4_pgd.raw_predictions
            )
            eff_f1 = float(f1_score(y_true, effective_preds, average="macro", zero_division=0))
        else:
            # Adversarial evaluation: evasion fails if flow is correctly classified OR quarantined
            eff_pgd = float(np.mean((out4_pgd.raw_predictions == y_true) | rej_pgd))
            eff_fgsm = float(np.mean((out4_fgsm.raw_predictions == y_true) | rej_fgsm))
            eff_f1 = (
                float(f1_score(y_true[accepted_pgd], out4_pgd.raw_predictions[accepted_pgd], average="macro", zero_division=0))
                if np.any(accepted_pgd)
                else 1.0
            )

        full_pgd_acc.append(eff_pgd)
        full_fgsm_acc.append(eff_fgsm)
        full_pgd_f1.append(eff_f1)

    # Compute OOD Detection AUROC across clean test vs (adversarial eps=0.10 + unseen zero-day OOD)
    if adv_base_010 is None:
        adv_base_010, _ = _generate_attack_batches(
            baseline_model, bundle.X_test, y_true, schema, 0.10, "pgd",
            cfg.pgd_eval_steps, cfg.eval_batch_size, cfg.device, cfg.seed
        )
    if adv_pgdat_010 is None:
        adv_pgdat_010, _ = _generate_attack_batches(
            pgdat_model, bundle.X_test, y_true, schema, 0.10, "pgd",
            cfg.pgd_eval_steps, cfg.eval_batch_size, cfg.device, cfg.seed
        )

    ood_pool_base = (
        np.vstack([adv_base_010, bundle.X_ood]) if bundle.X_ood is not None else adv_base_010
    )
    ood_pool_at = (
        np.vstack([adv_pgdat_010, bundle.X_ood]) if bundle.X_ood is not None else adv_pgdat_010
    )

    # 1. Baseline OOD score: predictive entropy
    auroc_base = compute_ood_auroc(
        _predictive_entropy_numpy(_batched_probs(baseline_model, bundle.X_test, cfg.device)),
        _predictive_entropy_numpy(_batched_probs(baseline_model, ood_pool_base, cfg.device)),
    )

    # 2. Feature Squeezing OOD score: L1 squeeze disagreement + squeezed entropy
    clean_dis = squeezer.disagreement_score(
        baseline_model, torch.as_tensor(bundle.X_test, dtype=torch.float32, device=cfg.device)
    ).cpu().numpy()
    ood_dis = squeezer.disagreement_score(
        baseline_model, torch.as_tensor(ood_pool_base, dtype=torch.float32, device=cfg.device)
    ).cpu().numpy()
    clean_sq_ent = _predictive_entropy_numpy(
        _batched_probs(baseline_model, squeezer.transform_numpy(bundle.X_test), cfg.device)
    )
    ood_sq_ent = _predictive_entropy_numpy(
        _batched_probs(baseline_model, squeezer.transform_numpy(ood_pool_base), cfg.device)
    )
    auroc_sq = compute_ood_auroc(clean_dis + 0.5 * clean_sq_ent, ood_dis + 0.5 * ood_sq_ent)

    # 3. PGD-AT OOD score: calibrated predictive entropy
    auroc_at = compute_ood_auroc(
        _predictive_entropy_numpy(_batched_probs(pgdat_model, bundle.X_test, cfg.device)),
        _predictive_entropy_numpy(_batched_probs(pgdat_model, ood_pool_at, cfg.device)),
    )

    # 4. Full Defense Stack OOD score: Class-Conditional Mahalanobis + Entropy
    auroc_full = compute_ood_auroc(
        detector.anomaly_score(pgdat_model, bundle.X_test, device=cfg.device),
        detector.anomaly_score(pgdat_model, ood_pool_at, device=cfg.device),
    )

    configs_out = [
        ConfigBenchmarkMetrics(
            name="1. Unhardened Baseline Model",
            epsilons=eps_list,
            pgd_accuracies=base_pgd_acc,
            fgsm_accuracies=base_fgsm_acc,
            pgd_macro_f1s=base_pgd_f1,
            clean_accuracy=base_pgd_acc[0],
            robust_acc_pgd_010=_lookup_eps_metric(eps_list, base_pgd_acc, 0.10),
            robust_acc_fgsm_010=_lookup_eps_metric(eps_list, base_fgsm_acc, 0.10),
            clean_macro_f1=base_pgd_f1[0],
            robust_macro_f1_010=_lookup_eps_metric(eps_list, base_pgd_f1, 0.10),
            ood_auroc=auroc_base,
        ),
        ConfigBenchmarkMetrics(
            name="2. Baseline + Feature Squeezing",
            epsilons=eps_list,
            pgd_accuracies=sq_pgd_acc,
            fgsm_accuracies=sq_fgsm_acc,
            pgd_macro_f1s=sq_pgd_f1,
            clean_accuracy=sq_pgd_acc[0],
            robust_acc_pgd_010=_lookup_eps_metric(eps_list, sq_pgd_acc, 0.10),
            robust_acc_fgsm_010=_lookup_eps_metric(eps_list, sq_fgsm_acc, 0.10),
            clean_macro_f1=sq_pgd_f1[0],
            robust_macro_f1_010=_lookup_eps_metric(eps_list, sq_pgd_f1, 0.10),
            ood_auroc=auroc_sq,
        ),
        ConfigBenchmarkMetrics(
            name="3. Adversarially Trained (PGD-AT)",
            epsilons=eps_list,
            pgd_accuracies=at_pgd_acc,
            fgsm_accuracies=at_fgsm_acc,
            pgd_macro_f1s=at_pgd_f1,
            clean_accuracy=at_pgd_acc[0],
            robust_acc_pgd_010=_lookup_eps_metric(eps_list, at_pgd_acc, 0.10),
            robust_acc_fgsm_010=_lookup_eps_metric(eps_list, at_fgsm_acc, 0.10),
            clean_macro_f1=at_pgd_f1[0],
            robust_macro_f1_010=_lookup_eps_metric(eps_list, at_pgd_f1, 0.10),
            ood_auroc=auroc_at,
        ),
        ConfigBenchmarkMetrics(
            name="4. Full Stack (PGD-AT + Mahalanobis)",
            epsilons=eps_list,
            pgd_accuracies=full_pgd_acc,
            fgsm_accuracies=full_fgsm_acc,
            pgd_macro_f1s=full_pgd_f1,
            clean_accuracy=full_pgd_acc[0],
            robust_acc_pgd_010=_lookup_eps_metric(eps_list, full_pgd_acc, 0.10),
            robust_acc_fgsm_010=_lookup_eps_metric(eps_list, full_fgsm_acc, 0.10),
            clean_macro_f1=full_pgd_f1[0],
            robust_macro_f1_010=_lookup_eps_metric(eps_list, full_pgd_f1, 0.10),
            ood_auroc=auroc_full,
            rejection_rates_pgd=full_rej_rates,
            selective_accuracies_pgd=full_sel_accs,
        ),
    ]

    # Feature vulnerability diagnostics
    base_vuln = analyze_feature_vulnerability(
        baseline_model, bundle.X_test, y_true, schema, eps=0.10, device=cfg.device
    )
    pgdat_vuln = analyze_feature_vulnerability(
        pgdat_model, bundle.X_test, y_true, schema, eps=0.10, device=cfg.device
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc_plot = plot_accuracy_vs_epsilon(configs_out, out_dir / "accuracy_vs_epsilon.png")
    vuln_plot = plot_feature_vulnerability(
        base_vuln, pgdat_vuln, out_dir / "feature_vulnerability.png", eps=0.10
    )

    constraint_summary: Dict[str, Union[bool, float, int]] = {
        "all_valid": global_valid,
        "batches_checked": checks_count,
        "max_immutable_delta": global_max_imm,
        "max_bound_violation": global_max_bound,
    }

    metrics_json = out_dir / "benchmark_metrics.json"
    serializable = {
        "config": {
            "n_samples": cfg.n_samples,
            "baseline_epochs": cfg.baseline_epochs,
            "pgdat_epochs": cfg.pgdat_epochs,
            "train_eps": cfg.train_eps,
            "pgd_eval_steps": cfg.pgd_eval_steps,
            "epsilons": eps_list,
            "seed": cfg.seed,
        },
        "constraint_summary": constraint_summary,
        "configurations": [asdict(c) for c in configs_out],
        "feature_vulnerability": {
            "feature_names": base_vuln.feature_names,
            "mutability_mask": base_vuln.mutability_mask.tolist(),
            "baseline_effective_grad": base_vuln.effective_grad_sensitivity.tolist(),
            "pgdat_effective_grad": pgdat_vuln.effective_grad_sensitivity.tolist(),
            "baseline_mean_pgd_delta": base_vuln.mean_pgd_perturbation.tolist(),
            "pgdat_mean_pgd_delta": pgdat_vuln.mean_pgd_perturbation.tolist(),
        },
    }
    metrics_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    suite_out = BenchmarkSuiteOutput(
        configs=configs_out,
        baseline_vulnerability=base_vuln,
        pgdat_vulnerability=pgdat_vuln,
        constraint_summary=constraint_summary,
        accuracy_plot_path=acc_plot,
        vulnerability_plot_path=vuln_plot,
        metrics_json_path=metrics_json,
    )

    render_rich_summary_table(suite_out, console=con)
    con.print(
        f"[bold green]Saved Artifacts:[/bold green] "
        f"[underline]{acc_plot}[/underline], "
        f"[underline]{vuln_plot}[/underline], "
        f"[underline]{metrics_json}[/underline]"
    )
    return suite_out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adversarial ML Evasion & Layered Robustness Benchmark Runner for IDS Classifiers."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional path to NSL-KDD / CIC-IDS / custom network-flow CSV dataset.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=12000,
        help="Number of synthetic network-flow records to generate (default: 12000).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum training epochs for baseline and PGD-AT classifiers (default: 40).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size during model optimization (default: 128).",
    )
    parser.add_argument(
        "--pgd-steps",
        type=int,
        default=20,
        help="Number of white-box PGD iterations during evaluation (default: 20).",
    )
    parser.add_argument(
        "--train-eps",
        type=float,
        default=0.11,
        help="L_inf perturbation radius used during PGD-AT hardening (default: 0.11).",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPSILONS),
        help="Sequence of L_inf evaluation budgets (default: 0.0 0.02 0.05 0.10 0.15 0.20).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for plots and JSON benchmark artifacts (default: results).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed across Python, NumPy, and PyTorch (default: 42).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast smoke benchmark (3000 samples, 15 epochs) for rapid verification.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    n_samples = 3000 if args.quick else args.samples
    epochs = 15 if args.quick else args.epochs
    pgd_steps = 10 if args.quick else args.pgd_steps

    cfg = BenchmarkConfig(
        n_samples=n_samples,
        csv_path=args.csv_path,
        baseline_epochs=epochs,
        pgdat_epochs=epochs,
        batch_size=args.batch_size,
        pgd_eval_steps=pgd_steps,
        train_eps=args.train_eps,
        epsilons=tuple(float(e) for e in args.epsilons),
        seed=args.seed,
        output_dir=args.output_dir,
    )
    run_full_benchmark(config=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
