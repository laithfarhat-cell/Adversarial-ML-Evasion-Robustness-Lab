"""Unified CLI Entrypoint for the Adversarial ML Evasion & Robustness Lab."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from rich.console import Console
from rich.panel import Panel

import train_baseline
from benchmarks.robustness_curves import (
    DEFAULT_EPSILONS,
    BenchmarkConfig,
    run_full_benchmark,
)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ids-robustness-lab",
        description=(
            "Adversarial ML Evasion & Layered Robustness Laboratory for "
            "Tabular Cybersecurity Intrusion Detection Classifiers."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # `benchmark` subcommand (also default when invoked without subcommand)
    bench_p = subparsers.add_parser(
        "benchmark",
        help="Run the full 4-configuration white-box robustness benchmark & generate plots.",
    )
    _add_benchmark_flags(bench_p)

    # `train-baseline` subcommand
    base_p = subparsers.add_parser(
        "train-baseline",
        help="Train and save only the clean baseline RobustIDSNet classifier.",
    )
    base_p.add_argument("--csv-path", type=str, default=None, help="Optional CSV flow dataset.")
    base_p.add_argument("--samples", type=int, default=12000, help="Synthetic sample count.")
    base_p.add_argument("--epochs", type=int, default=45, help="Max training epochs.")
    base_p.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    base_p.add_argument("--lr", type=float, default=2e-3, help="Initial learning rate.")
    base_p.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    base_p.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    base_p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/baseline_ids.pt",
        help="Path to save baseline model checkpoint.",
    )

    # Allow top-level flags directly (`python main.py --samples 12000 ...`)
    _add_benchmark_flags(parser)
    return parser


def _add_benchmark_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv-path", type=str, default=None, help="Optional CSV flow dataset.")
    parser.add_argument("--samples", type=int, default=12000, help="Synthetic sample count.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training mini-batch size.")
    parser.add_argument("--pgd-steps", type=int, default=20, help="White-box PGD steps.")
    parser.add_argument("--train-eps", type=float, default=0.11, help="PGD-AT training epsilon.")
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPSILONS),
        help="Evaluation epsilon budgets.",
    )
    parser.add_argument("--output-dir", type=str, default="results", help="Artifacts directory.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed.")
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test mode.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(argv)
    console = Console(width=116) if not sys.stdout.isatty() else Console()

    console.print(
        Panel.fit(
            "[bold white]Adversarial ML Evasion & Robustness Lab[/bold white]\n"
            "[dim]Domain-Constrained White-Box Evaluation (FGSM / PGD) & Layered IDS Hardening[/dim]",
            border_style="cyan",
        )
    )

    if args.command == "train-baseline":
        cli_args = [
            "--samples", str(args.samples),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--patience", str(args.patience),
            "--seed", str(args.seed),
            "--checkpoint", str(args.checkpoint),
        ]
        if args.csv_path is not None:
            cli_args.extend(["--csv-path", args.csv_path])
        return train_baseline.main(cli_args)

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
    run_full_benchmark(config=cfg, console=console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
