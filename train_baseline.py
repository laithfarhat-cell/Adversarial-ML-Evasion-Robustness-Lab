"""Standalone CLI pipeline to train and evaluate the baseline PyTorch IDS classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch
from rich.console import Console
from rich.table import Table

from data.dataset_loader import load_or_generate_dataset
from models.ids_classifier import (
    TrainingConfig,
    evaluate_classifier,
    train_baseline_classifier,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline RobustIDSNet classifier on network-flow data."
    )
    parser.add_argument("--csv-path", type=str, default=None, help="Optional CSV dataset path.")
    parser.add_argument("--samples", type=int, default=12000, help="Number of synthetic samples.")
    parser.add_argument("--epochs", type=int, default=45, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Initial AdamW learning rate.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/baseline_ids.pt",
        help="Output path to save trained baseline model weights.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    console = Console()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = load_or_generate_dataset(
        csv_path=args.csv_path,
        n_samples=args.samples,
        seed=args.seed,
    )

    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        device=device,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
    )

    console.print(
        f"[bold cyan]Training Baseline RobustIDSNet[/bold cyan] "
        f"(train={bundle.X_train.shape[0]}, val={bundle.X_val.shape[0]}, "
        f"test={bundle.X_test.shape[0]}, features={bundle.schema.num_features}, device={device})"
    )

    model, history = train_baseline_classifier(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_val=bundle.X_val,
        y_val=bundle.y_val,
        num_classes=bundle.schema.num_classes,
        config=cfg,
    )

    test_metrics = evaluate_classifier(model, bundle.X_test, bundle.y_test, device=device)

    table = Table(title="Baseline IDS Classifier Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Best Epoch", str(history.best_epoch + 1))
    table.add_row("Best Validation Loss", f"{history.best_val_loss:.4f}")
    table.add_row("Clean Test Accuracy", f"{test_metrics['accuracy'] * 100:.2f}%")
    table.add_row("Clean Test Macro F1", f"{test_metrics['macro_f1']:.4f}")
    if args.checkpoint:
        table.add_row("Saved Checkpoint", str(args.checkpoint))
    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
