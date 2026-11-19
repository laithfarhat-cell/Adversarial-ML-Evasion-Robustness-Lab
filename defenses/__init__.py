"""Layered ML defense and hardening suite: PGD-AT, Feature Squeezing, and Mahalanobis OOD."""

from defenses.hardening import (
    REJECT_LABEL,
    FeatureSqueezer,
    HardenedIDSStack,
    HardenedPrediction,
    MahalanobisOODDetector,
    PGDATConfig,
    PGDATHistory,
    compute_ood_auroc,
    train_pgd_adversarial,
)

__all__ = [
    "REJECT_LABEL",
    "FeatureSqueezer",
    "HardenedIDSStack",
    "HardenedPrediction",
    "MahalanobisOODDetector",
    "PGDATConfig",
    "PGDATHistory",
    "compute_ood_auroc",
    "train_pgd_adversarial",
]
