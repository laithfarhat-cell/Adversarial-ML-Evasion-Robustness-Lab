"""Domain-aware white-box gradient robustness evaluation harness (FGSM & PGD)."""

from robustness.gradient_eval import (
    FeatureVulnerabilityReport,
    RobustnessEvalResult,
    analyze_feature_vulnerability,
    evaluate_whitebox_robustness,
    fgsm_attack,
    pgd_attack,
    project_domain_perturbation,
    verify_perturbation_constraints,
)

__all__ = [
    "FeatureVulnerabilityReport",
    "RobustnessEvalResult",
    "analyze_feature_vulnerability",
    "evaluate_whitebox_robustness",
    "fgsm_attack",
    "pgd_attack",
    "project_domain_perturbation",
    "verify_perturbation_constraints",
]
