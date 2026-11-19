"""Comprehensive Pytest Suite for the Adversarial ML Evasion & Robustness Lab.

Verifies:
  1. Domain-constrained dataset generation, invertible physical scaling, and
     NSL-KDD / CIC-IDS CSV loader resilience (Inf/NaN/categorical handling).
  2. `RobustIDSNet` architecture, singleton-batch BatchNorm safety, deterministic
     seeding, and non-vanishing input Jacobian gradient flow.
  3. Exact mutability mask enforcement (x_adv[M == 0] == x[M == 0]), L_inf budget
     compliance, and physical telemetry bound projection for FGSM and PGD.
  4. Feature Squeezing quantization levels, immutable preservation, and smoothing modes.
  5. Mahalanobis OOD Detector precision matrix positive-definiteness, FPR calibration,
     and high OOD/evasion detection AUROC.
  6. Measurable white-box robustness gains of PGD-AT and the Full Defense Stack over
     the unhardened baseline classifier at epsilon = 0.10.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from data.dataset_loader import (
    FlowDatasetBundle,
    generate_synthetic_flows,
    load_csv_flows,
)
from defenses.hardening import (
    REJECT_LABEL,
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
    evaluate_classifier,
    set_deterministic_seed,
    train_baseline_classifier,
)
from robustness.gradient_eval import (
    analyze_feature_vulnerability,
    evaluate_whitebox_robustness,
    fgsm_attack,
    pgd_attack,
    verify_perturbation_constraints,
)


@pytest.fixture(scope="module")
def lab_bundle() -> FlowDatasetBundle:
    """Shared deterministic dataset bundle for fast module-scoped robustness tests."""
    return generate_synthetic_flows(n_samples=2800, seed=123, include_ood=True)


@pytest.fixture(scope="module")
def trained_models(lab_bundle: FlowDatasetBundle):
    """Train both Unhardened Baseline and PGD-AT models once for comparative verification."""
    base_cfg = TrainingConfig(epochs=22, batch_size=128, seed=123, device="cpu")
    baseline_model, base_hist = train_baseline_classifier(
        X_train=lab_bundle.X_train,
        y_train=lab_bundle.y_train,
        X_val=lab_bundle.X_val,
        y_val=lab_bundle.y_val,
        num_classes=lab_bundle.schema.num_classes,
        config=base_cfg,
    )

    pgd_cfg = PGDATConfig(
        epochs=24,
        batch_size=128,
        train_eps=0.11,
        pgd_steps=8,
        adv_mix_weight=0.65,
        warmup_epochs=4,
        seed=123,
        device="cpu",
    )
    pgdat_model, pgdat_hist = train_pgd_adversarial(
        X_train=lab_bundle.X_train,
        y_train=lab_bundle.y_train,
        X_val=lab_bundle.X_val,
        y_val=lab_bundle.y_val,
        schema=lab_bundle.schema,
        config=pgd_cfg,
    )
    return baseline_model, base_hist, pgdat_model, pgdat_hist


# ---------------------------------------------------------------------------
# 1. Dataset Schema, Physical Scaling & CSV Loader Tests
# ---------------------------------------------------------------------------


def test_synthetic_dataset_schema_and_bounds(lab_bundle: FlowDatasetBundle) -> None:
    schema = lab_bundle.schema
    assert schema.num_features == 20
    assert schema.num_classes == 4
    assert schema.immutable_indices.size == 6
    assert schema.mutable_indices.size == 14

    for split in (lab_bundle.X_train, lab_bundle.X_val, lab_bundle.X_test):
        assert np.all(np.isfinite(split))
        assert np.all(split >= schema.x_min[None, :] - 1e-6)
        assert np.all(split <= schema.x_max[None, :] + 1e-6)

    # Verify invertible DomainBoundedScaler round-trip
    raw_recovered = lab_bundle.scaler.inverse_transform(lab_bundle.X_test[:64])
    assert np.all(raw_recovered >= schema.raw_min[None, :] - 1e-3)
    assert np.all(raw_recovered <= schema.raw_max[None, :] + 1e-3)
    norm_roundtrip = lab_bundle.scaler.transform(raw_recovered)
    assert np.allclose(lab_bundle.X_test[:64], norm_roundtrip, atol=1e-4)


def test_csv_loader_cicids_and_nslkdd_compatibility(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 160

    # Simulate CIC-IDS2017 CSV quirks: leading spaces in headers, Inf, NaN, text labels
    cic_df = pd.DataFrame(
        {
            " Destination Port": rng.choice([22, 53, 80, 443, 8080], size=n),
            " Protocol": rng.choice([6, 17], size=n),
            " SYN Flag Count": rng.integers(0, 2, size=n),
            " Flow Duration": rng.uniform(10.0, 50000.0, size=n),
            " Flow Bytes/s": np.where(
                rng.random(n) < 0.05, np.inf, rng.uniform(100.0, 1e6, size=n)
            ),
            " Flow IAT Mean": np.where(
                rng.random(n) < 0.05, np.nan, rng.uniform(0.1, 2000.0, size=n)
            ),
            " Packet Length Variance": rng.uniform(0.0, 40000.0, size=n),
            " Label": rng.choice(
                ["BENIGN", "DoS Hulk", "PortScan", "SSH-Patator"], size=n
            ),
        }
    )
    cic_csv = tmp_path / "cicids_sample.csv"
    cic_df.to_csv(cic_csv, index=False)

    cic_bundle = load_csv_flows(cic_csv, seed=7)
    assert cic_bundle.schema.num_features == 7
    assert cic_bundle.schema.immutable_indices.size == 3  # Port, Protocol, Flag
    assert cic_bundle.schema.mutable_indices.size == 4
    assert np.all(np.isfinite(cic_bundle.X_train))
    assert np.all(np.isfinite(cic_bundle.X_test))

    # Simulate NSL-KDD CSV schema with string categorical protocol/service/flag columns
    nsl_df = pd.DataFrame(
        {
            "duration": rng.exponential(200.0, size=n),
            "protocol_type": rng.choice(["tcp", "udp", "icmp"], size=n),
            "service": rng.choice(["http", "ssh", "domain_u", "private"], size=n),
            "flag": rng.choice(["SF", "S0", "REJ"], size=n),
            "src_bytes": rng.uniform(0.0, 50000.0, size=n),
            "dst_bytes": rng.uniform(0.0, 50000.0, size=n),
            "serror_rate": rng.uniform(0.0, 1.0, size=n),
            "label": rng.choice(["normal", "neptune", "nmap", "guess_passwd"], size=n),
        }
    )
    nsl_csv = tmp_path / "nslkdd_sample.csv"
    nsl_df.to_csv(nsl_csv, index=False)

    nsl_bundle = load_csv_flows(nsl_csv, seed=7)
    assert nsl_bundle.schema.num_features == 7
    # protocol_type, service, flag -> immutable (M=0)
    assert nsl_bundle.schema.mutability_mask[1] == 0.0
    assert nsl_bundle.schema.mutability_mask[2] == 0.0
    assert nsl_bundle.schema.mutability_mask[3] == 0.0
    assert nsl_bundle.schema.mutability_mask[0] == 1.0


# ---------------------------------------------------------------------------
# 2. Model Architecture, Determinism & Gradient Flow Tests
# ---------------------------------------------------------------------------


def test_model_architecture_gradient_flow_and_singleton_batch() -> None:
    set_deterministic_seed(99)
    model_a = RobustIDSNet(input_dim=20, hidden_dims=(64, 32), num_classes=4)
    set_deterministic_seed(99)
    model_b = RobustIDSNet(input_dim=20, hidden_dims=(64, 32), num_classes=4)

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)

    # Singleton batch safety in train() mode with BatchNorm1d
    model_a.train()
    x_single = torch.rand(1, 20, requires_grad=True)
    logits_single, z_single = model_a(x_single, return_features=True)
    assert logits_single.shape == (1, 4)
    assert z_single.shape == (1, 32)

    # Verify non-zero input gradient flow
    model_a.eval()
    x_batch = torch.rand(16, 20, requires_grad=True)
    y_batch = torch.randint(0, 4, (16,))
    loss = nn.CrossEntropyLoss()(model_a(x_batch), y_batch)
    loss.backward()
    assert x_batch.grad is not None
    assert torch.isfinite(x_batch.grad).all()
    assert float(torch.norm(x_batch.grad).item()) > 0.0


# ---------------------------------------------------------------------------
# 3. White-Box FGSM & PGD Constraint Invariance Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attack_kind", ["fgsm", "pgd"])
@pytest.mark.parametrize("eps", [0.0, 0.02, 0.05, 0.10, 0.20, 0.25])
def test_whitebox_mutability_mask_and_linf_bounds(
    lab_bundle: FlowDatasetBundle,
    trained_models,
    attack_kind: str,
    eps: float,
) -> None:
    baseline_model = trained_models[0]
    schema = lab_bundle.schema
    mask_t, xmin_t, xmax_t = schema.to_torch("cpu")

    x = torch.as_tensor(lab_bundle.X_test[:128], dtype=torch.float32)
    y = torch.as_tensor(lab_bundle.y_test[:128], dtype=torch.long)

    # Ensure attack leaves model training flag and parameter grads untouched
    baseline_model.train()
    for p in baseline_model.parameters():
        p.grad = None

    if attack_kind == "fgsm":
        x_adv = fgsm_attack(baseline_model, x, y, eps, mask_t, xmin_t, xmax_t)
    else:
        x_adv = pgd_attack(
            baseline_model, x, y, eps, mask_t, xmin_t, xmax_t, steps=12, seed=42
        )

    assert baseline_model.training is True
    for p in baseline_model.parameters():
        assert p.grad is None
    baseline_model.eval()

    # Exact bit-for-bit equality on immutable protocol features M == 0
    imm_idx = schema.immutable_indices
    assert torch.equal(x_adv[:, imm_idx], x[:, imm_idx])

    # Zero budget must return unperturbed input
    if eps == 0.0:
        assert torch.allclose(x_adv, x, atol=1e-7)

    # Verify L_inf ball and physical bounds [x_min, x_max]
    report = verify_perturbation_constraints(
        x_orig=x,
        x_adv=x_adv,
        eps=eps,
        mutability_mask=mask_t,
        x_min=xmin_t,
        x_max=xmax_t,
        raise_on_violation=True,
    )
    assert report["valid"] is True
    assert report["max_immutable_delta"] == 0.0
    assert float(report["max_linf_delta"]) <= eps + 1e-6
    assert float(report["max_bound_violation"]) <= 1e-6


def test_pgd_stronger_than_fgsm_and_vulnerability_report(
    lab_bundle: FlowDatasetBundle,
    trained_models,
) -> None:
    baseline_model = trained_models[0]
    fgsm_res = evaluate_whitebox_robustness(
        baseline_model,
        lab_bundle.X_test,
        lab_bundle.y_test,
        lab_bundle.schema,
        epsilons=[0.0, 0.10],
        attack="fgsm",
    )
    pgd_res = evaluate_whitebox_robustness(
        baseline_model,
        lab_bundle.X_test,
        lab_bundle.y_test,
        lab_bundle.schema,
        epsilons=[0.0, 0.10],
        attack="pgd",
        pgd_steps=15,
    )

    assert fgsm_res[0].accuracy > 0.90
    assert pgd_res[1].accuracy <= fgsm_res[1].accuracy + 1e-4
    assert pgd_res[1].accuracy < fgsm_res[0].accuracy - 0.35

    vuln = analyze_feature_vulnerability(
        baseline_model, lab_bundle.X_test[:256], lab_bundle.y_test[:256], lab_bundle.schema
    )
    assert np.all(vuln.effective_grad_sensitivity[lab_bundle.schema.immutable_indices] == 0.0)
    assert np.all(vuln.mean_pgd_perturbation[lab_bundle.schema.immutable_indices] == 0.0)
    assert float(np.max(vuln.effective_grad_sensitivity[lab_bundle.schema.mutable_indices])) > 0.0


# ---------------------------------------------------------------------------
# 4. Layered Defenses Verification Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["none", "gaussian", "median"])
def test_feature_squeezer_invariants_and_recovery(
    lab_bundle: FlowDatasetBundle,
    trained_models,
    mode: str,
) -> None:
    baseline_model = trained_models[0]
    schema = lab_bundle.schema
    squeezer = FeatureSqueezer(schema=schema, bit_depth=4, smoothing=mode).fit(lab_bundle.X_train)

    x_sample = lab_bundle.X_test[:200]
    x_sq = squeezer.transform_numpy(x_sample)

    # Immutable protocol features untouched
    assert np.array_equal(x_sq[:, schema.immutable_indices], x_sample[:, schema.immutable_indices])
    assert np.all(x_sq >= schema.x_min[None, :] - 1e-6)
    assert np.all(x_sq <= schema.x_max[None, :] + 1e-6)

    if mode == "none":
        # Pure 4-bit rounding yields at most 2^4 = 16 distinct levels in [0, 1]
        levels = np.unique(np.round(x_sq[:, schema.mutable_indices] * 15.0))
        assert levels.size <= 16

    if mode == "gaussian":
        # Verify Tensor interface & measurable recovery at eps = 0.10 without retraining
        x_t = torch.as_tensor(lab_bundle.X_test, dtype=torch.float32)
        y_t = torch.as_tensor(lab_bundle.y_test, dtype=torch.long)
        mask_t, xmin_t, xmax_t = schema.to_torch("cpu")
        x_adv = pgd_attack(baseline_model, x_t, y_t, 0.10, mask_t, xmin_t, xmax_t, steps=12, seed=5)

        raw_acc = float((baseline_model.predict(x_adv) == y_t).float().mean().item())
        sq_acc = float((squeezer.predict(baseline_model, x_adv) == y_t).float().mean().item())
        assert sq_acc > raw_acc + 0.05


def test_mahalanobis_ood_detector_positive_definite_and_auroc(
    lab_bundle: FlowDatasetBundle,
    trained_models,
) -> None:
    pgdat_model = trained_models[2]
    detector = MahalanobisOODDetector(target_fpr=0.05).fit(
        model=pgdat_model,
        X_train=lab_bundle.X_train,
        y_train=lab_bundle.y_train,
        X_val=lab_bundle.X_val,
    )

    assert detector.precision_ is not None
    eigvals = np.linalg.eigvalsh(detector.precision_)
    assert np.all(eigvals > 0.0), "Tied precision matrix Sigma^{-1} must be strictly SPD."

    # Check validation FPR calibration
    val_rej = detector.is_rejected(pgdat_model, lab_bundle.X_val)
    assert 0.01 <= float(np.mean(val_rej)) <= 0.09

    # Check OOD AUROC against unseen zero-day traffic + PGD evasions
    assert lab_bundle.X_ood is not None
    mask_t, xmin_t, xmax_t = lab_bundle.schema.to_torch("cpu")
    x_adv = pgd_attack(
        pgdat_model,
        torch.as_tensor(lab_bundle.X_test[:250], dtype=torch.float32),
        torch.as_tensor(lab_bundle.y_test[:250], dtype=torch.long),
        eps=0.12,
        mutability_mask=mask_t,
        x_min=xmin_t,
        x_max=xmax_t,
        steps=12,
        seed=77,
    ).numpy()

    in_scores = detector.anomaly_score(pgdat_model, lab_bundle.X_test)
    out_scores = detector.anomaly_score(pgdat_model, np.vstack([x_adv, lab_bundle.X_ood]))
    auroc = compute_ood_auroc(in_scores, out_scores)
    assert auroc >= 0.80


def test_pgdat_and_full_defense_stack_robustness_gains(
    lab_bundle: FlowDatasetBundle,
    trained_models,
) -> None:
    baseline_model, _, pgdat_model, _ = trained_models
    schema = lab_bundle.schema

    clean_base = evaluate_classifier(baseline_model, lab_bundle.X_test, lab_bundle.y_test)
    clean_pgdat = evaluate_classifier(pgdat_model, lab_bundle.X_test, lab_bundle.y_test)
    assert clean_base["accuracy"] >= 0.92
    assert clean_pgdat["accuracy"] >= 0.88

    rob_base = evaluate_whitebox_robustness(
        baseline_model,
        lab_bundle.X_test,
        lab_bundle.y_test,
        schema,
        epsilons=[0.10],
        attack="pgd",
        pgd_steps=15,
    )[0]
    rob_pgdat = evaluate_whitebox_robustness(
        pgdat_model,
        lab_bundle.X_test,
        lab_bundle.y_test,
        schema,
        epsilons=[0.10],
        attack="pgd",
        pgd_steps=15,
    )[0]

    # PGD-AT must deliver major white-box robustness gain over the unhardened baseline
    assert rob_pgdat.accuracy > rob_base.accuracy + 0.25

    # Full layered stack (PGD-AT + Mahalanobis Reject Option) further increases defense rate
    detector = MahalanobisOODDetector(target_fpr=0.04).fit(
        pgdat_model, lab_bundle.X_train, lab_bundle.y_train, lab_bundle.X_val
    )
    squeezer = FeatureSqueezer(schema=schema, bit_depth=5, smoothing="none").fit(lab_bundle.X_train)
    stack = HardenedIDSStack(model=pgdat_model, detector=detector, squeezer=squeezer)

    mask_t, xmin_t, xmax_t = schema.to_torch("cpu")
    x_adv = pgd_attack(
        pgdat_model,
        torch.as_tensor(lab_bundle.X_test, dtype=torch.float32),
        torch.as_tensor(lab_bundle.y_test, dtype=torch.long),
        eps=0.10,
        mutability_mask=mask_t,
        x_min=xmin_t,
        x_max=xmax_t,
        steps=15,
        seed=42,
    ).numpy()

    detailed = stack.predict_detailed(x_adv)
    assert np.any(detailed.predictions_with_reject == REJECT_LABEL)
    defended_rate = float(
        np.mean((detailed.raw_predictions == lab_bundle.y_test) | detailed.is_rejected)
    )
    assert defended_rate >= rob_pgdat.accuracy


def test_cli_entrypoints_end_to_end(tmp_path: Path) -> None:
    import main as cli_main

    ckpt_file = tmp_path / "baseline.pt"
    rc_train = cli_main.main(
        [
            "train-baseline",
            "--samples", "600",
            "--epochs", "5",
            "--checkpoint", str(ckpt_file),
            "--seed", "11",
        ]
    )
    assert rc_train == 0
    assert ckpt_file.exists()

    out_dir = tmp_path / "bench_out"
    rc_bench = cli_main.main(
        [
            "benchmark",
            "--samples", "600",
            "--epochs", "5",
            "--pgd-steps", "4",
            "--epsilons", "0.0", "0.10",
            "--output-dir", str(out_dir),
            "--seed", "11",
        ]
    )
    assert rc_bench == 0
    assert (out_dir / "accuracy_vs_epsilon.png").exists()
    assert (out_dir / "feature_vulnerability.png").exists()
    assert (out_dir / "benchmark_metrics.json").exists()

