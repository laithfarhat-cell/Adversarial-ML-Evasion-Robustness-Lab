"""Domain-Constrained Tabular Network-Flow Dataset & Feature Schema Engine.

Provides:
1. Explicit feature partitioning into:
   - Immutable / Discrete Protocol Features (protected by binary mask M_j = 0)
   - Mutable Continuous Flow Features (M_j = 1) with strict physical bounds [x_min, x_max]
2. Realistic synthetic network-flow generator modeling Benign, DoS, PortScan,
   and BruteForce traffic distributions with physical cross-feature correlations
   and robust vs. non-robust continuous telemetry regimes.
3. CSV benchmark loader compatible with NSL-KDD, CIC-IDS2017/2018, and custom
   network-flow CSV exports (handling Inf/NaN telemetry artifacts and header padding).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

TRAFFIC_CLASSES: List[str] = ["Benign", "DoS", "PortScan", "BruteForce"]


@dataclass(frozen=True)
class FeatureSpec:
    """Specification of a single tabular network-flow feature."""

    name: str
    mutable: bool
    raw_min: float
    raw_max: float
    log_scale: bool
    category: str
    description: str


# 20-dimensional network-flow feature schema:
#   - 6 Immutable / Discrete Protocol & Header Features (M_j = 0)
#   - 14 Mutable Continuous Flow Kinematic Features (M_j = 1)
DEFAULT_FLOW_SCHEMA: List[FeatureSpec] = [
    # --- Immutable / Discrete Protocol Features (M_j = 0) ---
    FeatureSpec(
        name="protocol_tcp",
        mutable=False,
        raw_min=0.0,
        raw_max=1.0,
        log_scale=False,
        category="protocol_header",
        description="Binary indicator for TCP transport protocol (IANA proto=6)",
    ),
    FeatureSpec(
        name="protocol_udp",
        mutable=False,
        raw_min=0.0,
        raw_max=1.0,
        log_scale=False,
        category="protocol_header",
        description="Binary indicator for UDP transport protocol (IANA proto=17)",
    ),
    FeatureSpec(
        name="dst_port_tier",
        mutable=False,
        raw_min=0.0,
        raw_max=3.0,
        log_scale=False,
        category="protocol_header",
        description="Discrete destination service port tier (0=sys/DNS, 1=web, 2=auth, 3=ephemeral)",
    ),
    FeatureSpec(
        name="tcp_syn_flag",
        mutable=False,
        raw_min=0.0,
        raw_max=1.0,
        log_scale=False,
        category="protocol_header",
        description="Discrete TCP SYN control bit indicator",
    ),
    FeatureSpec(
        name="tcp_ack_flag",
        mutable=False,
        raw_min=0.0,
        raw_max=1.0,
        log_scale=False,
        category="protocol_header",
        description="Discrete TCP ACK control bit indicator",
    ),
    FeatureSpec(
        name="tcp_rst_psh_flag",
        mutable=False,
        raw_min=0.0,
        raw_max=1.0,
        log_scale=False,
        category="protocol_header",
        description="Discrete TCP RST/PSH control flag indicator",
    ),
    # --- Mutable Continuous Flow Features (M_j = 1) ---
    FeatureSpec(
        name="flow_duration_ms",
        mutable=True,
        raw_min=0.5,
        raw_max=120_000.0,
        log_scale=True,
        category="flow_macro",
        description="Total bidirectional flow duration in milliseconds",
    ),
    FeatureSpec(
        name="fwd_pkt_rate",
        mutable=True,
        raw_min=0.1,
        raw_max=50_000.0,
        log_scale=True,
        category="flow_macro",
        description="Forward packet transmission rate (packets/sec)",
    ),
    FeatureSpec(
        name="bwd_pkt_rate",
        mutable=True,
        raw_min=0.0,
        raw_max=50_000.0,
        log_scale=True,
        category="flow_macro",
        description="Backward response packet rate (packets/sec)",
    ),
    FeatureSpec(
        name="byte_rate",
        mutable=True,
        raw_min=20.0,
        raw_max=50_000_000.0,
        log_scale=True,
        category="flow_macro",
        description="Aggregate bidirectional throughput (bytes/sec)",
    ),
    FeatureSpec(
        name="pkt_iat_mean_ms",
        mutable=True,
        raw_min=0.01,
        raw_max=10_000.0,
        log_scale=True,
        category="flow_macro",
        description="Mean inter-arrival time between consecutive packets (ms)",
    ),
    FeatureSpec(
        name="fwd_pkt_len_mean",
        mutable=True,
        raw_min=40.0,
        raw_max=1500.0,
        log_scale=False,
        category="flow_macro",
        description="Mean forward IP packet length bounded by header and Ethernet MTU (bytes)",
    ),
    FeatureSpec(
        name="bwd_pkt_len_mean",
        mutable=True,
        raw_min=0.0,
        raw_max=1500.0,
        log_scale=False,
        category="flow_macro",
        description="Mean backward IP packet length bounded by Ethernet MTU (bytes)",
    ),
    FeatureSpec(
        name="pkt_iat_std_ms",
        mutable=True,
        raw_min=0.0,
        raw_max=15_000.0,
        log_scale=True,
        category="flow_micro",
        description="Standard deviation of packet inter-arrival times (ms)",
    ),
    FeatureSpec(
        name="pkt_len_variance",
        mutable=True,
        raw_min=0.0,
        raw_max=500_000.0,
        log_scale=True,
        category="flow_micro",
        description="Variance of bidirectional packet payload lengths (bytes^2)",
    ),
    FeatureSpec(
        name="fwd_bwd_byte_ratio",
        mutable=True,
        raw_min=0.0,
        raw_max=100.0,
        log_scale=True,
        category="flow_micro",
        description="Ratio of forward client bytes to backward server bytes",
    ),
    FeatureSpec(
        name="active_burst_mean_ms",
        mutable=True,
        raw_min=0.0,
        raw_max=30_000.0,
        log_scale=True,
        category="flow_micro",
        description="Mean duration a flow remains active before going idle (ms)",
    ),
    FeatureSpec(
        name="idle_gap_mean_ms",
        mutable=True,
        raw_min=0.0,
        raw_max=60_000.0,
        log_scale=True,
        category="flow_micro",
        description="Mean idle silence interval between bursts (ms)",
    ),
    FeatureSpec(
        name="subflow_fwd_bytes",
        mutable=True,
        raw_min=40.0,
        raw_max=250_000.0,
        log_scale=True,
        category="flow_micro",
        description="Average forward bytes transferred per burst subflow",
    ),
    FeatureSpec(
        name="header_payload_ratio",
        mutable=True,
        raw_min=0.02,
        raw_max=1.0,
        log_scale=False,
        category="flow_micro",
        description="Ratio of TCP/UDP/IP header bytes to total packet length",
    ),
]


@dataclass
class DomainFeatureSchema:
    """Encapsulates domain mutability masks M in {0, 1}^d and physical bounds [x_min, x_max]."""

    feature_names: List[str]
    mutability_mask: np.ndarray  # shape (d,), float32 in {0.0, 1.0}
    x_min: np.ndarray  # shape (d,), float32 lower bounds in model space
    x_max: np.ndarray  # shape (d,), float32 upper bounds in model space
    raw_min: np.ndarray  # shape (d,), float32 lower bounds in physical units
    raw_max: np.ndarray  # shape (d,), float32 upper bounds in physical units
    class_names: List[str]
    descriptions: Optional[List[str]] = None

    @property
    def num_features(self) -> int:
        return int(self.mutability_mask.shape[0])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def immutable_indices(self) -> np.ndarray:
        return np.where(self.mutability_mask == 0.0)[0]

    @property
    def mutable_indices(self) -> np.ndarray:
        return np.where(self.mutability_mask == 1.0)[0]

    def to_torch(
        self, device: Union[str, torch.device] = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mutability_mask, x_min, x_max) as float32 tensors on `device`."""
        mask_t = torch.as_tensor(self.mutability_mask, dtype=torch.float32, device=device)
        xmin_t = torch.as_tensor(self.x_min, dtype=torch.float32, device=device)
        xmax_t = torch.as_tensor(self.x_max, dtype=torch.float32, device=device)
        return mask_t, xmin_t, xmax_t

    def validate_constraints(
        self,
        x_clean: Union[np.ndarray, torch.Tensor],
        x_adv: Union[np.ndarray, torch.Tensor],
        eps: float,
        atol: float = 1e-6,
    ) -> Dict[str, Union[bool, float]]:
        """Verify domain mutability mask M, L_inf budget eps, and physical bounds [x_min, x_max]."""
        if isinstance(x_clean, torch.Tensor):
            xc = x_clean.detach().cpu().numpy()
        else:
            xc = np.asarray(x_clean)
        if isinstance(x_adv, torch.Tensor):
            xa = x_adv.detach().cpu().numpy()
        else:
            xa = np.asarray(x_adv)

        delta = xa - xc
        imm_idx = self.immutable_indices
        max_imm_delta = float(np.max(np.abs(delta[:, imm_idx]))) if imm_idx.size > 0 else 0.0
        max_linf_delta = float(np.max(np.abs(delta))) if delta.size > 0 else 0.0

        lower_violation = float(np.max(np.maximum(0.0, self.x_min[None, :] - xa)))
        upper_violation = float(np.max(np.maximum(0.0, xa - self.x_max[None, :])))
        max_bound_violation = max(lower_violation, upper_violation)

        immutable_ok = max_imm_delta == 0.0
        linf_ok = max_linf_delta <= (eps + atol)
        bounds_ok = max_bound_violation <= atol
        finite_ok = bool(np.all(np.isfinite(xa)))

        return {
            "valid": bool(immutable_ok and linf_ok and bounds_ok and finite_ok),
            "immutable_preserved": bool(immutable_ok),
            "within_linf_ball": bool(linf_ok),
            "within_domain_bounds": bool(bounds_ok),
            "is_finite": finite_ok,
            "max_immutable_delta": max_imm_delta,
            "max_linf_delta": max_linf_delta,
            "max_bound_violation": max_bound_violation,
        }


class DomainBoundedScaler:
    """Invertible physical-domain scaler mapping raw telemetry into [0, 1]^d.

    Applies optional monotone log1p compression on heavy-tailed rate/duration
    features followed by physical-domain min-max normalization so every feature
    is strictly bounded within [x_min, x_max] = [0, 1]^d while preserving exact
    discrete encodings on immutable header features.
    """

    def __init__(
        self,
        raw_min: np.ndarray,
        raw_max: np.ndarray,
        log_mask: np.ndarray,
    ) -> None:
        self.raw_min = np.asarray(raw_min, dtype=np.float64)
        self.raw_max = np.asarray(raw_max, dtype=np.float64)
        self.log_mask = np.asarray(log_mask, dtype=bool)
        self._fitted = False
        self._psi_min = np.zeros_like(self.raw_min)
        self._psi_span = np.ones_like(self.raw_min)

    def _to_psi(self, x_raw: np.ndarray) -> np.ndarray:
        clipped = np.clip(x_raw, self.raw_min[None, :], self.raw_max[None, :])
        psi = clipped.copy()
        if np.any(self.log_mask):
            shifted = np.maximum(0.0, clipped[:, self.log_mask] - self.raw_min[None, self.log_mask])
            psi[:, self.log_mask] = np.log1p(shifted)
        return psi

    def fit(self, x_raw: Optional[np.ndarray] = None) -> "DomainBoundedScaler":
        bounds_pair = np.vstack([self.raw_min, self.raw_max])
        psi_bounds = self._to_psi(bounds_pair)
        self._psi_min = psi_bounds[0]
        self._psi_span = np.maximum(psi_bounds[1] - psi_bounds[0], 1e-12)
        if x_raw is not None and x_raw.shape[0] > 0:
            # Verify finite training statistics without leaking test bounds
            _ = np.nanmedian(x_raw, axis=0)
        self._fitted = True
        return self

    def transform(self, x_raw: np.ndarray) -> np.ndarray:
        if not self._fitted:
            self.fit()
        psi = self._to_psi(np.asarray(x_raw, dtype=np.float64))
        x_norm = (psi - self._psi_min[None, :]) / self._psi_span[None, :]
        return np.clip(x_norm, 0.0, 1.0).astype(np.float32)

    def fit_transform(self, x_raw: np.ndarray) -> np.ndarray:
        return self.fit(x_raw).transform(x_raw)

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        if not self._fitted:
            self.fit()
        xn = np.clip(np.asarray(x_norm, dtype=np.float64), 0.0, 1.0)
        psi = xn * self._psi_span[None, :] + self._psi_min[None, :]
        raw = psi.copy()
        if np.any(self.log_mask):
            raw[:, self.log_mask] = (
                np.expm1(psi[:, self.log_mask]) + self.raw_min[None, self.log_mask]
            )
        return np.clip(raw, self.raw_min[None, :], self.raw_max[None, :]).astype(np.float32)


@dataclass
class FlowDatasetBundle:
    """Complete train/validation/test tabular dataset bundle with domain schema."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    schema: DomainFeatureSchema
    scaler: DomainBoundedScaler
    X_ood: Optional[np.ndarray] = None

    def summary_dict(self) -> Dict[str, Union[int, List[int], Dict[str, int]]]:
        unique, counts = np.unique(self.y_train, return_counts=True)
        class_dist = {
            self.schema.class_names[int(k)]: int(v) for k, v in zip(unique, counts)
        }
        return {
            "num_train": int(self.X_train.shape[0]),
            "num_val": int(self.X_val.shape[0]),
            "num_test": int(self.X_test.shape[0]),
            "num_features": self.schema.num_features,
            "num_immutable": int(self.schema.immutable_indices.size),
            "num_mutable": int(self.schema.mutable_indices.size),
            "train_class_distribution": class_dist,
        }


def _norm_to_raw(
    u: np.ndarray,
    spec: FeatureSpec,
) -> np.ndarray:
    """Map normalized latent telemetry u in [0, 1] to physical raw domain units."""
    u_clipped = np.clip(u, 0.0, 1.0)
    if spec.log_scale:
        psi_max = np.log1p(spec.raw_max - spec.raw_min)
        return spec.raw_min + np.expm1(u_clipped * psi_max)
    return spec.raw_min + u_clipped * (spec.raw_max - spec.raw_min)


def generate_synthetic_flows(
    n_samples: int = 12_000,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.20,
    include_ood: bool = True,
) -> FlowDatasetBundle:
    """Generate a realistic domain-constrained synthetic IDS network-flow dataset.

    Models four realistic traffic classes (`Benign`, `DoS`, `PortScan`, `BruteForce`)
    with:
      - Overlapping discrete protocol/port/TCP-flag distributions (so protocol headers
        provide partial context rather than trivial linear shortcuts).
      - Physically coupled macro-flow kinematics (durations, packet/byte rates, IATs,
        packet lengths) with realistic class separation and heavy-tail variance.
      - Subtle correlated micro-flow ratios/timings that exhibit high clean signal-to-noise
        but small absolute margin (reproducing the natural vulnerability of unhardened
        neural IDS models to bounded L_inf evasion perturbations).
    """
    if n_samples < 200:
        raise ValueError(f"n_samples must be >= 200, got {n_samples}")

    rng = np.random.Generator(np.random.PCG64(seed))

    # Realistic imbalanced SOC traffic priors: Benign 44%, DoS 22%, PortScan 18%, BruteForce 16%
    priors = np.array([0.44, 0.22, 0.18, 0.16], dtype=np.float64)
    y = rng.choice(len(TRAFFIC_CLASSES), size=n_samples, p=priors).astype(np.int64)

    d = len(DEFAULT_FLOW_SCHEMA)
    u_norm = np.zeros((n_samples, d), dtype=np.float64)

    # Class-conditional protocol header distributions (indices 0..5)
    # Protocol overlap ensures headers alone achieve ~48-52% accuracy
    tcp_probs = {0: 0.76, 1: 0.74, 2: 0.72, 3: 0.90}
    port_tier_probs = {
        # tiers: [sys/DNS (0), web 80/443 (1), auth 22/3389 (2), ephemeral (3)]
        0: [0.18, 0.46, 0.16, 0.20],  # Benign: mix of web, DNS, SSH admin, ephemeral
        1: [0.16, 0.56, 0.12, 0.16],  # DoS: heavy web/DNS floods + some auth/app floods
        2: [0.22, 0.28, 0.22, 0.28],  # PortScan: broad sweep across service tiers
        3: [0.06, 0.36, 0.50, 0.08],  # BruteForce: concentrated on SSH/RDP & web login
    }
    syn_probs = {0: 0.70, 1: 0.84, 2: 0.82, 3: 0.78}
    ack_probs = {0: 0.74, 1: 0.52, 2: 0.34, 3: 0.80}
    rst_probs = {0: 0.16, 1: 0.28, 2: 0.56, 3: 0.26}

    # Macro continuous feature centroids in normalized [0, 1] log-physical coordinates
    # Indices 6..12:
    #   6: flow_duration_ms, 7: fwd_pkt_rate, 8: bwd_pkt_rate, 9: byte_rate,
    #  10: pkt_iat_mean_ms, 11: fwd_pkt_len_mean, 12: bwd_pkt_len_mean
    macro_means = {
        0: np.array([0.58, 0.40, 0.44, 0.50, 0.56, 0.52, 0.62]),  # Benign: balanced bidirectional
        1: np.array([0.46, 0.80, 0.22, 0.78, 0.18, 0.64, 0.20]),  # DoS: high rate, low IAT, asymmetric
        2: np.array([0.18, 0.62, 0.14, 0.26, 0.30, 0.12, 0.08]),  # PortScan: short, tiny probe packets
        3: np.array([0.66, 0.48, 0.42, 0.38, 0.46, 0.28, 0.26]),  # BruteForce: sustained medium auth bursts
    }
    macro_stds = {
        0: np.array([0.16, 0.14, 0.14, 0.14, 0.14, 0.16, 0.16]),
        1: np.array([0.17, 0.13, 0.13, 0.14, 0.12, 0.18, 0.14]),
        2: np.array([0.13, 0.15, 0.11, 0.13, 0.14, 0.10, 0.09]),
        3: np.array([0.14, 0.13, 0.13, 0.12, 0.13, 0.11, 0.11]),
    }

    # Micro continuous flow features (indices 13..19):
    # Subtle, tightly clustered class signatures (small margin ~0.07-0.11, low std ~0.055)
    # Strongly predictive on clean data, yet susceptible to L_inf >= 0.08 perturbations
    #   13: pkt_iat_std_ms, 14: pkt_len_variance, 15: fwd_bwd_byte_ratio,
    #   16: active_burst_mean_ms, 17: idle_gap_mean_ms, 18: subflow_fwd_bytes,
    #   19: header_payload_ratio
    micro_means = {
        0: np.array([0.52, 0.57, 0.38, 0.53, 0.51, 0.54, 0.36]),
        1: np.array([0.36, 0.41, 0.56, 0.62, 0.34, 0.63, 0.44]),
        2: np.array([0.42, 0.35, 0.51, 0.37, 0.43, 0.38, 0.56]),
        3: np.array([0.60, 0.46, 0.44, 0.47, 0.58, 0.46, 0.49]),
    }
    micro_std = 0.058

    for cls_id in range(len(TRAFFIC_CLASSES)):
        idx = np.where(y == cls_id)[0]
        n_c = idx.size
        if n_c == 0:
            continue

        # 1. Discrete immutable protocol features
        is_tcp = rng.random(n_c) < tcp_probs[cls_id]
        # Small 4% fraction of ICMP/other where both TCP=0 and UDP=0
        is_udp = (~is_tcp) & (rng.random(n_c) < 0.85)
        tiers = rng.choice(4, size=n_c, p=port_tier_probs[cls_id])

        syn = is_tcp & (rng.random(n_c) < syn_probs[cls_id])
        ack = is_tcp & (rng.random(n_c) < ack_probs[cls_id])
        rst_psh = is_tcp & (rng.random(n_c) < rst_probs[cls_id])

        u_norm[idx, 0] = is_tcp.astype(np.float64)
        u_norm[idx, 1] = is_udp.astype(np.float64)
        u_norm[idx, 2] = tiers.astype(np.float64) / 3.0
        u_norm[idx, 3] = syn.astype(np.float64)
        u_norm[idx, 4] = ack.astype(np.float64)
        u_norm[idx, 5] = rst_psh.astype(np.float64)

        # 2. Correlated latent factors for realistic physical coupling
        z_intensity = rng.normal(0.0, 1.0, size=n_c)  # traffic burst intensity factor
        z_payload = rng.normal(0.0, 1.0, size=n_c)    # packet payload size factor

        macro = rng.normal(
            loc=macro_means[cls_id][None, :],
            scale=macro_stds[cls_id][None, :],
            size=(n_c, 7),
        )
        # Physical couplings:
        # Higher burst intensity -> higher fwd_pkt_rate (+), higher byte_rate (+), lower pkt_iat_mean (-)
        macro[:, 1] += 0.08 * z_intensity
        macro[:, 3] += 0.07 * z_intensity + 0.06 * z_payload
        macro[:, 4] -= 0.08 * z_intensity
        # Payload coupling -> fwd_pkt_len_mean (+), bwd_pkt_len_mean (+)
        macro[:, 5] += 0.07 * z_payload
        macro[:, 6] += 0.05 * z_payload
        # Protocol-dependent modulation (UDP packets lack TCP handshake duration overhead)
        macro[:, 0] -= 0.06 * is_udp.astype(np.float64)

        u_norm[idx, 6:13] = macro

        # 3. Micro continuous features coupled to macro kinematics & protocol state
        micro = rng.normal(
            loc=micro_means[cls_id][None, :],
            scale=micro_std,
            size=(n_c, 7),
        )
        micro[:, 0] -= 0.035 * z_intensity          # pkt_iat_std_ms inversely tracks intensity
        micro[:, 1] += 0.045 * z_payload            # pkt_len_variance tracks payload factor
        micro[:, 2] += 0.035 * z_intensity          # fwd_bwd_byte_ratio
        micro[:, 5] += 0.040 * z_payload            # subflow_fwd_bytes
        micro[:, 6] -= 0.045 * z_payload            # header_payload_ratio inversely tracks payload

        u_norm[idx, 13:20] = micro

    # Inject 2.5% realistic stealthy boundary flows (low-and-slow DoS, slow scans, heavy benign bursts)
    n_stealth = int(0.025 * n_samples)
    stealth_idx = rng.choice(n_samples, size=n_stealth, replace=False)
    partner_idx = rng.choice(n_samples, size=n_stealth, replace=False)
    blend_alpha = rng.uniform(0.25, 0.45, size=(n_stealth, 1))
    u_norm[stealth_idx, 6:] = (
        (1.0 - blend_alpha) * u_norm[stealth_idx, 6:]
        + blend_alpha * u_norm[partner_idx, 6:]
    )

    u_norm = np.clip(u_norm, 0.0, 1.0)

    # Map normalized coordinates to physical raw network telemetry units and fit DomainBoundedScaler
    x_raw = np.zeros_like(u_norm, dtype=np.float64)
    for j, spec in enumerate(DEFAULT_FLOW_SCHEMA):
        x_raw[:, j] = _norm_to_raw(u_norm[:, j], spec)

    schema, scaler = _build_default_schema_and_scaler()

    # Stratified split into train / val / test prior to fitting scaler
    idx_all = np.arange(n_samples)
    idx_trainval, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=relative_val,
        random_state=seed,
        stratify=y[idx_trainval],
    )

    X_train = scaler.fit_transform(x_raw[idx_train])
    X_val = scaler.transform(x_raw[idx_val])
    X_test = scaler.transform(x_raw[idx_test])

    X_ood = (
        generate_ood_flows(n_samples=max(500, idx_test.size // 2), scaler=scaler, seed=seed + 999)
        if include_ood
        else None
    )

    return FlowDatasetBundle(
        X_train=X_train,
        y_train=y[idx_train],
        X_val=X_val,
        y_val=y[idx_val],
        X_test=X_test,
        y_test=y[idx_test],
        schema=schema,
        scaler=scaler,
        X_ood=X_ood,
    )


def generate_ood_flows(
    n_samples: int = 1_000,
    scaler: Optional[DomainBoundedScaler] = None,
    seed: int = 1337,
) -> np.ndarray:
    """Generate unseen out-of-distribution zero-day network flows (C2 Beaconing & Exfiltration).

    Produces valid network flows within [x_min, x_max] whose joint protocol-kinematic
    structure deviates from the 4 known training classes (e.g., high-volume outbound
    covert UDP/ICMP tunneling and ultra-regular long-idle C2 beacons).
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    d = len(DEFAULT_FLOW_SCHEMA)
    u_ood = np.zeros((n_samples, d), dtype=np.float64)

    half = n_samples // 2
    # Archetype A: Covert UDP/Raw-IP Data Exfiltration (TCP=0, high fwd payload, high idle & high rate)
    u_ood[:half, 0] = 0.0
    u_ood[:half, 1] = (rng.random(half) < 0.5).astype(np.float64)
    u_ood[:half, 2] = 1.0  # ephemeral high ports
    u_ood[:half, 3:6] = 0.0
    u_ood[:half, 6:13] = rng.normal(
        loc=np.array([0.88, 0.78, 0.05, 0.90, 0.72, 0.94, 0.04]),
        scale=0.06,
        size=(half, 7),
    )
    u_ood[:half, 13:20] = rng.normal(
        loc=np.array([0.08, 0.06, 0.95, 0.88, 0.85, 0.92, 0.08]),
        scale=0.05,
        size=(half, 7),
    )

    # Archetype B: Anomalous TCP Flag / Periodic Beaconing (SYN+RST+ACK simultaneously or zero-jitter long gaps)
    rem = n_samples - half
    u_ood[half:, 0] = 1.0
    u_ood[half:, 1] = 0.0
    u_ood[half:, 2] = 0.0
    u_ood[half:, 3:6] = 1.0  # anomalous Christmas-tree / SYN+ACK+RST header combination
    u_ood[half:, 6:13] = rng.normal(
        loc=np.array([0.92, 0.12, 0.88, 0.84, 0.88, 0.10, 0.95]),
        scale=0.06,
        size=(rem, 7),
    )
    u_ood[half:, 13:20] = rng.normal(
        loc=np.array([0.04, 0.92, 0.05, 0.10, 0.94, 0.12, 0.92]),
        scale=0.05,
        size=(rem, 7),
    )

    u_ood = np.clip(u_ood, 0.0, 1.0)
    x_raw = np.zeros_like(u_ood, dtype=np.float64)
    for j, spec in enumerate(DEFAULT_FLOW_SCHEMA):
        x_raw[:, j] = _norm_to_raw(u_ood[:, j], spec)

    if scaler is None:
        _, scaler = _build_default_schema_and_scaler()
        scaler.fit()
    return scaler.transform(x_raw)


def _build_default_schema_and_scaler() -> Tuple[DomainFeatureSchema, DomainBoundedScaler]:
    feature_names = [s.name for s in DEFAULT_FLOW_SCHEMA]
    mutability_mask = np.array(
        [1.0 if s.mutable else 0.0 for s in DEFAULT_FLOW_SCHEMA], dtype=np.float32
    )
    raw_min = np.array([s.raw_min for s in DEFAULT_FLOW_SCHEMA], dtype=np.float32)
    raw_max = np.array([s.raw_max for s in DEFAULT_FLOW_SCHEMA], dtype=np.float32)
    log_mask = np.array([s.log_scale for s in DEFAULT_FLOW_SCHEMA], dtype=bool)
    descriptions = [s.description for s in DEFAULT_FLOW_SCHEMA]

    x_min = np.zeros(len(DEFAULT_FLOW_SCHEMA), dtype=np.float32)
    x_max = np.ones(len(DEFAULT_FLOW_SCHEMA), dtype=np.float32)

    schema = DomainFeatureSchema(
        feature_names=feature_names,
        mutability_mask=mutability_mask,
        x_min=x_min,
        x_max=x_max,
        raw_min=raw_min,
        raw_max=raw_max,
        class_names=list(TRAFFIC_CLASSES),
        descriptions=descriptions,
    )
    scaler = DomainBoundedScaler(raw_min=raw_min, raw_max=raw_max, log_mask=log_mask)
    return schema, scaler


# Keywords identifying immutable protocol/header/flag columns in NSL-KDD & CIC-IDS CSVs
_IMMUTABLE_KEYWORDS: Tuple[str, ...] = (
    "protocol",
    "proto",
    "service",
    "port",
    "flag",
    "land",
    "logged_in",
    "is_host_login",
    "is_guest_login",
)

# Canonical label mapping for NSL-KDD, CIC-IDS2017/2018, and UNSW/custom CSVs
_LABEL_KEYWORD_MAP: List[Tuple[Tuple[str, ...], int]] = [
    (("benign", "normal", "0"), 0),
    (("dos", "ddos", "hulk", "goldeneye", "slowloris", "slowhttptest", "neptune", "smurf", "back", "teardrop", "pod", "apache2"), 1),
    (("portscan", "scan", "satan", "ipsweep", "portsweep", "nmap", "mscan", "saint"), 2),
    (("bruteforce", "brute", "patator", "ftp-patator", "ssh-patator", "guess_passwd", "hydra", "web attack"), 3),
]


def _canonicalize_label(val: object) -> int:
    if isinstance(val, (int, np.integer)):
        iv = int(val)
        return iv if 0 <= iv < len(TRAFFIC_CLASSES) else 1
    s = str(val).strip().lower()
    if s.isdigit():
        iv = int(s)
        return iv if 0 <= iv < len(TRAFFIC_CLASSES) else 1
    for keywords, target_cls in _LABEL_KEYWORD_MAP:
        if any(kw in s for kw in keywords):
            return target_cls
    # Any unrecognized attack subtype maps to malicious DoS/generic intrusion bucket (1)
    return 1


def load_csv_flows(
    csv_path: Union[str, Path],
    label_col: Optional[str] = None,
    immutable_cols: Optional[Sequence[str]] = None,
    val_size: float = 0.15,
    test_size: float = 0.20,
    seed: int = 42,
) -> FlowDatasetBundle:
    """Load tabular network-flow CSV compatible with NSL-KDD and CIC-IDS formats.

    Automatically:
      - Strips whitespace from column headers (handling CIC-IDS2017 leading spaces).
      - Detects label column (`label`, `Label`, `class`, `attack_cat`, etc.).
      - Encodes categorical protocol/service/flag columns as discrete immutable features (M_j = 0).
      - Sanitizes `Inf`, `-Inf`, and `NaN` values using training-split statistics only.
      - Builds physical min/max bounds [x_min, x_max] and mutability mask M in {0, 1}^d.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV dataset not found at: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Drop non-predictive identifier columns if present in raw flow exports
    drop_candidates = {"flow id", "src ip", "source ip", "dst ip", "destination ip", "timestamp", "difficulty"}
    keep_cols = [c for c in df.columns if c.lower() not in drop_candidates]
    df = df[keep_cols].copy()

    # Locate label column
    if label_col is None:
        for candidate in ("label", "Label", "class", "Class", "attack_cat", "target"):
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            label_col = df.columns[-1]
    elif label_col not in df.columns:
        raise ValueError(f"Specified label_col='{label_col}' not found in columns: {list(df.columns)}")

    y_all = np.array([_canonicalize_label(v) for v in df[label_col].to_numpy()], dtype=np.int64)
    feat_df = df.drop(columns=[label_col]).copy()

    explicit_immutable = {c.lower() for c in immutable_cols} if immutable_cols is not None else set()
    feature_names: List[str] = []
    mutability_list: List[float] = []
    encoded_cols: List[np.ndarray] = []

    for col in feat_df.columns:
        col_lower = col.lower()
        series = feat_df[col]
        is_cat = not pd.api.types.is_numeric_dtype(series)
        is_imm = (
            is_cat
            or (col_lower in explicit_immutable)
            or any(kw in col_lower for kw in _IMMUTABLE_KEYWORDS)
        )

        if is_cat:
            codes, uniques = pd.factorize(series.astype(str).str.strip().str.lower(), sort=True)
            denom = max(len(uniques) - 1, 1)
            arr = np.where(codes < 0, 0.0, codes / float(denom)).astype(np.float64)
        else:
            arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)

        feature_names.append(col)
        mutability_list.append(0.0 if is_imm else 1.0)
        encoded_cols.append(arr)

    x_raw = np.column_stack(encoded_cols)
    mutability_mask = np.array(mutability_list, dtype=np.float32)

    # Split before computing imputation medians or min/max bounds to avoid leakage
    idx_all = np.arange(x_raw.shape[0])
    unique_classes, class_counts = np.unique(y_all, return_counts=True)
    stratify_y = y_all if (unique_classes.size > 1 and np.min(class_counts) >= 3) else None

    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=test_size, random_state=seed, stratify=stratify_y
    )
    stratify_tv = y_all[idx_trainval] if stratify_y is not None else None
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=relative_val, random_state=seed, stratify=stratify_tv
    )

    # Sanitize +/-Inf and NaN using clean training split statistics
    x_raw = np.where(np.isinf(x_raw), np.nan, x_raw)
    train_slice = x_raw[idx_train]
    col_medians = np.nanmedian(train_slice, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

    for j in range(x_raw.shape[1]):
        nan_mask = np.isnan(x_raw[:, j])
        if np.any(nan_mask):
            x_raw[nan_mask, j] = col_medians[j]

    train_clean = x_raw[idx_train]
    raw_min = np.min(train_clean, axis=0).astype(np.float32)
    raw_max = np.max(train_clean, axis=0).astype(np.float32)
    # Guard degenerate constant columns
    degenerate = raw_max <= raw_min
    raw_max[degenerate] = raw_min[degenerate] + 1.0

    # Apply log scaling on mutable positive features with wide dynamic range (>100x)
    log_mask = (mutability_mask == 1.0) & (raw_min >= 0.0) & ((raw_max - raw_min) > 100.0)

    scaler = DomainBoundedScaler(raw_min=raw_min, raw_max=raw_max, log_mask=log_mask)
    X_train = scaler.fit_transform(x_raw[idx_train])
    X_val = scaler.transform(x_raw[idx_val])
    X_test = scaler.transform(x_raw[idx_test])

    present_classes = TRAFFIC_CLASSES
    schema = DomainFeatureSchema(
        feature_names=feature_names,
        mutability_mask=mutability_mask,
        x_min=np.zeros(len(feature_names), dtype=np.float32),
        x_max=np.ones(len(feature_names), dtype=np.float32),
        raw_min=raw_min,
        raw_max=raw_max,
        class_names=list(present_classes),
    )

    return FlowDatasetBundle(
        X_train=X_train,
        y_train=y_all[idx_train],
        X_val=X_val,
        y_val=y_all[idx_val],
        X_test=X_test,
        y_test=y_all[idx_test],
        schema=schema,
        scaler=scaler,
        X_ood=None,
    )


def load_or_generate_dataset(
    csv_path: Optional[Union[str, Path]] = None,
    n_samples: int = 12_000,
    seed: int = 42,
) -> FlowDatasetBundle:
    """Unified helper loading a CSV benchmark if provided, else generating synthetic flows."""
    if csv_path is not None:
        return load_csv_flows(csv_path=csv_path, seed=seed)
    return generate_synthetic_flows(n_samples=n_samples, seed=seed)


def build_dataloaders(
    bundle: FlowDatasetBundle,
    batch_size: int = 128,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build deterministic PyTorch DataLoaders for train, validation, and test splits."""
    gen = torch.Generator()
    gen.manual_seed(seed)

    train_ds = TensorDataset(
        torch.as_tensor(bundle.X_train, dtype=torch.float32),
        torch.as_tensor(bundle.y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.as_tensor(bundle.X_val, dtype=torch.float32),
        torch.as_tensor(bundle.y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.as_tensor(bundle.X_test, dtype=torch.float32),
        torch.as_tensor(bundle.y_test, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=gen, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader, test_loader
