"""Automated robustness benchmarking and publication-grade visualization suite."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.robustness_curves import (
        BenchmarkConfig,
        BenchmarkSuiteOutput,
        ConfigBenchmarkMetrics,
        plot_accuracy_vs_epsilon,
        plot_feature_vulnerability,
        render_rich_summary_table,
        run_full_benchmark,
    )

__all__ = [
    "BenchmarkConfig",
    "BenchmarkSuiteOutput",
    "ConfigBenchmarkMetrics",
    "plot_accuracy_vs_epsilon",
    "plot_feature_vulnerability",
    "render_rich_summary_table",
    "run_full_benchmark",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from benchmarks import robustness_curves

        return getattr(robustness_curves, name)
    raise AttributeError(f"module 'benchmarks' has no attribute {name!r}")
