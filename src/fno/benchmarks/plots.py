"""Plotting helpers for benchmark results: bar-chart comparisons across
models for one equation, and across equations for the cross-equation report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe -- no display server needed to save PNGs
import matplotlib.pyplot as plt

from fno.benchmarks.harness import BenchmarkResult

_COLORS = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2")
_METRICS: tuple[tuple[str, str], ...] = (
    ("n_parameters", "Parameters"),
    ("train_time_s", "Train time (s)"),
    ("inference_latency_ms", "Inference latency (ms)"),
    ("test_l2_error", "Test relative L2 error"),
)


def plot_benchmark_summary(
    results: list[BenchmarkResult], equation_name: str, save_path: str | Path
) -> Path:
    """2x2 grid comparing params/train-time/inference-latency/test-error
    across all models benchmarked for one equation."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    names = [r.name for r in results]
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(names))]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{equation_name}: FNO vs DeepONet vs PINN")

    for ax, (attr, label) in zip(axes.flat, _METRICS, strict=True):
        values = [getattr(r, attr) for r in results]
        ax.bar(names, values, color=colors)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_cross_equation_errors(
    results_by_equation: dict[str, list[BenchmarkResult]], save_path: str | Path
) -> Path:
    """Grouped bar chart of test_l2_error across equations, one bar group per model."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    equations = list(results_by_equation)
    model_names = sorted(
        {r.name for results in results_by_equation.values() for r in results}
    )
    n_models = max(len(model_names), 1)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model_name in enumerate(model_names):
        values = []
        for equation in equations:
            match = next(
                (r for r in results_by_equation[equation] if r.name == model_name),
                None,
            )
            values.append(match.test_l2_error if match else float("nan"))
        offsets = [x + i * width for x in range(len(equations))]
        ax.bar(
            offsets,
            values,
            width=width,
            label=model_name,
            color=_COLORS[i % len(_COLORS)],
        )

    ax.set_xticks([x + width * (n_models - 1) / 2 for x in range(len(equations))])
    ax.set_xticklabels(equations)
    ax.set_ylabel("Test relative L2 error")
    ax.set_title("Cross-equation comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
