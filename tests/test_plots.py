"""Unit tests for fno.benchmarks.plots."""

from fno.benchmarks.harness import BenchmarkResult
from fno.benchmarks.plots import plot_benchmark_summary, plot_cross_equation_errors


def _toy_results(prefix: str) -> list[BenchmarkResult]:
    return [
        BenchmarkResult(
            name=f"{prefix}-FNO",
            n_parameters=100,
            train_time_s=1.0,
            inference_latency_ms=2.0,
            test_l2_error=0.1,
            superres_l2_error=0.15,
        ),
        BenchmarkResult(
            name=f"{prefix}-DeepONet",
            n_parameters=50,
            train_time_s=0.5,
            inference_latency_ms=1.0,
            test_l2_error=0.2,
            superres_l2_error=0.25,
        ),
        BenchmarkResult(
            name=f"{prefix}-PINN",
            n_parameters=10,
            train_time_s=2.0,
            inference_latency_ms=3.0,
            test_l2_error=0.3,
            note="fit to ONE instance",
        ),
    ]


def test_plot_benchmark_summary_creates_file(tmp_path):
    results = _toy_results("darcy")
    save_path = tmp_path / "darcy_summary.png"

    returned = plot_benchmark_summary(results, "darcy", save_path)

    assert returned == save_path
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_cross_equation_errors_creates_file(tmp_path):
    results_by_equation = {
        "darcy": _toy_results("darcy"),
        "burgers": _toy_results("burgers"),
    }
    save_path = tmp_path / "cross_equation.png"

    returned = plot_cross_equation_errors(results_by_equation, save_path)

    assert returned == save_path
    assert save_path.exists()
    assert save_path.stat().st_size > 0
