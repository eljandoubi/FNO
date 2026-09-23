"""Resumable end-to-end pipeline: download -> train -> evaluate -> benchmark
-> plot, for all three equations (Darcy, Burgers, Navier-Stokes) and all
three architectures (FNO, DeepONet, PINN).

Resumability works at two levels:
  1. Pipeline step level: which steps (download/benchmark/plot per equation,
     plus a final report) have completed is tracked in a JSON state file at
     `<run_dir>/pipeline_state.json`. Re-running with the same `--run-dir`
     skips already-completed steps.
  2. Training epoch level (`fno.training.trainer.fit`): each equation's
     benchmark step passes a `checkpoint_dir`, so even if "benchmark_<eq>"
     itself gets interrupted mid-training, retrying it resumes FNO/DeepONet
     from their last saved epoch instead of restarting. PINN is not
     checkpointed and retrains from scratch on retry (cheap, single instance).

Run with: `uv run fno-pipeline --run-dir runs/demo`
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fno.benchmarks.harness import (
    BenchmarkResult,
    format_summary,
    load_results,
    run_burgers_benchmark,
    run_darcy_benchmark,
    run_navier_stokes_benchmark,
    save_results,
)
from fno.benchmarks.plots import plot_benchmark_summary, plot_cross_equation_errors
from fno.data.download import main as download_main

EQUATIONS: tuple[str, ...] = ("darcy", "burgers", "navier_stokes")

# equation -> (download.py dataset key, variant, expected filename)
_DEFAULT_FILES: dict[str, tuple[str, str, str]] = {
    "darcy": ("pdebench-darcy2d", "beta1.0", "2D_DarcyFlow_beta1.0_Train.hdf5"),
    "burgers": ("pdebench-burgers1d", "nu0.01", "1D_Burgers_Sols_Nu0.01.hdf5"),
    "navier_stokes": (
        "pdebench-navierstokes2d",
        "shard0",
        "ns_incom_inhom_2d_512-0.h5",
    ),
}


@dataclass
class PipelineConfig:
    run_dir: Path
    equations: tuple[str, ...] = EQUATIONS
    data_root: Path = field(default_factory=lambda: Path("data"))
    n_train: int = 512
    n_test: int = 64
    epochs: int = 20
    pinn_epochs: int = 500
    device: str | None = None
    force: bool = False  # re-run every step, ignoring saved state


def _state_path(run_dir: Path) -> Path:
    return run_dir / "pipeline_state.json"


def _load_state(run_dir: Path) -> dict:
    path = _state_path(run_dir)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"steps": {}}


def _save_state(run_dir: Path, state: dict) -> None:
    path = _state_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)  # atomic -- avoids a half-written state file if interrupted


def _is_done(state: dict, step: str) -> bool:
    return state["steps"].get(step, {}).get("status") == "completed"


def _mark_done(run_dir: Path, state: dict, step: str) -> None:
    state["steps"][step] = {
        "status": "completed",
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _save_state(run_dir, state)


def _run_step(
    run_dir: Path, state: dict, step: str, force: bool, fn: Callable[[], None]
) -> None:
    if _is_done(state, step) and not force:
        print(f"[fno-pipeline] skip  {step} (already completed)")
        return
    print(f"[fno-pipeline] start {step}")
    fn()
    _mark_done(run_dir, state, step)
    print(f"[fno-pipeline] done  {step}")


def _results_path(run_dir: Path, equation: str) -> Path:
    return run_dir / "results" / f"{equation}.json"


def _plot_path(run_dir: Path, equation: str) -> Path:
    return run_dir / "plots" / f"{equation}_summary.png"


def _run_benchmark(
    equation: str, config: PipelineConfig, file_path: Path
) -> list[BenchmarkResult]:
    checkpoint_dir = config.run_dir / "checkpoints" / equation
    if equation == "darcy":
        return run_darcy_benchmark(
            file_path,
            n_train=config.n_train,
            n_test=config.n_test,
            epochs=config.epochs,
            pinn_epochs=config.pinn_epochs,
            device=config.device,
            checkpoint_dir=checkpoint_dir,
        )
    if equation == "burgers":
        return run_burgers_benchmark(
            file_path,
            n_train=config.n_train,
            n_test=config.n_test,
            epochs=config.epochs,
            pinn_epochs=config.pinn_epochs,
            device=config.device,
            checkpoint_dir=checkpoint_dir,
        )
    if equation == "navier_stokes":
        return run_navier_stokes_benchmark(
            file_path,
            epochs=config.epochs,
            pinn_epochs=config.pinn_epochs,
            device=config.device,
            checkpoint_dir=checkpoint_dir,
        )
    raise ValueError(f"unknown equation: {equation!r}")


def _write_report(
    run_dir: Path, results_by_equation: dict[str, list[BenchmarkResult]]
) -> Path:
    lines = ["# FNO / DeepONet / PINN benchmark report", ""]
    for equation, results in results_by_equation.items():
        lines.append(f"## {equation}")
        lines.append("")
        lines.append("```")
        lines.append(format_summary(results))
        lines.append("```")
        lines.append("")
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def _run_equation_pipeline(
    equation: str, config: PipelineConfig, run_dir: Path, state: dict
) -> list[BenchmarkResult]:
    dataset_key, variant, filename = _DEFAULT_FILES[equation]
    file_path = Path(config.data_root) / "raw" / dataset_key / filename

    def _download() -> None:
        download_main(
            [dataset_key, "--variant", variant, "--data-root", str(config.data_root)]
        )

    _run_step(run_dir, state, f"download_{equation}", config.force, _download)

    if not file_path.exists():
        raise RuntimeError(
            f"{equation}: expected {file_path} after the download step -- "
            "was the data directory removed after a previous successful run?"
        )

    results_path = _results_path(run_dir, equation)

    def _benchmark() -> None:
        save_results(_run_benchmark(equation, config, file_path), results_path)

    _run_step(run_dir, state, f"benchmark_{equation}", config.force, _benchmark)
    results = load_results(results_path)

    plot_path = _plot_path(run_dir, equation)

    def _plot() -> None:
        plot_benchmark_summary(results, equation, plot_path)

    _run_step(run_dir, state, f"plot_{equation}", config.force, _plot)

    return results


def run_pipeline(config: PipelineConfig) -> dict[str, list[BenchmarkResult]]:
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(run_dir)

    results_by_equation = {
        equation: _run_equation_pipeline(equation, config, run_dir, state)
        for equation in config.equations
    }

    def _report() -> None:
        _write_report(run_dir, results_by_equation)
        if len(results_by_equation) > 1:
            plot_cross_equation_errors(
                results_by_equation, run_dir / "plots" / "cross_equation_errors.png"
            )

    _run_step(run_dir, state, "report", config.force, _report)

    return results_by_equation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full download -> train -> evaluate -> benchmark -> plot "
            "pipeline for FNO/DeepONet/PINN on Darcy/Burgers/Navier-Stokes. "
            "Safe to interrupt and re-run with the same --run-dir: "
            "already-completed steps are skipped and in-progress model "
            "training resumes from its last checkpoint."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Output directory for state, checkpoints, results, and plots.",
    )
    parser.add_argument(
        "--equation",
        action="append",
        choices=list(EQUATIONS),
        dest="equations",
        help="Repeatable. Defaults to all three (darcy, burgers, navier_stokes).",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument(
        "--epochs", type=int, default=20, help="Epochs for FNO/DeepONet."
    )
    parser.add_argument("--pinn-epochs", type=int, default=500)
    parser.add_argument(
        "--device", default=None, help="Defaults to auto-detected (cuda/mps/cpu)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every step even if already marked completed.",
    )
    args = parser.parse_args(argv)

    config = PipelineConfig(
        run_dir=args.run_dir,
        equations=tuple(args.equations) if args.equations else EQUATIONS,
        data_root=args.data_root,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
        pinn_epochs=args.pinn_epochs,
        device=args.device,
        force=args.force,
    )

    results_by_equation = run_pipeline(config)

    print()
    for equation, results in results_by_equation.items():
        print(f"=== {equation} ===")
        print(format_summary(results))
        print()
    print(f"Report written to {config.run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
