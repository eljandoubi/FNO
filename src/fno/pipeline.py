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
    BenchmarkHyperparams,
    BenchmarkResult,
    format_summary,
    load_results,
    run_burgers_benchmark,
    run_darcy_benchmark,
    run_navier_stokes_benchmark,
    save_results,
)
from fno.benchmarks.plots import plot_benchmark_summary, plot_cross_equation_errors
from fno.benchmarks.search import (
    STRATEGIES,
    SearchSpace,
    load_best_hyperparams,
    save_search_result,
)
from fno.benchmarks.search import search_hyperparams as _search_hyperparams
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
    hyperparams: BenchmarkHyperparams = field(default_factory=BenchmarkHyperparams)
    search: bool = False  # run a hyperparameter search before the full-data benchmark
    search_space: SearchSpace = field(default_factory=SearchSpace)
    search_n_train: int = 256
    search_n_test: int = 64
    search_epochs: int = 5
    search_strategy: str = "grid"  # "grid" or "random"
    search_n_trials: int = 10  # only used when search_strategy="random"
    search_seed: int = 0


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


def _search_result_path(run_dir: Path, equation: str) -> Path:
    return run_dir / "search" / f"{equation}.json"


def _run_benchmark(
    equation: str,
    config: PipelineConfig,
    file_path: Path,
    hyperparams: BenchmarkHyperparams,
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
            hyperparams=hyperparams,
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
            hyperparams=hyperparams,
        )
    if equation == "navier_stokes":
        return run_navier_stokes_benchmark(
            file_path,
            epochs=config.epochs,
            pinn_epochs=config.pinn_epochs,
            device=config.device,
            checkpoint_dir=checkpoint_dir,
            hyperparams=hyperparams,
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

    search_result_path = _search_result_path(run_dir, equation)
    if config.search:

        def _search() -> None:
            best_hp, trials = _search_hyperparams(
                equation,
                file_path,
                config.search_space,
                n_train=config.search_n_train,
                n_test=config.search_n_test,
                epochs=config.search_epochs,
                device=config.device,
                strategy=config.search_strategy,
                n_trials=config.search_n_trials,
                seed=config.search_seed,
            )
            save_search_result(best_hp, trials, search_result_path)

        _run_step(run_dir, state, f"search_{equation}", config.force, _search)

    # A prior search's result is reused on resume even if --search isn't
    # passed again, so it isn't silently discarded/redone.
    hyperparams = (
        load_best_hyperparams(search_result_path)
        if search_result_path.exists()
        else config.hyperparams
    )

    results_path = _results_path(run_dir, equation)

    def _benchmark() -> None:
        save_results(
            _run_benchmark(equation, config, file_path, hyperparams), results_path
        )

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
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for FNO/DeepONet/PINN."
    )
    parser.add_argument(
        "--scheduler-step",
        type=int,
        default=None,
        help="StepLR: decay lr every N epochs (unset: constant lr). PDEBench's own FNO baselines use 100.",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.5,
        help="StepLR decay factor, only used if --scheduler-step is set.",
    )
    parser.add_argument(
        "--fno-modes",
        type=int,
        default=None,
        help="Override FNO's Fourier mode count (per-equation default otherwise).",
    )
    parser.add_argument(
        "--fno-width",
        type=int,
        default=None,
        help="Override FNO's channel width (per-equation default otherwise).",
    )
    parser.add_argument(
        "--fno-n-layers",
        type=int,
        default=None,
        help="Override FNO's number of spectral layers (per-equation default otherwise).",
    )
    parser.add_argument(
        "--deeponet-hidden-dim",
        type=int,
        default=None,
        help="Override DeepONet's branch/trunk hidden width (default 128).",
    )
    parser.add_argument(
        "--deeponet-latent-dim",
        type=int,
        default=None,
        help="Override DeepONet's latent (dot-product) dimension (default 64).",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help=(
            "Before the full-data benchmark, grid-search FNO/DeepONet "
            "hyperparameters on a small data subset and use the winner "
            "instead of --lr/--fno-*/--deeponet-* above."
        ),
    )
    parser.add_argument("--search-n-train", type=int, default=256)
    parser.add_argument("--search-n-test", type=int, default=64)
    parser.add_argument("--search-epochs", type=int, default=5)
    parser.add_argument(
        "--search-lr",
        type=float,
        action="append",
        help="Repeatable candidate lr values for --search (default: 1e-3, 5e-4).",
    )
    parser.add_argument(
        "--search-fno-width",
        type=int,
        action="append",
        help="Repeatable candidate values for --search (default: per-equation default, 16).",
    )
    parser.add_argument(
        "--search-deeponet-hidden-dim",
        type=int,
        action="append",
        help="Repeatable candidate values for --search (default: per-equation default, 64).",
    )
    parser.add_argument(
        "--search-strategy",
        choices=list(STRATEGIES),
        default="grid",
        help=(
            "'grid': try every combination. 'random': sample up to "
            "--search-n-trials unique combinations (usually more efficient "
            "once more than a couple of --search-* flags have multiple "
            "values). Default: grid."
        ),
    )
    parser.add_argument(
        "--search-n-trials",
        type=int,
        default=10,
        help="Max candidates to try when --search-strategy=random (ignored for grid).",
    )
    parser.add_argument(
        "--search-seed", type=int, default=0, help="Random seed for --search-strategy=random."
    )
    args = parser.parse_args(argv)

    default_space = SearchSpace()
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
        hyperparams=BenchmarkHyperparams(
            lr=args.lr,
            scheduler_step=args.scheduler_step,
            scheduler_gamma=args.scheduler_gamma,
            fno_modes=args.fno_modes,
            fno_width=args.fno_width,
            fno_n_layers=args.fno_n_layers,
            deeponet_hidden_dim=args.deeponet_hidden_dim,
            deeponet_latent_dim=args.deeponet_latent_dim,
        ),
        search=args.search,
        search_space=SearchSpace(
            lr=args.search_lr or default_space.lr,
            fno_width=args.search_fno_width or default_space.fno_width,
            deeponet_hidden_dim=args.search_deeponet_hidden_dim
            or default_space.deeponet_hidden_dim,
        ),
        search_n_train=args.search_n_train,
        search_n_test=args.search_n_test,
        search_epochs=args.search_epochs,
        search_strategy=args.search_strategy,
        search_n_trials=args.search_n_trials,
        search_seed=args.search_seed,
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
