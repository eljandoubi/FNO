"""Hyperparameter search: grid search over a small data subset (fast), then
hand the winning `BenchmarkHyperparams` to the caller for a full-data run.

This is a small, dependency-free grid search -- not a general-purpose HPO
library -- matching this project's "keep it simple, verify honestly"
philosophy. PINN is always skipped during search: it fits one instance (not
an operator), doesn't share FNO/DeepONet's architecture hyperparameters in a
meaningful way, and is cheap enough to just run once at full scale anyway.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fno.benchmarks.harness import (
    BenchmarkHyperparams,
    BenchmarkResult,
    run_burgers_benchmark,
    run_darcy_benchmark,
    run_navier_stokes_benchmark,
)

EQUATIONS: tuple[str, ...] = ("darcy", "burgers", "navier_stokes")

_SEARCH_SPACE_FIELDS: tuple[str, ...] = (
    "lr",
    "scheduler_step",
    "scheduler_gamma",
    "fno_modes",
    "fno_width",
    "fno_n_layers",
    "deeponet_hidden_dim",
    "deeponet_latent_dim",
)


@dataclass
class SearchSpace:
    """Each field is the list of candidate values to try for that knob; a
    single-element list means "don't search this knob, just use this value"
    (`[None]` means "use that function's per-equation default"). The full
    grid is the Cartesian product across all fields.
    """

    lr: list[float] = field(default_factory=lambda: [1e-3, 5e-4])
    scheduler_step: list[int | None] = field(default_factory=lambda: [None])
    scheduler_gamma: list[float] = field(default_factory=lambda: [0.5])
    fno_modes: list[int | None] = field(default_factory=lambda: [None])
    fno_width: list[int | None] = field(default_factory=lambda: [None, 16])
    fno_n_layers: list[int | None] = field(default_factory=lambda: [None])
    deeponet_hidden_dim: list[int | None] = field(default_factory=lambda: [None, 64])
    deeponet_latent_dim: list[int | None] = field(default_factory=lambda: [None])


@dataclass
class SearchTrial:
    hyperparams: dict
    scores: dict[str, float]  # model name -> test_l2_error
    combined_score: float  # sum of scores.values() -- lower is better


def _expand_search_space(space: SearchSpace) -> list[BenchmarkHyperparams]:
    value_lists = [getattr(space, name) for name in _SEARCH_SPACE_FIELDS]
    return [
        BenchmarkHyperparams(**dict(zip(_SEARCH_SPACE_FIELDS, combo, strict=True)))
        for combo in itertools.product(*value_lists)
    ]


def _run_trial(
    equation: str,
    file_path_or_paths: str | Path | Sequence[str | Path],
    hp: BenchmarkHyperparams,
    n_train: int,
    n_test: int,
    epochs: int,
    device: str | None,
) -> list[BenchmarkResult]:
    if equation == "darcy":
        return run_darcy_benchmark(
            file_path_or_paths,
            n_train=n_train,
            n_test=n_test,
            epochs=epochs,
            device=device,
            hyperparams=hp,
            skip_pinn=True,
        )
    if equation == "burgers":
        return run_burgers_benchmark(
            file_path_or_paths,
            n_train=n_train,
            n_test=n_test,
            epochs=epochs,
            device=device,
            hyperparams=hp,
            skip_pinn=True,
        )
    if equation == "navier_stokes":
        return run_navier_stokes_benchmark(
            file_path_or_paths,
            epochs=epochs,
            device=device,
            hyperparams=hp,
            skip_pinn=True,
        )
    raise ValueError(f"unknown equation: {equation!r}")


def search_hyperparams(
    equation: str,
    file_path_or_paths: str | Path | Sequence[str | Path],
    search_space: SearchSpace | None = None,
    n_train: int = 256,
    n_test: int = 64,
    epochs: int = 5,
    device: str | None = None,
) -> tuple[BenchmarkHyperparams, list[SearchTrial]]:
    """Grid-search `search_space` on a SMALL data subset (`n_train`/`n_test`,
    `epochs` all much smaller than a real run) to find good FNO/DeepONet
    hyperparameters fast, before spending real compute on the full dataset.
    Returns the winning `BenchmarkHyperparams` (lowest combined FNO+DeepONet
    test L2 error) plus the full trial log.
    """
    space = search_space or SearchSpace()
    candidates = _expand_search_space(space)

    trials: list[SearchTrial] = []
    best_hp = candidates[0]
    best_score = float("inf")

    for hp in candidates:
        results = _run_trial(
            equation, file_path_or_paths, hp, n_train, n_test, epochs, device
        )
        scores = {r.name: r.test_l2_error for r in results}
        combined_score = sum(scores.values())
        trials.append(
            SearchTrial(
                hyperparams=asdict(hp), scores=scores, combined_score=combined_score
            )
        )
        if combined_score < best_score:
            best_score = combined_score
            best_hp = hp

    return best_hp, trials


def save_search_result(
    best_hp: BenchmarkHyperparams, trials: list[SearchTrial], path: str | Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "best_hyperparams": asdict(best_hp),
                "trials": [asdict(t) for t in trials],
            },
            f,
            indent=2,
        )


def load_best_hyperparams(path: str | Path) -> BenchmarkHyperparams:
    with open(path) as f:
        data = json.load(f)
    return BenchmarkHyperparams(**data["best_hyperparams"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search FNO/DeepONet hyperparameters on a small data subset "
            "(fast), then print/save the winner for a full-data training run "
            "(see fno-pipeline --search for the automated two-phase version)."
        )
    )
    parser.add_argument("equation", choices=list(EQUATIONS))
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Dataset file (for navier_stokes, the first/only shard unless --navier-stokes-file is repeated).",
    )
    parser.add_argument(
        "--navier-stokes-file",
        type=Path,
        action="append",
        default=None,
        help="Additional Navier-Stokes shard paths (repeatable, combined with --file).",
    )
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--device", default=None, help="Defaults to auto-detected (cuda/mps/cpu)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        action="append",
        help="Repeatable candidate lr values (default: 1e-3, 5e-4).",
    )
    parser.add_argument(
        "--scheduler-step",
        type=int,
        action="append",
        help="Repeatable candidate values.",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        action="append",
        help="Repeatable candidate values.",
    )
    parser.add_argument(
        "--fno-modes", type=int, action="append", help="Repeatable candidate values."
    )
    parser.add_argument(
        "--fno-width",
        type=int,
        action="append",
        help="Repeatable candidate values (default: per-equation default, 16).",
    )
    parser.add_argument(
        "--fno-n-layers", type=int, action="append", help="Repeatable candidate values."
    )
    parser.add_argument(
        "--deeponet-hidden-dim",
        type=int,
        action="append",
        help="Repeatable candidate values (default: per-equation default, 64).",
    )
    parser.add_argument(
        "--deeponet-latent-dim",
        type=int,
        action="append",
        help="Repeatable candidate values.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Save the winning hyperparams + full trial log as JSON here.",
    )
    args = parser.parse_args(argv)

    default_space = SearchSpace()
    space = SearchSpace(
        lr=args.lr or default_space.lr,
        scheduler_step=args.scheduler_step or default_space.scheduler_step,
        scheduler_gamma=args.scheduler_gamma or default_space.scheduler_gamma,
        fno_modes=args.fno_modes or default_space.fno_modes,
        fno_width=args.fno_width or default_space.fno_width,
        fno_n_layers=args.fno_n_layers or default_space.fno_n_layers,
        deeponet_hidden_dim=args.deeponet_hidden_dim
        or default_space.deeponet_hidden_dim,
        deeponet_latent_dim=args.deeponet_latent_dim
        or default_space.deeponet_latent_dim,
    )

    file_arg: str | Path | Sequence[str | Path] = (
        [args.file, *args.navier_stokes_file]
        if args.equation == "navier_stokes" and args.navier_stokes_file
        else args.file
    )

    best_hp, trials = search_hyperparams(
        args.equation,
        file_arg,
        space,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
        device=args.device,
    )

    print(f"Tried {len(trials)} candidate(s):")
    for trial in trials:
        print(
            f"  {trial.hyperparams} -> {trial.scores} (combined={trial.combined_score:.4f})"
        )
    print(f"\nBest: {asdict(best_hp)}")

    if args.results_file:
        save_search_result(best_hp, trials, args.results_file)
        print(f"\nSaved to {args.results_file}")


if __name__ == "__main__":
    main()
