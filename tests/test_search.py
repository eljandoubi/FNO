"""Unit tests for fno.benchmarks.search."""

import h5py
import numpy as np
import pytest

from fno.benchmarks.harness import BenchmarkHyperparams
from fno.benchmarks.search import (
    SearchSpace,
    SearchTrial,
    _expand_search_space,
    load_best_hyperparams,
    save_search_result,
    search_hyperparams,
)


def test_expand_search_space_produces_cartesian_product():
    space = SearchSpace(
        lr=[1e-3, 5e-4],
        scheduler_step=[None],
        scheduler_gamma=[0.5],
        fno_modes=[None],
        fno_width=[16, 32],
        fno_n_layers=[None],
        deeponet_hidden_dim=[None],
        deeponet_latent_dim=[None],
    )

    candidates = _expand_search_space(space)

    assert len(candidates) == 4  # 2 lr x 2 fno_width
    assert all(isinstance(c, BenchmarkHyperparams) for c in candidates)
    combos = {(c.lr, c.fno_width) for c in candidates}
    assert combos == {(1e-3, 16), (1e-3, 32), (5e-4, 16), (5e-4, 32)}


def test_expand_search_space_single_value_lists_give_one_candidate():
    space = SearchSpace(
        lr=[1e-3],
        scheduler_step=[None],
        scheduler_gamma=[0.5],
        fno_modes=[None],
        fno_width=[None],
        fno_n_layers=[None],
        deeponet_hidden_dim=[None],
        deeponet_latent_dim=[None],
    )
    assert len(_expand_search_space(space)) == 1


@pytest.fixture
def tiny_darcy_file(tmp_path):
    path = tmp_path / "darcy.hdf5"
    with h5py.File(path, "w") as f:
        f["nu"] = np.random.rand(20, 16, 16).astype(np.float32)
        f["tensor"] = np.random.rand(20, 16, 16).astype(np.float32)
        f["x-coordinate"] = np.linspace(0, 1, 16, dtype=np.float32)
        f["y-coordinate"] = np.linspace(0, 1, 16, dtype=np.float32)
    return path


def test_search_hyperparams_tries_every_candidate_and_skips_pinn(tiny_darcy_file):
    space = SearchSpace(
        lr=[1e-3, 5e-4],
        scheduler_step=[None],
        scheduler_gamma=[0.5],
        fno_modes=[None],
        fno_width=[None],
        fno_n_layers=[None],
        deeponet_hidden_dim=[None],
        deeponet_latent_dim=[None],
    )

    best_hp, trials = search_hyperparams(
        "darcy", tiny_darcy_file, space, n_train=4, n_test=2, epochs=1, device="cpu"
    )

    assert len(trials) == 2
    assert isinstance(best_hp, BenchmarkHyperparams)
    assert best_hp.lr in (1e-3, 5e-4)
    for trial in trials:
        assert isinstance(trial, SearchTrial)
        assert set(trial.scores) == {"FNO2d", "DeepONet"}  # no PINN row
        assert trial.combined_score == pytest.approx(sum(trial.scores.values()))
    assert min(t.combined_score for t in trials) == pytest.approx(
        next(t.combined_score for t in trials if t.hyperparams["lr"] == best_hp.lr)
    )


def test_save_load_search_result_roundtrip(tmp_path):
    best_hp = BenchmarkHyperparams(lr=5e-4, fno_width=16)
    trials = [
        SearchTrial(
            hyperparams={"lr": 5e-4}, scores={"FNO2d": 0.5}, combined_score=0.5
        ),
        SearchTrial(
            hyperparams={"lr": 1e-3}, scores={"FNO2d": 0.7}, combined_score=0.7
        ),
    ]
    path = tmp_path / "search.json"

    save_search_result(best_hp, trials, path)
    loaded = load_best_hyperparams(path)

    assert loaded == best_hp


def test_main_cli_prints_and_saves_results(tiny_darcy_file, tmp_path, capsys):
    from fno.benchmarks import search as search_module

    results_file = tmp_path / "search_result.json"
    search_module.main(
        [
            "darcy",
            "--file",
            str(tiny_darcy_file),
            "--n-train",
            "4",
            "--n-test",
            "2",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--lr",
            "1e-3",
            "--results-file",
            str(results_file),
        ]
    )

    captured = capsys.readouterr()
    assert "Best:" in captured.out
    assert results_file.exists()
    assert load_best_hyperparams(results_file).lr == pytest.approx(1e-3)
