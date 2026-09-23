"""Unit tests for fno.pipeline -- the resumable download/benchmark/plot
orchestrator. Download and benchmark steps are faked (call-counted) so these
tests run fast and don't touch the network or do real training; one
real-data-conditional smoke test exercises the actual wiring end to end.
"""

import math
from pathlib import Path

import pytest

from fno import pipeline
from fno.benchmarks.harness import BenchmarkResult

DARCY_FILE = Path("data/raw/pdebench-darcy2d/2D_DarcyFlow_beta1.0_Train.hdf5")


def _fake_download_main_factory(data_root: Path):
    """Returns (fake_download_main, calls) -- the fake just touches the
    expected destination file instead of hitting the network."""
    calls: list[list[str]] = []

    def fake_download_main(argv: list[str]) -> None:
        calls.append(argv)
        dataset_key = argv[0]
        for key, _variant, filename in pipeline._DEFAULT_FILES.values():
            if key == dataset_key:
                dest = data_root / "raw" / key / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.touch()
                return
        raise AssertionError(f"unexpected dataset key {dataset_key!r}")

    return fake_download_main, calls


def _fake_benchmark_factory(name: str, *, fail_first_n: int = 0):
    """Returns (fake_fn, calls) -- fake_fn matches run_{darcy,burgers,navier_stokes}_benchmark's
    signature closely enough for the pipeline's call sites (file_path, **kwargs)."""
    calls: list[Path] = []

    def fake(file_path, **_kwargs) -> list[BenchmarkResult]:
        calls.append(Path(file_path))
        if len(calls) <= fail_first_n:
            raise RuntimeError(f"simulated interruption in {name} benchmark")
        return [
            BenchmarkResult(
                name=f"{name}-toy",
                n_parameters=10,
                train_time_s=0.01,
                inference_latency_ms=0.1,
                test_l2_error=0.5,
            )
        ]

    return fake, calls


@pytest.fixture
def patched_pipeline(monkeypatch, tmp_path):
    """Patches download + all three benchmark functions with fast fakes.
    Returns a dict of the call-count lists, keyed by equation name plus 'download'.
    """
    data_root = tmp_path / "data"
    fake_download, download_calls = _fake_download_main_factory(data_root)
    monkeypatch.setattr(pipeline, "download_main", fake_download)

    fake_darcy, darcy_calls = _fake_benchmark_factory("darcy")
    fake_burgers, burgers_calls = _fake_benchmark_factory("burgers")
    fake_ns, ns_calls = _fake_benchmark_factory("navier_stokes")
    monkeypatch.setattr(pipeline, "run_darcy_benchmark", fake_darcy)
    monkeypatch.setattr(pipeline, "run_burgers_benchmark", fake_burgers)
    monkeypatch.setattr(pipeline, "run_navier_stokes_benchmark", fake_ns)

    return {
        "data_root": data_root,
        "download": download_calls,
        "darcy": darcy_calls,
        "burgers": burgers_calls,
        "navier_stokes": ns_calls,
    }


def test_pipeline_runs_all_steps_and_writes_report(patched_pipeline, tmp_path):
    run_dir = tmp_path / "run"
    config = pipeline.PipelineConfig(
        run_dir=run_dir, data_root=patched_pipeline["data_root"], epochs=1
    )

    results = pipeline.run_pipeline(config)

    assert set(results) == set(pipeline.EQUATIONS)
    for equation in pipeline.EQUATIONS:
        assert patched_pipeline[equation] == [
            patched_pipeline["data_root"]
            / "raw"
            / pipeline._DEFAULT_FILES[equation][0]
            / pipeline._DEFAULT_FILES[equation][2]
        ]
        assert (run_dir / "results" / f"{equation}.json").exists()
        assert (run_dir / "plots" / f"{equation}_summary.png").exists()

    assert (run_dir / "report.md").exists()
    assert (run_dir / "plots" / "cross_equation_errors.png").exists()
    assert (run_dir / "pipeline_state.json").exists()


def test_rerunning_completed_pipeline_skips_all_steps(patched_pipeline, tmp_path):
    run_dir = tmp_path / "run"
    config = pipeline.PipelineConfig(
        run_dir=run_dir, data_root=patched_pipeline["data_root"], epochs=1
    )

    pipeline.run_pipeline(config)
    call_counts_after_first_run = {
        eq: len(patched_pipeline[eq]) for eq in pipeline.EQUATIONS
    }

    pipeline.run_pipeline(config)

    for equation in pipeline.EQUATIONS:
        assert len(patched_pipeline[equation]) == call_counts_after_first_run[equation]
    assert len(patched_pipeline["download"]) == len(pipeline.EQUATIONS)  # not re-called


def test_force_reruns_every_step(patched_pipeline, tmp_path):
    run_dir = tmp_path / "run"
    config = pipeline.PipelineConfig(
        run_dir=run_dir, data_root=patched_pipeline["data_root"], epochs=1
    )

    pipeline.run_pipeline(config)
    config.force = True
    pipeline.run_pipeline(config)

    for equation in pipeline.EQUATIONS:
        assert len(patched_pipeline[equation]) == 2
    assert len(patched_pipeline["download"]) == 2 * len(pipeline.EQUATIONS)


def test_pipeline_resumes_after_simulated_interruption(monkeypatch, tmp_path):
    """Burgers' benchmark step fails on its first attempt (simulating a crash
    mid-pipeline). Darcy (which ran first) must not be redone on retry, and
    Burgers must actually complete on the second attempt."""
    data_root = tmp_path / "data"
    fake_download, _download_calls = _fake_download_main_factory(data_root)
    monkeypatch.setattr(pipeline, "download_main", fake_download)

    fake_darcy, darcy_calls = _fake_benchmark_factory("darcy")
    fake_burgers, burgers_calls = _fake_benchmark_factory("burgers", fail_first_n=1)
    monkeypatch.setattr(pipeline, "run_darcy_benchmark", fake_darcy)
    monkeypatch.setattr(pipeline, "run_burgers_benchmark", fake_burgers)

    run_dir = tmp_path / "run"
    config = pipeline.PipelineConfig(
        run_dir=run_dir,
        data_root=data_root,
        equations=("darcy", "burgers"),
        epochs=1,
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.run_pipeline(config)

    state = pipeline._load_state(run_dir)
    assert pipeline._is_done(state, "benchmark_darcy")
    assert pipeline._is_done(state, "download_burgers")
    assert not pipeline._is_done(state, "benchmark_burgers")
    assert len(darcy_calls) == 1
    assert len(burgers_calls) == 1  # the failed attempt

    # Resume: Darcy must be skipped, Burgers must actually run again and succeed.
    results = pipeline.run_pipeline(config)

    assert len(darcy_calls) == 1  # unchanged -- not redone
    assert len(burgers_calls) == 2  # retried and this time succeeded
    assert set(results) == {"darcy", "burgers"}


def test_run_benchmark_dispatches_by_equation_and_rejects_unknown(tmp_path):
    with pytest.raises(ValueError, match="unknown equation"):
        pipeline._run_benchmark(
            "unknown",
            pipeline.PipelineConfig(run_dir=tmp_path),
            Path("x"),
            pipeline.BenchmarkHyperparams(),
        )


def test_pipeline_search_step_runs_before_benchmark_and_result_is_reused(
    monkeypatch, tmp_path
):
    data_root = tmp_path / "data"
    fake_download, _download_calls = _fake_download_main_factory(data_root)
    monkeypatch.setattr(pipeline, "download_main", fake_download)

    winning_hp = pipeline.BenchmarkHyperparams(lr=9.99e-5)
    search_calls = []

    def fake_search_hyperparams(equation, _file_path, _search_space, **_kwargs):
        search_calls.append(equation)
        return winning_hp, []

    monkeypatch.setattr(pipeline, "_search_hyperparams", fake_search_hyperparams)

    used_hyperparams = []

    def fake_darcy_benchmark(_file_path, **kwargs):
        used_hyperparams.append(kwargs["hyperparams"])
        return [
            pipeline.BenchmarkResult(
                name="FNO2d-toy",
                n_parameters=1,
                train_time_s=0.01,
                inference_latency_ms=0.1,
                test_l2_error=0.5,
            )
        ]

    monkeypatch.setattr(pipeline, "run_darcy_benchmark", fake_darcy_benchmark)

    run_dir = tmp_path / "run"
    config = pipeline.PipelineConfig(
        run_dir=run_dir,
        data_root=data_root,
        equations=("darcy",),
        epochs=1,
        search=True,
    )
    pipeline.run_pipeline(config)

    assert search_calls == ["darcy"]
    assert used_hyperparams == [winning_hp]
    assert (run_dir / "search" / "darcy.json").exists()

    # Force-redo the benchmark step without re-enabling search: the saved
    # search result must still be the one used, not config.hyperparams.
    config.search = False
    config.force = True
    pipeline.run_pipeline(config)

    assert search_calls == ["darcy"]  # search itself was NOT redone
    assert used_hyperparams == [winning_hp, winning_hp]


def test_cli_main_wires_args_into_pipeline_config(monkeypatch, tmp_path):
    captured_config = {}

    def fake_run_pipeline(config: pipeline.PipelineConfig):
        captured_config["config"] = config
        return {"darcy": []}

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)

    pipeline.main(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--equation",
            "darcy",
            "--epochs",
            "3",
            "--force",
        ]
    )

    config = captured_config["config"]
    assert config.run_dir == tmp_path / "run"
    assert config.equations == ("darcy",)
    assert config.epochs == 3
    assert config.force is True


def test_cli_main_wires_search_flags_into_pipeline_config(monkeypatch, tmp_path):
    captured_config = {}

    def fake_run_pipeline(config: pipeline.PipelineConfig):
        captured_config["config"] = config
        return {"darcy": []}

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)

    pipeline.main(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--equation",
            "darcy",
            "--search",
            "--search-n-train",
            "128",
            "--search-lr",
            "1e-3",
            "--search-lr",
            "1e-4",
            "--search-fno-width",
            "8",
        ]
    )

    config = captured_config["config"]
    assert config.search is True
    assert config.search_n_train == 128
    assert config.search_space.lr == [1e-3, 1e-4]
    assert config.search_space.fno_width == [8]


def test_cli_main_wires_search_strategy_flags_into_pipeline_config(
    monkeypatch, tmp_path
):
    captured_config = {}

    def fake_run_pipeline(config: pipeline.PipelineConfig):
        captured_config["config"] = config
        return {"darcy": []}

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)

    pipeline.main(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--equation",
            "darcy",
            "--search",
            "--search-strategy",
            "random",
            "--search-n-trials",
            "3",
            "--search-seed",
            "7",
        ]
    )

    config = captured_config["config"]
    assert config.search_strategy == "random"
    assert config.search_n_trials == 3
    assert config.search_seed == 7


@pytest.mark.skipif(not DARCY_FILE.exists(), reason="Darcy sample not downloaded")
def test_run_pipeline_darcy_only_end_to_end_smoke(tmp_path):
    run_dir = tmp_path / "run"
    config = pipeline.PipelineConfig(
        run_dir=run_dir,
        data_root=Path("data"),
        equations=("darcy",),
        n_train=16,
        n_test=8,
        epochs=1,
        pinn_epochs=5,
        device="cpu",
    )

    results = pipeline.run_pipeline(config)

    assert set(results) == {"darcy"}
    for result in results["darcy"]:
        assert math.isfinite(result.test_l2_error)
    assert (run_dir / "report.md").exists()
    assert (run_dir / "plots" / "darcy_summary.png").exists()

    # Re-running must skip every step (no crash, same results reloaded from disk).
    results_again = pipeline.run_pipeline(config)
    assert [r.name for r in results_again["darcy"]] == [
        r.name for r in results["darcy"]
    ]
