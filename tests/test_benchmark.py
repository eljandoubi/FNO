"""Unit tests for fno.benchmarks.harness.

Uses tiny configs (few samples/epochs, small models) throughout -- these are
mechanics checks confirming the harness wires everything together correctly,
not a claim of converged/meaningful benchmark numbers.
"""

import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from fno.benchmarks.harness import (
    BenchmarkResult,
    _deeponet_superres_collate,
    _DeepONetSuperresView,
    _DownsampledDarcyView,
    benchmark_operator,
    benchmark_pinn_darcy,
    count_parameters,
    format_summary,
    run_darcy_benchmark,
)
from fno.data.pdebench import DarcyFlowDataset
from fno.models.fno import FNO2d
from fno.training import trainer
from fno.training.collate import darcy_collate

DARCY_FILE = Path("data/raw/pdebench-darcy2d/2D_DarcyFlow_beta1.0_Train.hdf5")


def test_count_parameters():
    model = torch.nn.Linear(4, 2)  # 4*2 weights + 2 bias = 10
    assert count_parameters(model) == 10


def test_downsampled_darcy_view_shapes(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "darcy.hdf5"
    with h5py.File(path, "w") as f:
        f["nu"] = np.random.rand(4, 16, 16).astype(np.float32)
        f["tensor"] = np.random.rand(4, 16, 16).astype(np.float32)
        f["x-coordinate"] = np.linspace(0, 1, 16, dtype=np.float32)
        f["y-coordinate"] = np.linspace(0, 1, 16, dtype=np.float32)

    base = DarcyFlowDataset(path, split="train", split_fractions=(1.0, 0.0, 0.0))
    view = _DownsampledDarcyView(base, stride=2)
    assert len(view) == len(base)
    field, target, grid = view[0]
    assert field.shape == (1, 8, 8)
    assert target.shape == (1, 8, 8)
    assert grid.shape == (8, 8, 2)


def test_deeponet_superres_view_keeps_field_low_res_but_target_grid_high_res(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "darcy.hdf5"
    with h5py.File(path, "w") as f:
        f["nu"] = np.random.rand(4, 16, 16).astype(np.float32)
        f["tensor"] = np.random.rand(4, 16, 16).astype(np.float32)
        f["x-coordinate"] = np.linspace(0, 1, 16, dtype=np.float32)
        f["y-coordinate"] = np.linspace(0, 1, 16, dtype=np.float32)

    base = DarcyFlowDataset(path, split="train", split_fractions=(1.0, 0.0, 0.0))
    view = _DeepONetSuperresView(base, stride=2)
    field, target, grid = view[0]
    assert field.shape == (1, 8, 8)  # downsampled -- matches the branch's fixed sensor count
    assert target.shape == (1, 16, 16)  # native resolution -- real ground truth, not approximated
    assert grid.shape == (16, 16, 2)

    batch = [view[0], view[1]]
    branch_input, trunk_input, targets = _deeponet_superres_collate(batch)
    assert branch_input.shape == (2, 64)  # 8*8
    assert trunk_input.shape == (256, 2)  # 16*16
    assert targets.shape == (2, 256, 1)


def test_format_summary_includes_all_rows_and_note():
    results = [
        BenchmarkResult(
            name="FNO2d", n_parameters=1000, train_time_s=1.5, inference_latency_ms=2.0, test_l2_error=0.05
        ),
        BenchmarkResult(
            name="PINN (single-instance fit)",
            n_parameters=500,
            train_time_s=3.0,
            inference_latency_ms=1.0,
            test_l2_error=0.1,
            note="fit to ONE instance, not an operator -- not directly comparable to the rows above",
        ),
    ]
    summary = format_summary(results)
    assert "FNO2d" in summary
    assert "PINN (single-instance fit)" in summary
    assert "not directly comparable" in summary
    assert "n/a" in summary  # FNO row has no superres_l2_error in this test


def test_benchmark_operator_mechanics():
    torch.manual_seed(0)
    n, h, w, c_in, c_out = 16, 8, 8, 3, 1
    x = torch.randn(n, h, w, c_in)
    y = x[..., :c_out]
    ds = torch.utils.data.TensorDataset(x, y)
    train_loader = DataLoader(Subset(ds, range(12)), batch_size=4)
    test_loader = DataLoader(Subset(ds, range(12, 16)), batch_size=4)

    model = FNO2d(modes1=4, modes2=4, width=8, in_channels=c_in, out_channels=c_out, n_layers=2)
    config = trainer.TrainConfig(epochs=2, device="cpu", use_wandb=False)

    result = benchmark_operator("FNO2d-toy", model, train_loader, test_loader, config, test_loader)

    assert result.n_parameters > 0
    assert result.train_time_s >= 0
    assert result.inference_latency_ms >= 0
    assert math.isfinite(result.test_l2_error)
    assert result.superres_l2_error is not None
    assert math.isfinite(result.superres_l2_error)


def test_benchmark_pinn_darcy_mechanics():
    torch.manual_seed(0)
    coefficient_field = torch.ones(8, 8)
    target_field = torch.zeros(8, 8)

    result = benchmark_pinn_darcy(
        "PINN-toy",
        coefficient_field,
        target_field,
        domain=(0.0, 1.0, 0.0, 1.0),
        forcing=1.0,
        epochs=5,
        device="cpu",
    )

    assert result.n_parameters > 0
    assert math.isfinite(result.test_l2_error)
    assert result.superres_l2_error is None
    assert "not directly comparable" in result.note


@pytest.mark.skipif(not DARCY_FILE.exists(), reason="Darcy sample not downloaded")
def test_run_darcy_benchmark_end_to_end_smoke():
    results = run_darcy_benchmark(
        DARCY_FILE, n_train=16, n_test=8, epochs=1, pinn_epochs=5, device="cpu"
    )

    assert len(results) == 3
    names = {r.name for r in results}
    assert names == {"FNO2d", "DeepONet", "PINN (single-instance fit)"}
    for r in results:
        assert r.n_parameters > 0
        assert math.isfinite(r.test_l2_error)

    fno_result = next(r for r in results if r.name == "FNO2d")
    assert fno_result.superres_l2_error is not None
    assert math.isfinite(fno_result.superres_l2_error)

    deeponet_result = next(r for r in results if r.name == "DeepONet")
    assert deeponet_result.superres_l2_error is not None
    assert math.isfinite(deeponet_result.superres_l2_error)

    # smoke-test the collate/model wiring directly too
    ds = DarcyFlowDataset(DARCY_FILE, split="test")
    loader = DataLoader(Subset(ds, range(4)), batch_size=2, collate_fn=darcy_collate)
    inputs, targets = next(iter(loader))
    assert inputs.shape[-1] == 3
    assert targets.shape[-1] == 1
