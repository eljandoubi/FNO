"""Unit tests for fno.data.pdebench, using synthetic HDF5 fixtures.

These validate parsing/shape logic against PDEBench's documented schema;
they are not a substitute for testing against a real downloaded file.
"""

import h5py
import numpy as np
import pytest

from fno.data.pdebench import Burgers1DDataset, DarcyFlowDataset, NavierStokes2DDataset


def _write_darcy_h5(path, n=10, nx=8, ny=8):
    with h5py.File(path, "w") as f:
        f["nu"] = np.random.rand(n, nx, ny).astype(np.float32)
        f["tensor"] = np.random.rand(n, nx, ny).astype(np.float32)
        f["x-coordinate"] = np.linspace(0, 1, nx, dtype=np.float32)
        f["y-coordinate"] = np.linspace(0, 1, ny, dtype=np.float32)


def _write_burgers_h5(path, n=10, t=5, x=16):
    with h5py.File(path, "w") as f:
        f["tensor"] = np.random.rand(n, t, x).astype(np.float32)
        f["x-coordinate"] = np.linspace(0, 1, x, dtype=np.float32)


def test_darcy_dataset_shapes(tmp_path):
    path = tmp_path / "darcy.hdf5"
    _write_darcy_h5(path, n=10, nx=8, ny=8)
    ds = DarcyFlowDataset(path, train=True, train_split=0.8)
    assert len(ds) == 8
    inp, target, grid = ds[0]
    assert inp.shape == (1, 8, 8)
    assert target.shape == (1, 8, 8)
    assert grid.shape == (8, 8, 2)


def test_darcy_dataset_train_test_split_sizes(tmp_path):
    path = tmp_path / "darcy.hdf5"
    _write_darcy_h5(path, n=10)
    train_ds = DarcyFlowDataset(path, train=True, train_split=0.8)
    test_ds = DarcyFlowDataset(path, train=False, train_split=0.8)
    assert len(train_ds) == 8
    assert len(test_ds) == 2


def test_darcy_dataset_handles_singleton_time_axis(tmp_path):
    path = tmp_path / "darcy_with_time.hdf5"
    with h5py.File(path, "w") as f:
        f["nu"] = np.random.rand(6, 1, 8, 8).astype(np.float32)
        f["tensor"] = np.random.rand(6, 1, 8, 8).astype(np.float32)
        f["x-coordinate"] = np.linspace(0, 1, 8, dtype=np.float32)
        f["y-coordinate"] = np.linspace(0, 1, 8, dtype=np.float32)
    ds = DarcyFlowDataset(path, train=True, train_split=1.0)
    inp, target, _grid = ds[0]
    assert inp.shape == (1, 8, 8)
    assert target.shape == (1, 8, 8)


def test_darcy_missing_key_raises(tmp_path):
    path = tmp_path / "bad.hdf5"
    with h5py.File(path, "w") as f:
        f["tensor"] = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(KeyError):
        DarcyFlowDataset(path)


def test_burgers_dataset_shapes(tmp_path):
    path = tmp_path / "burgers.hdf5"
    _write_burgers_h5(path, n=10, t=5, x=16)
    ds = Burgers1DDataset(path, initial_step=2, train=True, train_split=0.8)
    assert len(ds) == 8
    inp, target, grid = ds[0]
    assert inp.shape == (2, 16)
    assert target.shape == (5, 16)
    assert grid.shape == (16, 1)


def test_burgers_dataset_train_test_split_sizes(tmp_path):
    path = tmp_path / "burgers.hdf5"
    _write_burgers_h5(path, n=10)
    train_ds = Burgers1DDataset(path, train=True, train_split=0.7)
    test_ds = Burgers1DDataset(path, train=False, train_split=0.7)
    assert len(train_ds) == 7
    assert len(test_ds) == 3


def test_burgers_missing_key_raises(tmp_path):
    path = tmp_path / "bad.hdf5"
    with h5py.File(path, "w") as f:
        f["x-coordinate"] = np.zeros(4, dtype=np.float32)
    with pytest.raises(KeyError):
        Burgers1DDataset(path)


def _write_ns_incom_h5(path, n=4, t=6, h=8, w=8):
    with h5py.File(path, "w") as f:
        f["velocity"] = np.random.rand(n, t, h, w, 2).astype(np.float32)
        f["particles"] = np.random.rand(n, t, h, w, 1).astype(np.float32)
        f["force"] = np.random.rand(n, h, w, 2).astype(np.float32)
        f["t"] = np.tile(np.arange(t, dtype=np.float32), (n, 1))


def test_navier_stokes_dataset_shapes(tmp_path):
    path = tmp_path / "shard0.h5"
    _write_ns_incom_h5(path, n=4, t=6, h=8, w=8)
    ds = NavierStokes2DDataset(path, initial_step=2, train=True, train_split=1.0)
    assert len(ds) == 4
    inp, target, grid = ds[0]
    assert inp.shape == (2, 2, 8, 8)
    assert target.shape == (6, 2, 8, 8)
    assert grid.shape == (8, 8, 2)


def test_navier_stokes_dataset_concatenates_multiple_shards(tmp_path):
    path0 = tmp_path / "shard0.h5"
    path1 = tmp_path / "shard1.h5"
    _write_ns_incom_h5(path0, n=4)
    _write_ns_incom_h5(path1, n=4)
    ds = NavierStokes2DDataset([path0, path1], train=True, train_split=1.0)
    assert len(ds) == 8


def test_navier_stokes_dataset_spatial_stride(tmp_path):
    path = tmp_path / "shard0.h5"
    _write_ns_incom_h5(path, n=4, t=6, h=16, w=16)
    ds = NavierStokes2DDataset(
        path, initial_step=2, train=True, train_split=1.0, spatial_stride=4
    )
    inp, target, grid = ds[0]
    assert inp.shape == (2, 2, 4, 4)
    assert target.shape == (6, 2, 4, 4)
    assert grid.shape == (4, 4, 2)


def test_navier_stokes_dataset_train_test_split_sizes(tmp_path):
    path = tmp_path / "shard0.h5"
    _write_ns_incom_h5(path, n=4)
    train_ds = NavierStokes2DDataset(path, train=True, train_split=0.75)
    test_ds = NavierStokes2DDataset(path, train=False, train_split=0.75)
    assert len(train_ds) == 3
    assert len(test_ds) == 1


def test_navier_stokes_missing_key_raises(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f["t"] = np.zeros((4, 6), dtype=np.float32)
    with pytest.raises(KeyError):
        NavierStokes2DDataset(path)
