"""Unit tests for fno.models.fno (FNO1d/FNO2d), plus integration checks
against real downloaded PDEBench data when available."""

from pathlib import Path

import pytest
import torch

from fno.data.pdebench import Burgers1DDataset, DarcyFlowDataset
from fno.models.fno import FNO1d, FNO2d, SpectralConv1d, SpectralConv2d

DARCY_FILE = Path("data/raw/pdebench-darcy2d/2D_DarcyFlow_beta1.0_Train.hdf5")
BURGERS_FILE = Path("data/raw/pdebench-burgers1d/1D_Burgers_Sols_Nu0.01.hdf5")


def test_spectral_conv1d_shape():
    conv = SpectralConv1d(in_channels=4, out_channels=6, modes=8)
    x = torch.randn(2, 4, 64)
    out = conv(x)
    assert out.shape == (2, 6, 64)


def test_spectral_conv2d_shape():
    conv = SpectralConv2d(in_channels=4, out_channels=6, modes1=8, modes2=8)
    x = torch.randn(2, 4, 32, 32)
    out = conv(x)
    assert out.shape == (2, 6, 32, 32)


def test_fno1d_output_shape():
    model = FNO1d(modes=8, width=16, in_channels=2, out_channels=1, n_layers=2)
    x = torch.randn(3, 64, 2)  # (B, X, C_in)
    out = model(x)
    assert out.shape == (3, 64, 1)


def test_fno2d_output_shape():
    model = FNO2d(
        modes1=6, modes2=6, width=16, in_channels=3, out_channels=1, n_layers=2
    )
    x = torch.randn(2, 32, 32, 3)  # (B, H, W, C_in)
    out = model(x)
    assert out.shape == (2, 32, 32, 1)


def test_fno1d_zero_shot_super_resolution():
    """Same trained weights must run on a different spatial resolution."""
    model = FNO1d(modes=8, width=16, in_channels=2, out_channels=1, n_layers=2)
    low_res = model(torch.randn(1, 64, 2))
    high_res = model(torch.randn(1, 256, 2))
    assert low_res.shape == (1, 64, 1)
    assert high_res.shape == (1, 256, 1)


def test_fno2d_zero_shot_super_resolution():
    model = FNO2d(
        modes1=6, modes2=6, width=16, in_channels=3, out_channels=1, n_layers=2
    )
    low_res = model(torch.randn(1, 32, 32, 3))
    high_res = model(torch.randn(1, 64, 64, 3))
    assert low_res.shape == (1, 32, 32, 1)
    assert high_res.shape == (1, 64, 64, 1)


def test_fno1d_gradients_flow():
    model = FNO1d(modes=8, width=16, in_channels=2, out_channels=1, n_layers=2)
    x = torch.randn(2, 64, 2)
    out = model(x)
    out.sum().backward()
    grad = model.lifting.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_fno2d_gradients_flow():
    model = FNO2d(
        modes1=6, modes2=6, width=16, in_channels=3, out_channels=1, n_layers=2
    )
    x = torch.randn(2, 32, 32, 3)
    out = model(x)
    out.sum().backward()
    grad = model.lifting.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS backend not available"
)
def test_fno2d_runs_on_mps():
    model = FNO2d(
        modes1=6, modes2=6, width=16, in_channels=3, out_channels=1, n_layers=2
    ).to("mps")
    x = torch.randn(1, 32, 32, 3, device="mps")
    out = model(x)
    assert out.shape == (1, 32, 32, 1)
    assert out.device.type == "mps"


@pytest.mark.skipif(not DARCY_FILE.exists(), reason="Darcy sample not downloaded")
def test_fno2d_on_real_darcy_batch():
    ds = DarcyFlowDataset(DARCY_FILE, train=True, train_split=0.9)
    field, target, grid = ds[0]  # field/target: (1, 128, 128), grid: (128, 128, 2)
    x = torch.cat([field.permute(1, 2, 0), grid], dim=-1).unsqueeze(
        0
    )  # (1, 128, 128, 3)

    model = FNO2d(
        modes1=8, modes2=8, width=16, in_channels=3, out_channels=1, n_layers=2
    )
    out = model(x)
    assert out.shape == (1, *target.shape[1:], target.shape[0])


@pytest.mark.skipif(not BURGERS_FILE.exists(), reason="Burgers sample not downloaded")
def test_fno1d_on_real_burgers_batch():
    ds = Burgers1DDataset(BURGERS_FILE, initial_step=10, train=True, train_split=0.9)
    window, _full_trajectory, grid = ds[0]  # window: (10, 1024), grid: (1024, 1)
    x = torch.cat([window.permute(1, 0), grid], dim=-1).unsqueeze(0)  # (1, 1024, 11)

    model = FNO1d(modes=16, width=32, in_channels=11, out_channels=1, n_layers=2)
    out = model(x)
    assert out.shape == (1, 1024, 1)
