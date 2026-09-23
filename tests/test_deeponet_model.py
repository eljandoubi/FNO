"""Unit tests for fno.models.deeponet, plus integration checks against real
downloaded PDEBench data when available."""

from pathlib import Path

import pytest
import torch

from fno.data.pdebench import Burgers1DDataset, DarcyFlowDataset
from fno.models.deeponet import DeepONet

DARCY_FILE = Path("data/raw/pdebench-darcy2d/2D_DarcyFlow_beta1.0_Train.hdf5")
BURGERS_FILE = Path("data/raw/pdebench-burgers1d/1D_Burgers_Sols_Nu0.01.hdf5")


def test_deeponet_output_shape():
    model = DeepONet(
        branch_input_dim=64,
        trunk_input_dim=2,
        hidden_dim=32,
        latent_dim=16,
        out_channels=1,
    )
    branch_input = torch.randn(3, 64)  # (B, sensors)
    trunk_input = torch.randn(20, 2)  # (N query points, coord dim)
    out = model(branch_input, trunk_input)
    assert out.shape == (3, 20, 1)


def test_deeponet_multi_output_channels():
    model = DeepONet(
        branch_input_dim=64,
        trunk_input_dim=2,
        hidden_dim=32,
        latent_dim=16,
        out_channels=2,
    )
    branch_input = torch.randn(3, 64)
    trunk_input = torch.randn(20, 2)
    out = model(branch_input, trunk_input)
    assert out.shape == (3, 20, 2)


def test_deeponet_trunk_resolution_independence():
    """Same trained weights must accept a different number/set of query points --
    branch (input function sampling) is fixed, but the trunk (query locations) isn't.
    """
    model = DeepONet(
        branch_input_dim=64,
        trunk_input_dim=2,
        hidden_dim=32,
        latent_dim=16,
        out_channels=1,
    )
    branch_input = torch.randn(2, 64)
    coarse = model(branch_input, torch.randn(10, 2))
    fine = model(branch_input, torch.randn(500, 2))
    assert coarse.shape == (2, 10, 1)
    assert fine.shape == (2, 500, 1)


def test_deeponet_gradients_flow():
    model = DeepONet(
        branch_input_dim=32,
        trunk_input_dim=1,
        hidden_dim=16,
        latent_dim=8,
        out_channels=1,
    )
    branch_input = torch.randn(2, 32)
    trunk_input = torch.randn(10, 1)
    out = model(branch_input, trunk_input)
    out.sum().backward()
    grad = model.branch[0].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS backend not available"
)
def test_deeponet_runs_on_mps():
    model = DeepONet(
        branch_input_dim=32,
        trunk_input_dim=2,
        hidden_dim=16,
        latent_dim=8,
        out_channels=1,
    ).to("mps")
    branch_input = torch.randn(2, 32, device="mps")
    trunk_input = torch.randn(10, 2, device="mps")
    out = model(branch_input, trunk_input)
    assert out.shape == (2, 10, 1)
    assert out.device.type == "mps"


@pytest.mark.skipif(not DARCY_FILE.exists(), reason="Darcy sample not downloaded")
def test_deeponet_on_real_darcy_batch():
    ds = DarcyFlowDataset(DARCY_FILE, split="train")
    field, target, grid = ds[0]  # field/target: (1, 128, 128), grid: (128, 128, 2)
    branch_input = field.reshape(1, -1)  # (1, 128*128)
    trunk_input = grid.reshape(-1, 2)  # (128*128, 2)

    model = DeepONet(
        branch_input_dim=128 * 128,
        trunk_input_dim=2,
        hidden_dim=32,
        latent_dim=16,
        out_channels=1,
    )
    out = model(branch_input, trunk_input)
    assert out.shape == (1, 128 * 128, target.shape[0])


@pytest.mark.skipif(not BURGERS_FILE.exists(), reason="Burgers sample not downloaded")
def test_deeponet_on_real_burgers_batch():
    ds = Burgers1DDataset(BURGERS_FILE, initial_step=10, split="train")
    window, _full_trajectory, grid = ds[0]  # window: (10, 1024), grid: (1024, 1)
    branch_input = window.reshape(1, -1)  # (1, 10*1024)
    trunk_input = grid  # (1024, 1)

    model = DeepONet(
        branch_input_dim=10 * 1024,
        trunk_input_dim=1,
        hidden_dim=32,
        latent_dim=16,
        out_channels=1,
    )
    out = model(branch_input, trunk_input)
    assert out.shape == (1, 1024, 1)
