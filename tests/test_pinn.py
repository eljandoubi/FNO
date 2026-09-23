"""Unit tests for fno.models.pinn / fno.training.pinn.

The residual test verifies autograd-computed derivatives against hand-derived
calculus on a known polynomial (not a trained network), independent of any
training dynamics. The training tests use the classic Raissi et al. Burgers
benchmark setup (domain [-1, 1], nu=0.01/pi, IC=-sin(pi*x)).
"""

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from fno.data.pdebench import Burgers1DDataset
from fno.models.pinn import PINN
from fno.training.pinn import burgers_pinn_loss, burgers_residual, train_pinn_burgers

BURGERS_FILE = Path("data/raw/pdebench-burgers1d/1D_Burgers_Sols_Nu0.01.hdf5")


def test_pinn_output_shape():
    model = PINN(in_dim=2, out_dim=1, hidden_dim=16, n_layers=3)
    coords = torch.randn(10, 2)
    out = model(coords)
    assert out.shape == (10, 1)


class _QuadraticToy(nn.Module):
    """u(x, t) = x^2 * t -- a fixed, non-trained function with known derivatives,
    used to verify the autograd residual formula independent of training."""

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        t = coords[:, 1:2]
        return x**2 * t


def test_burgers_residual_matches_analytic_derivatives():
    model = _QuadraticToy()
    x = torch.linspace(0.1, 0.9, 5).unsqueeze(-1).requires_grad_(True)
    t = torch.linspace(0.1, 0.9, 5).unsqueeze(-1).requires_grad_(True)
    nu = 0.05

    residual = burgers_residual(model, x, t, nu)

    with torch.no_grad():
        u = x**2 * t
        u_x = 2 * x * t
        u_t = x**2
        u_xx = 2 * t
        expected = u_t + u * u_x - nu * u_xx

    assert torch.allclose(residual, expected, atol=1e-5)


def test_burgers_pinn_loss_gradients_flow():
    model = PINN(in_dim=2, out_dim=1, hidden_dim=16, n_layers=3)
    x_c = torch.rand(20, 1, requires_grad=True)
    t_c = torch.rand(20, 1, requires_grad=True)
    x_ic = torch.linspace(-1, 1, 10).unsqueeze(-1)
    u_ic = -torch.sin(torch.pi * x_ic)

    loss, residual_loss, ic_loss = burgers_pinn_loss(model, x_c, t_c, nu=0.01, x_ic=x_ic, u_ic=u_ic)
    loss.backward()

    grad = next(model.parameters()).grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert math.isfinite(residual_loss.item())
    assert math.isfinite(ic_loss.item())


def test_train_pinn_burgers_reduces_loss():
    torch.manual_seed(0)
    x_ic = torch.linspace(-1, 1, 50).unsqueeze(-1)
    u_ic = -torch.sin(torch.pi * x_ic)  # classic Raissi et al. Burgers benchmark IC
    model = PINN(in_dim=2, out_dim=1, hidden_dim=20, n_layers=4)

    history = train_pinn_burgers(
        model,
        x_ic,
        u_ic,
        nu=0.01 / math.pi,
        domain_x=(-1.0, 1.0),
        domain_t=(0.0, 1.0),
        n_collocation=200,
        epochs=200,
        lr=1e-3,
        device="cpu",
    )

    early = sum(history[:20]) / 20
    late = sum(history[-20:]) / 20
    assert late < early


@pytest.mark.skipif(not BURGERS_FILE.exists(), reason="Burgers sample not downloaded")
def test_train_pinn_on_real_burgers_ic():
    ds = Burgers1DDataset(BURGERS_FILE, split="train")
    _window, trajectory, grid = ds[0]
    x_ic = grid  # (X, 1)
    u_ic = trajectory[0].unsqueeze(-1)  # (X, 1) -- t=0 slice

    model = PINN(in_dim=2, out_dim=1, hidden_dim=20, n_layers=4)
    history = train_pinn_burgers(
        model,
        x_ic,
        u_ic,
        nu=0.01,  # matches this file's name (1D_Burgers_Sols_Nu0.01.hdf5)
        domain_x=(grid.min().item(), grid.max().item()),
        domain_t=(ds.t.min().item(), ds.t.max().item()),
        n_collocation=200,
        epochs=20,
        lr=1e-3,
        device="cpu",
    )

    assert len(history) == 20
    assert all(math.isfinite(v) for v in history)
