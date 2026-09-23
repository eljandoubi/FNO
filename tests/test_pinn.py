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

from fno.data.pdebench import Burgers1DDataset, DarcyFlowDataset, NavierStokes2DDataset
from fno.models.pinn import PINN
from fno.training.pinn import (
    _grad,
    _interpolate_field,
    burgers_pinn_loss,
    burgers_residual,
    darcy_pinn_loss,
    darcy_residual,
    navier_stokes_pinn_loss,
    navier_stokes_residual,
    train_pinn_burgers,
    train_pinn_darcy,
    train_pinn_navier_stokes,
)

BURGERS_FILE = Path("data/raw/pdebench-burgers1d/1D_Burgers_Sols_Nu0.01.hdf5")
DARCY_FILE = Path("data/raw/pdebench-darcy2d/2D_DarcyFlow_beta1.0_Train.hdf5")
NS_FILE = Path("data/raw/pdebench-navierstokes2d/ns_incom_inhom_2d_512-0.h5")


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

    loss, residual_loss, ic_loss = burgers_pinn_loss(
        model, x_c, t_c, nu=0.01, x_ic=x_ic, u_ic=u_ic
    )
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


def test_interpolate_field_matches_grid_values():
    """Query the interpolant at exact grid points -- if x/y axes were swapped
    this would return the transpose of `field` and fail (field is asymmetric)."""
    h, w = 5, 5
    field = torch.arange(h * w, dtype=torch.float32).reshape(h, w)
    xs = torch.linspace(0, 1, h)
    ys = torch.linspace(0, 1, w)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    x = xx.reshape(-1, 1)
    y = yy.reshape(-1, 1)

    out = _interpolate_field(field, x, y, domain=(0.0, 1.0, 0.0, 1.0))

    assert torch.allclose(out.reshape(h, w), field, atol=1e-4)


class _DarcyToy(nn.Module):
    """u(x, y) = x^2 + y^2 -- a fixed, non-trained function with known
    derivatives, used with a constant coefficient field to verify the
    darcy_residual formula independent of training."""

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        return x**2 + y**2


def test_darcy_residual_zero_for_manufactured_solution():
    """With a(x,y)=1 (constant) and u=x^2+y^2, -div(a*grad(u)) = -4 exactly,
    so forcing f=-4 should drive the residual to (near) zero everywhere."""
    model = _DarcyToy()
    coefficient_field = torch.ones(16, 16)
    domain = (0.0, 1.0, 0.0, 1.0)
    x = torch.linspace(0.1, 0.9, 5).unsqueeze(-1).requires_grad_(True)
    y = torch.linspace(0.1, 0.9, 5).unsqueeze(-1).requires_grad_(True)

    residual = darcy_residual(model, x, y, coefficient_field, domain, forcing=-4.0)

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-3)


def test_darcy_pinn_loss_gradients_flow():
    model = PINN(in_dim=2, out_dim=1, hidden_dim=16, n_layers=3)
    coefficient_field = torch.rand(16, 16)
    domain = (0.0, 1.0, 0.0, 1.0)
    x_c = torch.rand(20, 1, requires_grad=True)
    y_c = torch.rand(20, 1, requires_grad=True)
    x_bc = torch.rand(8, 1)
    y_bc = torch.rand(8, 1)

    loss, residual_loss, bc_loss = darcy_pinn_loss(
        model, x_c, y_c, coefficient_field, domain, x_bc, y_bc
    )
    loss.backward()

    grad = next(model.parameters()).grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert math.isfinite(residual_loss.item())
    assert math.isfinite(bc_loss.item())


def test_train_pinn_darcy_reduces_loss():
    torch.manual_seed(0)
    coefficient_field = torch.ones(16, 16)  # constant coefficient -- simplest case
    model = PINN(in_dim=2, out_dim=1, hidden_dim=20, n_layers=4)

    history = train_pinn_darcy(
        model,
        coefficient_field,
        domain=(0.0, 1.0, 0.0, 1.0),
        forcing=1.0,
        n_collocation=200,
        n_boundary=80,
        epochs=200,
        lr=1e-3,
        device="cpu",
    )

    early = sum(history[:20]) / 20
    late = sum(history[-20:]) / 20
    assert late < early


@pytest.mark.skipif(not DARCY_FILE.exists(), reason="Darcy sample not downloaded")
def test_train_pinn_on_real_darcy_coefficient_field():
    ds = DarcyFlowDataset(DARCY_FILE, split="train")
    coefficient_field, _target, grid = ds[0]  # coefficient_field: (1, 128, 128)

    model = PINN(in_dim=2, out_dim=1, hidden_dim=20, n_layers=4)
    history = train_pinn_darcy(
        model,
        coefficient_field.squeeze(0),  # (128, 128)
        domain=(
            grid[..., 0].min().item(),
            grid[..., 0].max().item(),
            grid[..., 1].min().item(),
            grid[..., 1].max().item(),
        ),
        forcing=1.0,  # standard FNO-paper Darcy Flow convention
        n_collocation=200,
        n_boundary=80,
        epochs=20,
        lr=1e-3,
        device="cpu",
    )

    assert len(history) == 20
    assert all(math.isfinite(v) for v in history)


def test_stream_function_satisfies_continuity_automatically():
    """u=psi_y, v=-psi_x guarantees u_x+v_y=0 for ANY smooth psi, by
    construction -- checked here for a real (randomly initialized) network,
    not just a hand-picked toy."""
    torch.manual_seed(0)
    model = PINN(in_dim=3, out_dim=2, hidden_dim=16, n_layers=3)
    x = torch.rand(10, 1, requires_grad=True)
    y = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)

    coords = torch.cat([x, y, t], dim=-1)
    psi = model(coords)[:, 0:1]
    u = _grad(psi, y)
    v = -_grad(psi, x)
    continuity = _grad(u, x) + _grad(v, y)

    assert torch.allclose(continuity, torch.zeros_like(continuity), atol=1e-4)


class _NavierStokesToy(nn.Module):
    """(psi, p) = (x*y, 2*x + 3*y) -- known derivatives (stream-function
    formulation), used with linear forcing fields (exactly reproduced by
    bilinear interpolation) to verify navier_stokes_residual independent of
    training."""

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        psi = x * y
        p = 2 * x + 3 * y
        return torch.cat([psi, p], dim=-1)


def test_navier_stokes_residual_zero_for_manufactured_solution():
    """With psi=x*y, p=2x+3y (=> u=x, v=-y, p_x=2, p_y=3), the momentum
    residuals reduce to (x + 2 - f_x, y + 3 - f_y); forcing f_x=x+2, f_y=y+3
    (both affine, so bilinear interpolation reproduces them exactly) should
    drive both residuals to (near) zero everywhere."""
    model = _NavierStokesToy()
    h, w = 8, 8
    xs = torch.linspace(0.0, 1.0, h)
    ys = torch.linspace(0.0, 1.0, w)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    force_x_field = xx + 2.0
    force_y_field = yy + 3.0
    domain = (0.0, 1.0, 0.0, 1.0)

    x = torch.linspace(0.1, 0.9, 5).unsqueeze(-1).requires_grad_(True)
    y = torch.linspace(0.1, 0.9, 5).unsqueeze(-1).requires_grad_(True)
    t = torch.zeros_like(x).requires_grad_(True)

    res_x, res_y = navier_stokes_residual(
        model,
        x,
        y,
        t,
        nu=0.1,
        force_x_field=force_x_field,
        force_y_field=force_y_field,
        domain=domain,
    )

    assert torch.allclose(res_x, torch.zeros_like(res_x), atol=1e-3)
    assert torch.allclose(res_y, torch.zeros_like(res_y), atol=1e-3)


def test_navier_stokes_pinn_loss_gradients_flow():
    model = PINN(in_dim=3, out_dim=2, hidden_dim=16, n_layers=3)
    force_field = torch.rand(16, 16)
    domain = (0.0, 1.0, 0.0, 1.0)
    x_c = torch.rand(20, 1, requires_grad=True)
    y_c = torch.rand(20, 1, requires_grad=True)
    t_c = torch.rand(20, 1, requires_grad=True)
    x_ic = torch.rand(10, 1, requires_grad=True)
    y_ic = torch.rand(10, 1, requires_grad=True)
    u_ic = torch.rand(10, 1)
    v_ic = torch.rand(10, 1)

    loss, residual_loss, ic_loss = navier_stokes_pinn_loss(
        model,
        x_c,
        y_c,
        t_c,
        0.01,
        force_field,
        force_field,
        domain,
        x_ic,
        y_ic,
        u_ic,
        v_ic,
    )
    loss.backward()

    grad = next(model.parameters()).grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert math.isfinite(residual_loss.item())
    assert math.isfinite(ic_loss.item())


def test_train_pinn_navier_stokes_reduces_loss():
    torch.manual_seed(0)
    force_field = torch.zeros(16, 16)  # no forcing -- simplest case
    x_ic = torch.linspace(0.0, 1.0, 20).unsqueeze(-1)
    y_ic = torch.linspace(0.0, 1.0, 20).unsqueeze(-1)
    u_ic = torch.sin(torch.pi * x_ic)
    v_ic = torch.zeros_like(y_ic)
    model = PINN(in_dim=3, out_dim=2, hidden_dim=20, n_layers=4)

    history = train_pinn_navier_stokes(
        model,
        force_field,
        force_field,
        domain=(0.0, 1.0, 0.0, 1.0),
        nu=0.01,
        x_ic=x_ic,
        y_ic=y_ic,
        u_ic=u_ic,
        v_ic=v_ic,
        domain_t=(0.0, 1.0),
        n_collocation=100,
        epochs=150,
        lr=1e-3,
        device="cpu",
    )

    early = sum(history[:15]) / 15
    late = sum(history[-15:]) / 15
    assert late < early


@pytest.mark.skipif(not NS_FILE.exists(), reason="NS_incom shard not downloaded")
def test_train_pinn_on_real_navier_stokes_ic():
    # spatial_stride=32 (512 -> 16) keeps this fast for a mechanics-only check.
    ds = NavierStokes2DDataset(
        NS_FILE, initial_step=1, split="train", spatial_stride=32
    )
    force_x_field = ds.force[0, 0]  # (16, 16)
    force_y_field = ds.force[0, 1]
    grid = ds.grid  # (16, 16, 2)
    x_ic = grid[..., 0].reshape(-1, 1)
    y_ic = grid[..., 1].reshape(-1, 1)
    u_ic = ds.velocity[0, 0, 0].reshape(-1, 1)  # t=0 slice, x-velocity
    v_ic = ds.velocity[0, 0, 1].reshape(-1, 1)  # t=0 slice, y-velocity

    model = PINN(in_dim=3, out_dim=2, hidden_dim=20, n_layers=4)
    history = train_pinn_navier_stokes(
        model,
        force_x_field,
        force_y_field,
        domain=(
            grid[..., 0].min().item(),
            grid[..., 0].max().item(),
            grid[..., 1].min().item(),
            grid[..., 1].max().item(),
        ),
        nu=0.01,  # assumed placeholder -- not verified against PDEBench's actual generation params
        x_ic=x_ic,
        y_ic=y_ic,
        u_ic=u_ic,
        v_ic=v_ic,
        domain_t=(ds.t.min().item(), ds.t.max().item()),
        n_collocation=100,
        epochs=10,
        lr=1e-3,
        device="cpu",
    )

    assert len(history) == 10
    assert all(math.isfinite(v) for v in history)
