"""PDE residual loss and training loop for PINNs (autograd-based).

Covers the viscous 1D Burgers' equation (u_t + u*u_x - nu*u_xx = 0) and 2D
Darcy Flow (-div(a*grad(u)) = f, a(x,y)=0 on the domain boundary), the two
closed-form-residual PDEs among this repo's three target equations. NS_Incom
is deliberately not covered: its residual needs the full incompressible
Navier-Stokes momentum + continuity equations, which requires a pressure
field we don't have in the data (the standard fix is to have the network
jointly predict pressure/stream-function, not implemented here).

Note the fundamentally different training paradigm vs. fno.training.trainer:
a PINN fits ONE PDE instance (fixed parameters, fixed initial/boundary data)
by minimizing its own residual, not a data loss over many samples -- there is
no DataLoader of (input, target) pairs here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _grad(output: Tensor, inputs: Tensor) -> Tensor:
    return torch.autograd.grad(
        output, inputs, grad_outputs=torch.ones_like(output), create_graph=True
    )[0]


def burgers_residual(model: nn.Module, x: Tensor, t: Tensor, nu: float) -> Tensor:
    """PDE residual u_t + u*u_x - nu*u_xx for the viscous Burgers' equation.

    `x`/`t` must be leaf tensors with `requires_grad=True` (collocation points).
    """
    coords = torch.cat([x, t], dim=-1)
    u = model(coords)
    u_x = _grad(u, x)
    u_t = _grad(u, t)
    u_xx = _grad(u_x, x)
    return u_t + u * u_x - nu * u_xx


def burgers_pinn_loss(
    model: nn.Module,
    x_collocation: Tensor,
    t_collocation: Tensor,
    nu: float,
    x_ic: Tensor,
    u_ic: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (total_loss, residual_loss, ic_loss)."""
    residual = burgers_residual(model, x_collocation, t_collocation, nu)
    residual_loss = (residual**2).mean()

    t_ic = torch.zeros_like(x_ic)
    u_pred_ic = model(torch.cat([x_ic, t_ic], dim=-1))
    ic_loss = nn.functional.mse_loss(u_pred_ic, u_ic)

    return residual_loss + ic_loss, residual_loss, ic_loss


def train_pinn_burgers(
    model: nn.Module,
    x_ic: Tensor,
    u_ic: Tensor,
    nu: float,
    domain_x: tuple[float, float],
    domain_t: tuple[float, float],
    n_collocation: int = 2000,
    epochs: int = 1000,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Train a PINN to solve one instance of Burgers' equation. Collocation
    points are resampled uniformly at random from the domain every epoch.
    Returns the per-epoch total loss history.
    """
    device_ = torch.device(device)
    model.to(device_)
    x_ic = x_ic.to(device_)
    u_ic = u_ic.to(device_)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_lo, x_hi = domain_x
    t_lo, t_hi = domain_t

    history: list[float] = []
    for _ in range(epochs):
        x_c = (
            torch.rand(n_collocation, 1, device=device_) * (x_hi - x_lo) + x_lo
        ).requires_grad_(True)
        t_c = (
            torch.rand(n_collocation, 1, device=device_) * (t_hi - t_lo) + t_lo
        ).requires_grad_(True)

        optimizer.zero_grad()
        loss, _residual_loss, _ic_loss = burgers_pinn_loss(
            model, x_c, t_c, nu, x_ic, u_ic
        )
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return history


def _interpolate_field(
    field_hw: Tensor, x: Tensor, y: Tensor, domain: tuple[float, float, float, float]
) -> Tensor:
    """Differentiably bilinear-interpolate a (H, W) field, where field[i, j]
    sits at (x=domain_x_grid[i], y=domain_y_grid[j]), at continuous query
    points (x, y) -- each (N, 1). `domain` is (x_min, x_max, y_min, y_max).
    """
    x_min, x_max, y_min, y_max = domain
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
    # grid_sample's grid[..., 0] indexes the input's LAST dim (our W/y axis),
    # grid[..., 1] indexes the input's SECOND-TO-LAST dim (our H/x axis).
    grid = torch.stack([y_norm, x_norm], dim=-1).view(1, -1, 1, 2)
    field = field_hw.view(1, 1, *field_hw.shape)
    sampled = F.grid_sample(
        field, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    return sampled.view(-1, 1)


def darcy_residual(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    coefficient_field: Tensor,
    domain: tuple[float, float, float, float],
    forcing: float = 1.0,
) -> Tensor:
    """PDE residual -div(a*grad(u)) - f = -(a*(u_xx+u_yy) + a_x*u_x + a_y*u_y) - f
    for 2D Darcy Flow. `coefficient_field` is the discretized (H, W) 'a' field
    for ONE sample; `x`/`y` must be leaf tensors with `requires_grad=True`.
    The standard FNO-paper convention (Li et al. 2020) uses constant forcing
    f=1 on the unit square with Dirichlet u=0 on the boundary.
    """
    coords = torch.cat([x, y], dim=-1)
    u = model(coords)
    u_x = _grad(u, x)
    u_y = _grad(u, y)
    u_xx = _grad(u_x, x)
    u_yy = _grad(u_y, y)

    a = _interpolate_field(coefficient_field, x, y, domain)
    a_x = _grad(a, x)
    a_y = _grad(a, y)

    return -(a * (u_xx + u_yy) + a_x * u_x + a_y * u_y) - forcing


def _sample_boundary(
    n: int, domain: tuple[float, float, float, float], device: torch.device
) -> tuple[Tensor, Tensor]:
    x_min, x_max, y_min, y_max = domain
    n_per_edge = max(n // 4, 1)

    def _edge(x_vals: Tensor, y_vals: Tensor) -> Tensor:
        return torch.stack([x_vals, y_vals], dim=-1)

    rand = torch.rand(n_per_edge, device=device)
    edges = [
        _edge(
            torch.full((n_per_edge,), x_min, device=device),
            rand * (y_max - y_min) + y_min,
        ),
        _edge(
            torch.full((n_per_edge,), x_max, device=device),
            rand * (y_max - y_min) + y_min,
        ),
        _edge(
            rand * (x_max - x_min) + x_min,
            torch.full((n_per_edge,), y_min, device=device),
        ),
        _edge(
            rand * (x_max - x_min) + x_min,
            torch.full((n_per_edge,), y_max, device=device),
        ),
    ]
    points = torch.cat(edges, dim=0)
    return points[:, 0:1], points[:, 1:2]


def darcy_pinn_loss(
    model: nn.Module,
    x_collocation: Tensor,
    y_collocation: Tensor,
    coefficient_field: Tensor,
    domain: tuple[float, float, float, float],
    x_bc: Tensor,
    y_bc: Tensor,
    forcing: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (total_loss, residual_loss, bc_loss). BC is homogeneous
    Dirichlet (u=0), matching the standard Darcy Flow benchmark convention.
    """
    residual = darcy_residual(
        model, x_collocation, y_collocation, coefficient_field, domain, forcing
    )
    residual_loss = (residual**2).mean()

    u_bc = model(torch.cat([x_bc, y_bc], dim=-1))
    bc_loss = (u_bc**2).mean()

    return residual_loss + bc_loss, residual_loss, bc_loss


def train_pinn_darcy(
    model: nn.Module,
    coefficient_field: Tensor,
    domain: tuple[float, float, float, float],
    forcing: float = 1.0,
    n_collocation: int = 2000,
    n_boundary: int = 200,
    epochs: int = 1000,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Train a PINN to solve one instance of Darcy Flow (one coefficient
    field). Collocation/boundary points are resampled every epoch. Returns
    the per-epoch total loss history.
    """
    device_ = torch.device(device)
    model.to(device_)
    coefficient_field = coefficient_field.to(device_)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_min, x_max, y_min, y_max = domain

    history: list[float] = []
    for _ in range(epochs):
        x_c = (
            torch.rand(n_collocation, 1, device=device_) * (x_max - x_min) + x_min
        ).requires_grad_(True)
        y_c = (
            torch.rand(n_collocation, 1, device=device_) * (y_max - y_min) + y_min
        ).requires_grad_(True)
        x_bc, y_bc = _sample_boundary(n_boundary, domain, device_)

        optimizer.zero_grad()
        loss, _residual_loss, _bc_loss = darcy_pinn_loss(
            model, x_c, y_c, coefficient_field, domain, x_bc, y_bc, forcing
        )
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return history
