"""PDE residual loss and training loop for PINNs (autograd-based).

Currently covers the viscous 1D Burgers' equation u_t + u*u_x - nu*u_xx = 0,
the standard closed-form PINN benchmark. Darcy Flow is deliberately not
covered here: its coefficient field is a discretized per-sample array, not a
closed-form function, so its residual (div(a * grad(u)) = f) would need a
differentiable interpolation of that field -- not yet implemented.

Note the fundamentally different training paradigm vs. fno.training.trainer:
a PINN fits ONE PDE instance (fixed nu, fixed initial condition) by minimizing
its own residual, not a data loss over many samples -- there is no DataLoader
of (input, target) pairs here.
"""

from __future__ import annotations

import torch
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
