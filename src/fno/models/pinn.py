"""Physics-Informed Neural Network (PINN): a coordinate-to-solution MLP.

Reference: Raissi et al., "Physics-informed neural networks: A deep learning
framework for solving forward and inverse problems involving nonlinear
partial differential equations" (https://arxiv.org/abs/1711.10561).

Unlike FNO/DeepONet, a PINN is NOT an operator: it approximates the solution
of ONE specific PDE instance (fixed parameters/initial condition), trained by
minimizing the PDE residual (via autograd) plus a data-fit term on the
initial/boundary conditions -- it must be retrained for every new instance.
"""

from __future__ import annotations

from torch import Tensor, nn


class PINN(nn.Module):
    """Coordinate -> solution MLP with tanh activations (smooth, so it has
    well-behaved higher-order derivatives for the physics residual)."""

    def __init__(
        self, in_dim: int, out_dim: int = 1, hidden_dim: int = 64, n_layers: int = 6
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(max(n_layers - 2, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: Tensor) -> Tensor:
        return self.net(coords)
