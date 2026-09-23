"""Deep Operator Network (DeepONet).

Reference: Lu et al., "Learning nonlinear operators via DeepONet based on the
universal approximation theorem of operators" (https://arxiv.org/abs/1910.03193).

Branch net encodes the input function sampled at a fixed set of sensor points;
trunk net encodes the query coordinate(s) where the output is evaluated. The
output is their dot product (plus a bias), summed over a latent dimension.
Unlike FNO, the trunk can be queried at arbitrary/new coordinates without
retraining (output resolution-independent), but the branch net's input size is
fixed at construction time (sensor locations for the input function are not
resolution-independent).
"""

from __future__ import annotations

import torch
from torch import nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(max(n_layers - 2, 0)):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """Maps an input function (sampled at fixed sensors) to an output function
    evaluated at arbitrary query points.

    Forward: (branch_input, trunk_input) -> output.
    - branch_input: (B, branch_input_dim) -- flattened input function samples.
    - trunk_input: (N, trunk_input_dim) -- N query coordinates, shared across
      the batch.
    - output: (B, N, out_channels).
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        out_channels: int = 1,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.branch = _mlp(
            branch_input_dim, hidden_dim, latent_dim * out_channels, n_layers
        )
        self.trunk = _mlp(
            trunk_input_dim, hidden_dim, latent_dim * out_channels, n_layers
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self, branch_input: torch.Tensor, trunk_input: torch.Tensor
    ) -> torch.Tensor:
        branch_out = self.branch(branch_input)  # (B, latent_dim * C)
        trunk_out = self.trunk(trunk_input)  # (N, latent_dim * C)

        batch_size = branch_out.shape[0]
        n_points = trunk_out.shape[0]
        branch_out = branch_out.view(batch_size, self.out_channels, self.latent_dim)
        trunk_out = trunk_out.view(n_points, self.out_channels, self.latent_dim)

        out = torch.einsum("bcp,ncp->bnc", branch_out, trunk_out)
        return out + self.bias
