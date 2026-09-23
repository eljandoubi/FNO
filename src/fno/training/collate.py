"""Adapters that turn a PDEBench Dataset's (field, target, grid) triplets into
the tensors that fno.models.fno / fno.models.deeponet expect."""

from __future__ import annotations

import torch


def darcy_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate for DarcyFlowDataset -> FNO2d.

    Concatenates the permeability field with the (shared) grid coordinates
    along the channel axis, and moves both input and target to channel-last.
    """
    fields, targets, grids = zip(*batch, strict=True)
    fields = torch.stack(fields).permute(0, 2, 3, 1)  # (B, H, W, 1)
    targets = torch.stack(targets).permute(0, 2, 3, 1)  # (B, H, W, 1)
    grid = grids[0].unsqueeze(0).expand(fields.shape[0], -1, -1, -1)  # (B, H, W, 2)
    inputs = torch.cat([fields, grid], dim=-1)  # (B, H, W, 3)
    return inputs, targets


def burgers_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate for Burgers1DDataset -> FNO1d.

    Input: initial window + grid coordinate, channel-last (B, X, initial_step + 1).
    Target: the trajectory's final timestep, channel-last (B, X, 1).
    """
    windows, trajectories, grids = zip(*batch, strict=True)
    windows = torch.stack(windows).permute(0, 2, 1)  # (B, X, initial_step)
    grid = grids[0].unsqueeze(0).expand(windows.shape[0], -1, -1)  # (B, X, 1)
    inputs = torch.cat([windows, grid], dim=-1)  # (B, X, initial_step + 1)

    trajectories = torch.stack(trajectories)  # (B, T, X)
    targets = trajectories[:, -1, :].unsqueeze(-1)  # (B, X, 1) -- final timestep
    return inputs, targets


def navier_stokes_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate for NavierStokes2DDataset -> FNO2d.

    Input: initial velocity frames (time and velocity-component axes merged
    into channels) + grid, channel-last (B, H, W, initial_step * 2 + 2).
    Target: final-timestep velocity, channel-last (B, H, W, 2).
    """
    windows, trajectories, grids = zip(*batch, strict=True)

    windows = torch.stack(windows)  # (B, initial_step, 2, H, W)
    b, t, c, h, w = windows.shape
    windows = windows.reshape(b, t * c, h, w).permute(0, 2, 3, 1)  # (B, H, W, t*c)
    grid = grids[0].unsqueeze(0).expand(b, -1, -1, -1)  # (B, H, W, 2)
    inputs = torch.cat([windows, grid], dim=-1)  # (B, H, W, t*c + 2)

    trajectories = torch.stack(trajectories)  # (B, T, 2, H, W)
    targets = trajectories[:, -1].permute(0, 2, 3, 1)  # (B, H, W, 2) -- final timestep
    return inputs, targets


def darcy_deeponet_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate for DarcyFlowDataset -> DeepONet.

    Branch input: flattened permeability field, (B, H*W). Trunk input: shared
    grid coordinates, (H*W, 2). Target: pressure field at every grid point,
    (B, H*W, 1).
    """
    fields, targets, grids = zip(*batch, strict=True)
    fields = torch.stack(fields)  # (B, 1, H, W)
    branch_input = fields.reshape(fields.shape[0], -1)  # (B, H*W)
    trunk_input = grids[0].reshape(-1, 2)  # (H*W, 2)
    targets = torch.stack(targets).permute(0, 2, 3, 1).reshape(fields.shape[0], -1, 1)
    return branch_input, trunk_input, targets


def burgers_deeponet_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate for Burgers1DDataset -> DeepONet.

    Branch input: flattened initial window, (B, initial_step*X). Trunk input:
    shared grid coordinates, (X, 1). Target: final timestep, (B, X, 1).
    """
    windows, trajectories, grids = zip(*batch, strict=True)
    windows = torch.stack(windows)  # (B, initial_step, X)
    branch_input = windows.reshape(windows.shape[0], -1)  # (B, initial_step*X)
    trunk_input = grids[0]  # (X, 1)

    trajectories = torch.stack(trajectories)  # (B, T, X)
    targets = trajectories[:, -1, :].unsqueeze(-1)  # (B, X, 1) -- final timestep
    return branch_input, trunk_input, targets


def navier_stokes_deeponet_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate for NavierStokes2DDataset -> DeepONet.

    Branch input: flattened initial velocity frames, (B, initial_step*2*H*W).
    Trunk input: shared grid coordinates, (H*W, 2). Target: final-timestep
    velocity at every grid point, (B, H*W, 2).
    """
    windows, trajectories, grids = zip(*batch, strict=True)
    windows = torch.stack(windows)  # (B, initial_step, 2, H, W)
    branch_input = windows.reshape(windows.shape[0], -1)  # (B, initial_step*2*H*W)
    trunk_input = grids[0].reshape(-1, 2)  # (H*W, 2)

    trajectories = torch.stack(trajectories)  # (B, T, 2, H, W)
    targets = trajectories[:, -1].permute(0, 2, 3, 1).reshape(windows.shape[0], -1, 2)
    return branch_input, trunk_input, targets
