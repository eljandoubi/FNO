"""PyTorch datasets for PDEBench HDF5 files (Burgers 1D, Darcy Flow 2D, NS_Incom 2D).

Schema reference: pdebench/models/fno/utils.py in github.com/pdebench/PDEBench.
All three dataset classes below are confirmed against real downloaded files via
`fno-inspect-h5` + an end-to-end load (not just synthetic test fixtures).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _require_keys(h5_keys: set[str], required: set[str], file_path: Path) -> None:
    missing = required - h5_keys
    if missing:
        raise KeyError(
            f"{file_path} is missing expected key(s) {sorted(missing)}; "
            f"found: {sorted(h5_keys)}"
        )


def _split_slice(
    n_samples: int,
    split: str,
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> slice:
    """Index slice for 'train'/'val'/'test' out of `n_samples`, given
    (train_frac, val_frac, test_frac). `test` gets whatever remains after
    train+val, so the three fractions don't need to sum to exactly 1.
    """
    train_frac, val_frac, _test_frac = split_fractions
    if train_frac + val_frac > 1.0 + 1e-9:
        raise ValueError(
            f"train_frac + val_frac must be <= 1.0, got {train_frac + val_frac}"
        )
    train_end = round(n_samples * train_frac)
    val_end = train_end + round(n_samples * val_frac)
    if split == "train":
        return slice(0, train_end)
    if split == "val":
        return slice(train_end, val_end)
    if split == "test":
        return slice(val_end, n_samples)
    raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")


class DarcyFlowDataset(Dataset):
    """2D Darcy Flow: permeability field -> steady-state pressure field.

    Expects a PDEBench-style HDF5 file with "nu" (input coefficient field),
    "tensor" (target pressure field), and "x-coordinate"/"y-coordinate" keys.
    """

    def __init__(
        self,
        file_path: str | Path,
        split: str = "train",
        split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> None:
        self.file_path = Path(file_path)
        with h5py.File(self.file_path, "r") as f:
            _require_keys(
                set(f.keys()),
                {"nu", "tensor", "x-coordinate", "y-coordinate"},
                self.file_path,
            )
            nu = np.asarray(f["nu"], dtype=np.float32)
            tensor = np.asarray(f["tensor"], dtype=np.float32)
            x = np.asarray(f["x-coordinate"], dtype=np.float32)
            y = np.asarray(f["y-coordinate"], dtype=np.float32)

        # Some PDEBench dumps keep a singleton time axis on these fields; drop it.
        if nu.ndim == 4:
            nu = nu[:, 0]
        if tensor.ndim == 4:
            tensor = tensor[:, 0]

        n_samples = nu.shape[0]
        sel = _split_slice(n_samples, split, split_fractions)

        self.input = torch.from_numpy(nu[sel]).unsqueeze(1)  # (N, 1, nx, ny)
        self.target = torch.from_numpy(tensor[sel]).unsqueeze(1)  # (N, 1, nx, ny)
        grid_x, grid_y = torch.meshgrid(
            torch.from_numpy(x), torch.from_numpy(y), indexing="ij"
        )
        self.grid = torch.stack((grid_x, grid_y), dim=-1)  # (nx, ny, 2)

    def __len__(self) -> int:
        return self.input.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input[idx], self.target[idx], self.grid


class Burgers1DDataset(Dataset):
    """1D Burgers' equation: initial timesteps -> full spatio-temporal solution.

    Expects a PDEBench-style HDF5 file with "tensor" (N, T, X) and
    "x-coordinate" keys (each file is a single fixed viscosity).
    """

    def __init__(
        self,
        file_path: str | Path,
        initial_step: int = 10,
        split: str = "train",
        split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> None:
        self.file_path = Path(file_path)
        self.initial_step = initial_step
        with h5py.File(self.file_path, "r") as f:
            _require_keys(set(f.keys()), {"tensor", "x-coordinate"}, self.file_path)
            tensor = np.asarray(f["tensor"], dtype=np.float32)  # (N, T, X)
            x = np.asarray(f["x-coordinate"], dtype=np.float32)

        n_samples = tensor.shape[0]
        sel = _split_slice(n_samples, split, split_fractions)

        self.data = torch.from_numpy(tensor[sel])  # (N, T, X)
        self.grid = torch.from_numpy(x).unsqueeze(-1)  # (X, 1)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.data[idx]  # (T, X)
        return sample[: self.initial_step], sample, self.grid


class NavierStokes2DDataset(Dataset):
    """2D incompressible Navier-Stokes: initial velocity frames -> full trajectory.

    Expects one or more PDEBench NS_Incom-style HDF5 files ('velocity' (N, T, H, W, 2),
    'particles' (N, T, H, W, 1), 'force' (N, H, W, 2), 't' (N, T)). Each shard file
    only holds a handful of trajectories (N=4 as downloaded), so multiple shard
    paths can be passed and are concatenated along the sample axis. There is no
    x/y-coordinate key in the file; the grid is assumed to be the unit square.

    `spatial_stride` subsamples H/W at read time (e.g. 8 -> 512 becomes 64),
    which avoids loading the full-resolution arrays (~8GB+ per shard) into memory.

    Each shard only has 4 trajectories, so a train/val/test split on a single
    shard is extremely tight (may round to 0 for a split) -- pass several
    shard paths or adjust `split_fractions` accordingly.
    """

    def __init__(
        self,
        file_paths: str | Path | list[str | Path],
        initial_step: int = 10,
        split: str = "train",
        split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
        spatial_stride: int = 1,
    ) -> None:
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        self.file_paths = [Path(p) for p in file_paths]
        self.initial_step = initial_step

        velocities, particles_list, forces, times = [], [], [], []
        for path in self.file_paths:
            with h5py.File(path, "r") as f:
                _require_keys(
                    set(f.keys()), {"velocity", "particles", "force", "t"}, path
                )
                s = spatial_stride
                # Slicing the h5py Dataset (not a numpy array) subsamples at read
                # time, so a stride > 1 never materializes the full-resolution array.
                velocity_ds = cast("h5py.Dataset", f["velocity"])
                particles_ds = cast("h5py.Dataset", f["particles"])
                force_ds = cast("h5py.Dataset", f["force"])
                velocities.append(
                    np.asarray(velocity_ds[:, :, ::s, ::s, :], dtype=np.float32)
                )
                particles_list.append(
                    np.asarray(particles_ds[:, :, ::s, ::s, :], dtype=np.float32)
                )
                forces.append(np.asarray(force_ds[:, ::s, ::s, :], dtype=np.float32))
                times.append(np.asarray(f["t"], dtype=np.float32))

        velocity = np.concatenate(velocities, axis=0)  # (N, T, H, W, 2)
        particles = np.concatenate(particles_list, axis=0)  # (N, T, H, W, 1)
        force = np.concatenate(forces, axis=0)  # (N, H, W, 2)
        t = np.concatenate(times, axis=0)  # (N, T)

        # channel-first: (N, T, C, H, W) / (N, C, H, W)
        velocity = np.moveaxis(velocity, -1, 2)
        particles = np.moveaxis(particles, -1, 2)
        force = np.moveaxis(force, -1, 1)

        n_samples = velocity.shape[0]
        sel = _split_slice(n_samples, split, split_fractions)

        self.velocity = torch.from_numpy(velocity[sel])  # (N, T, 2, H, W)
        self.particles = torch.from_numpy(particles[sel])  # (N, T, 1, H, W)
        self.force = torch.from_numpy(force[sel])  # (N, 2, H, W)
        self.t = torch.from_numpy(t[sel])  # (N, T)

        height, width = velocity.shape[-2], velocity.shape[-1]
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij"
        )
        self.grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)

    def __len__(self) -> int:
        return self.velocity.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory = self.velocity[idx]  # (T, 2, H, W)
        return trajectory[: self.initial_step], trajectory, self.grid
