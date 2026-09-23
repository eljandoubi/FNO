"""PyTorch datasets for PDEBench HDF5 files (Burgers 1D, Darcy Flow 2D).

Schema reference: pdebench/models/fno/utils.py in github.com/pdebench/PDEBench.
Not yet validated against a real downloaded file -- run against one before
trusting shapes blindly (NS_Incom shards are not covered here; their internal
layout differs and hasn't been confirmed).
"""

from __future__ import annotations

from pathlib import Path

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


class DarcyFlowDataset(Dataset):
    """2D Darcy Flow: permeability field -> steady-state pressure field.

    Expects a PDEBench-style HDF5 file with "nu" (input coefficient field),
    "tensor" (target pressure field), and "x-coordinate"/"y-coordinate" keys.
    """

    def __init__(
        self,
        file_path: str | Path,
        train: bool = True,
        train_split: float = 0.9,
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
        split = int(n_samples * train_split)
        sel = slice(0, split) if train else slice(split, n_samples)

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
        train: bool = True,
        train_split: float = 0.9,
    ) -> None:
        self.file_path = Path(file_path)
        self.initial_step = initial_step
        with h5py.File(self.file_path, "r") as f:
            _require_keys(set(f.keys()), {"tensor", "x-coordinate"}, self.file_path)
            tensor = np.asarray(f["tensor"], dtype=np.float32)  # (N, T, X)
            x = np.asarray(f["x-coordinate"], dtype=np.float32)

        n_samples = tensor.shape[0]
        split = int(n_samples * train_split)
        sel = slice(0, split) if train else slice(split, n_samples)

        self.data = torch.from_numpy(tensor[sel])  # (N, T, X)
        self.grid = torch.from_numpy(x).unsqueeze(-1)  # (X, 1)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.data[idx]  # (T, X)
        return sample[: self.initial_step], sample, self.grid
