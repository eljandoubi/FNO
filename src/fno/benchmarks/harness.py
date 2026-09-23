"""Benchmark harness comparing FNO, DeepONet, and PINN on the same data.

Includes a genuine (not approximated) zero-shot super-resolution test for the
operator models: they are trained on 64x64-downsampled Darcy fields and
evaluated against the REAL 128x128 ground truth PDEBench already provides --
no interpolated/synthetic "high-res" data is needed since Darcy's native
resolution already gives us that for free.

PINN is fundamentally not comparable to FNO/DeepONet: it fits ONE instance
(no train/test split, no operator generalization) rather than learning from
many samples. Its result is reported alongside for context, not as an
apples-to-apples row, and it is fit directly on the same held-out instance
the operators are evaluated on for the super-resolution check.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from fno.data.pdebench import Burgers1DDataset, DarcyFlowDataset, NavierStokes2DDataset
from fno.models.deeponet import DeepONet
from fno.models.fno import FNO1d, FNO2d
from fno.models.pinn import PINN
from fno.training import trainer
from fno.training.collate import (
    burgers_collate,
    burgers_deeponet_collate,
    darcy_collate,
    darcy_deeponet_collate,
    navier_stokes_collate,
    navier_stokes_deeponet_collate,
)
from fno.training.pinn import (
    train_pinn_burgers,
    train_pinn_darcy,
    train_pinn_navier_stokes,
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


@dataclass
class BenchmarkResult:
    name: str
    n_parameters: int
    train_time_s: float
    inference_latency_ms: float
    test_l2_error: float
    superres_l2_error: float | None = None
    note: str = ""


@dataclass
class BenchmarkHyperparams:
    """Optimizer/architecture overrides for `run_*_benchmark`. Fields left as
    `None` fall back to that function's own per-equation default (tuned for a
    fast demo run, not necessarily good accuracy -- see README for real-world
    values based on PDEBench's own published FNO baseline configs).
    """

    lr: float = 1e-3
    scheduler_step: int | None = None  # StepLR: decay every N epochs if set
    scheduler_gamma: float = 0.5
    fno_modes: int | None = None
    fno_width: int | None = None
    fno_n_layers: int | None = None
    deeponet_hidden_dim: int | None = None
    deeponet_latent_dim: int | None = None


def save_results(results: list[BenchmarkResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def load_results(path: str | Path) -> list[BenchmarkResult]:
    with open(path) as f:
        data = json.load(f)
    return [BenchmarkResult(**d) for d in data]


class _DownsampledDarcyView(Dataset):
    """Wraps a DarcyFlowDataset, strided-downsampling field/target/grid (e.g.
    128x128 -> 64x64) -- used to build the "low-res training data" side of the
    zero-shot super-resolution test."""

    def __init__(self, base: Dataset, stride: int) -> None:
        self.base = base
        self.stride = stride

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        field, target, grid = self.base[idx]
        s = self.stride
        return field[:, ::s, ::s], target[:, ::s, ::s], grid[::s, ::s]


class _DeepONetSuperresView(Dataset):
    """Pairs a LOW-RES coefficient field (downsampled to match the branch
    net's fixed sensor count) with the HIGH-RES target/grid. This is what
    genuine zero-shot super-resolution means for DeepONet: only the trunk's
    query resolution is flexible, not the branch's input sampling -- unlike
    FNO, which is resolution-independent on both ends.
    """

    def __init__(self, base: Dataset, stride: int) -> None:
        self.base = base
        self.stride = stride

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        field, target, grid = self.base[idx]
        low_res_field = field[:, :: self.stride, :: self.stride]
        return (
            low_res_field,
            target,
            grid,
        )  # target/grid stay at native (high) resolution


def _deeponet_superres_collate(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    fields, targets, grids = zip(*batch, strict=True)
    fields = torch.stack(
        fields
    )  # (B, 1, h_low, w_low) -- matches the branch's fixed input size
    branch_input = fields.reshape(fields.shape[0], -1)
    trunk_input = grids[0].reshape(-1, 2)  # high-res query grid
    targets = torch.stack(targets).permute(0, 2, 3, 1).reshape(fields.shape[0], -1, 1)
    return branch_input, trunk_input, targets


def _inference_latency_ms(
    model: nn.Module, loader: DataLoader, device: torch.device, n_repeats: int = 10
) -> float:
    model.eval()
    batch = next(iter(loader))
    *inputs, _targets = (t.to(device) for t in batch)
    with torch.no_grad():
        model(*inputs)  # warmup
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(n_repeats):
            model(*inputs)
        _synchronize(device)
    return (time.perf_counter() - start) / n_repeats * 1000


def benchmark_operator(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: trainer.TrainConfig,
    superres_loader: DataLoader | None = None,
) -> BenchmarkResult:
    """Train `model` via `trainer.fit`, then measure parameter count,
    inference latency, held-out test error, and (if `superres_loader` is
    given) error at a different resolution than it was trained on.
    """
    device = torch.device(config.device)

    start = time.perf_counter()
    trainer.fit(model, train_loader, test_loader, config)
    train_time_s = time.perf_counter() - start

    test_l2_error = trainer.evaluate(model, test_loader, device)
    inference_latency_ms = _inference_latency_ms(model, test_loader, device)

    superres_l2_error = None
    if superres_loader is not None:
        superres_l2_error = trainer.evaluate(model, superres_loader, device)

    return BenchmarkResult(
        name=name,
        n_parameters=count_parameters(model),
        train_time_s=train_time_s,
        inference_latency_ms=inference_latency_ms,
        test_l2_error=test_l2_error,
        superres_l2_error=superres_l2_error,
    )


def benchmark_pinn_darcy(
    name: str,
    coefficient_field: Tensor,
    target_field: Tensor,
    domain: tuple[float, float, float, float],
    forcing: float = 1.0,
    epochs: int = 500,
    lr: float = 1e-3,
    device: str = "cpu",
) -> BenchmarkResult:
    """Fit a PINN directly to ONE Darcy instance and compare against its
    ground truth. Not an operator -- see module docstring.
    """
    device_ = torch.device(device)
    model = PINN(in_dim=2, out_dim=1, hidden_dim=64, n_layers=6)

    start = time.perf_counter()
    train_pinn_darcy(
        model,
        coefficient_field,
        domain,
        forcing=forcing,
        epochs=epochs,
        lr=lr,
        device=device,
    )
    train_time_s = time.perf_counter() - start

    model.to(device_)
    height, width = target_field.shape[-2:]
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(domain[0], domain[1], height),
        torch.linspace(domain[2], domain[3], width),
        indexing="ij",
    )
    query = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).to(device_)

    model.eval()
    with torch.no_grad():
        _synchronize(device_)
        start = time.perf_counter()
        pred = model(query)
        _synchronize(device_)
        inference_latency_ms = (time.perf_counter() - start) * 1000

    pred = pred.reshape(1, 1, height, width)
    target = target_field.reshape(1, 1, height, width).to(device_)
    test_l2_error = trainer.relative_l2_error(pred, target).item()

    return BenchmarkResult(
        name=name,
        n_parameters=count_parameters(model),
        train_time_s=train_time_s,
        inference_latency_ms=inference_latency_ms,
        test_l2_error=test_l2_error,
        superres_l2_error=None,
        note="fit to ONE instance, not an operator -- not directly comparable to the rows above",
    )


def _checkpoint_path(checkpoint_dir: str | Path | None, name: str) -> str | None:
    if checkpoint_dir is None:
        return None
    return str(Path(checkpoint_dir) / f"{name}.pt")


def _train_config(
    hp: BenchmarkHyperparams, epochs: int, device: str, checkpoint_path: str | None
) -> trainer.TrainConfig:
    return trainer.TrainConfig(
        epochs=epochs,
        lr=hp.lr,
        scheduler_step=hp.scheduler_step,
        scheduler_gamma=hp.scheduler_gamma,
        device=device,
        use_wandb=False,
        checkpoint_path=checkpoint_path,
    )


def run_darcy_benchmark(
    file_path: str | Path,
    n_train: int = 512,
    n_test: int = 64,
    epochs: int = 20,
    pinn_epochs: int = 500,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
    hyperparams: BenchmarkHyperparams | None = None,
    skip_pinn: bool = False,
) -> list[BenchmarkResult]:
    device = device or trainer.auto_device()
    hp = hyperparams or BenchmarkHyperparams()

    full_train = DarcyFlowDataset(file_path, split="train")
    full_test = DarcyFlowDataset(file_path, split="test")

    low_res_train = Subset(_DownsampledDarcyView(full_train, stride=2), range(n_train))
    low_res_test = Subset(_DownsampledDarcyView(full_test, stride=2), range(n_test))
    full_res_test = Subset(
        full_test, range(n_test)
    )  # native 128x128, for the superres check

    results = []

    fno_train_loader = DataLoader(
        low_res_train, batch_size=32, collate_fn=darcy_collate, shuffle=True
    )
    fno_test_loader = DataLoader(low_res_test, batch_size=32, collate_fn=darcy_collate)
    fno_superres_loader = DataLoader(
        full_res_test, batch_size=32, collate_fn=darcy_collate
    )
    fno_model = FNO2d(
        modes1=hp.fno_modes or 12,
        modes2=hp.fno_modes or 12,
        width=hp.fno_width or 32,
        in_channels=3,
        out_channels=1,
        n_layers=hp.fno_n_layers or 4,
    )
    fno_config = _train_config(
        hp, epochs, device, _checkpoint_path(checkpoint_dir, "darcy_fno2d")
    )
    results.append(
        benchmark_operator(
            "FNO2d",
            fno_model,
            fno_train_loader,
            fno_test_loader,
            fno_config,
            fno_superres_loader,
        )
    )

    deeponet_train_loader = DataLoader(
        low_res_train, batch_size=32, collate_fn=darcy_deeponet_collate, shuffle=True
    )
    deeponet_test_loader = DataLoader(
        low_res_test, batch_size=32, collate_fn=darcy_deeponet_collate
    )
    # Branch input must stay at the trained (low-res) sensor count; only the
    # trunk's query grid goes to high-res -- see _DeepONetSuperresView.
    deeponet_superres_data = Subset(
        _DeepONetSuperresView(full_test, stride=2), range(n_test)
    )
    deeponet_superres_loader = DataLoader(
        deeponet_superres_data, batch_size=32, collate_fn=_deeponet_superres_collate
    )
    field0, _target0, _low_res_grid = low_res_train[0]  # type: ignore[misc]
    deeponet_model = DeepONet(
        branch_input_dim=field0.numel(),
        trunk_input_dim=2,
        hidden_dim=hp.deeponet_hidden_dim or 128,
        latent_dim=hp.deeponet_latent_dim or 64,
        out_channels=1,
    )
    deeponet_config = _train_config(
        hp, epochs, device, _checkpoint_path(checkpoint_dir, "darcy_deeponet")
    )
    results.append(
        benchmark_operator(
            "DeepONet",
            deeponet_model,
            deeponet_train_loader,
            deeponet_test_loader,
            deeponet_config,
            deeponet_superres_loader,
        )
    )

    pinn_field, pinn_target, pinn_grid = full_test[
        0
    ]  # same held-out instance as the superres check
    if not skip_pinn:
        results.append(
            benchmark_pinn_darcy(
                "PINN (single-instance fit)",
                pinn_field.squeeze(0),
                pinn_target.squeeze(0),
                domain=(
                    pinn_grid[..., 0].min().item(),
                    pinn_grid[..., 0].max().item(),
                    pinn_grid[..., 1].min().item(),
                    pinn_grid[..., 1].max().item(),
                ),
                epochs=pinn_epochs,
                lr=hp.lr,
                device=device,
            )
        )

    return results


class _DownsampledBurgersView(Dataset):
    """Wraps a Burgers1DDataset, strided-downsampling window/trajectory/grid
    along X -- the 1D analog of _DownsampledDarcyView."""

    def __init__(self, base: Dataset, stride: int) -> None:
        self.base = base
        self.stride = stride

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        window, trajectory, grid = self.base[idx]
        s = self.stride
        return window[:, ::s], trajectory[:, ::s], grid[::s]


class _DeepONetSuperres1DView(Dataset):
    """1D analog of _DeepONetSuperresView: keeps the branch's window at the
    trained (low-res) sensor count while trajectory/grid stay native-res."""

    def __init__(self, base: Dataset, stride: int) -> None:
        self.base = base
        self.stride = stride

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        window, trajectory, grid = self.base[idx]
        return window[:, :: self.stride], trajectory, grid


def _deeponet_superres_collate_1d(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    windows, trajectories, grids = zip(*batch, strict=True)
    windows = torch.stack(windows)  # (B, initial_step, X_low)
    branch_input = windows.reshape(windows.shape[0], -1)
    trunk_input = grids[0]  # (X_full, 1) -- native resolution
    trajectories = torch.stack(trajectories)  # (B, T, X_full)
    targets = trajectories[:, -1, :].unsqueeze(-1)
    return branch_input, trunk_input, targets


class _DeepONetNSSuperresView(Dataset):
    """Pairs a LOW-RES NS window (from `low_res_ds`, matching the branch's
    fixed sensor count) with the HIGH-RES target/grid (from `high_res_ds`).
    The two datasets must be constructed from the same file_paths/split so
    their sample ordering lines up index-for-index (spatial_stride doesn't
    affect which samples end up in a split, only their spatial content).
    """

    def __init__(self, low_res_ds: Dataset, high_res_ds: Dataset) -> None:
        self.low_res_ds = low_res_ds
        self.high_res_ds = high_res_ds

    def __len__(self) -> int:
        return len(self.low_res_ds)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        low_res_window, _low_res_traj, _low_res_grid = self.low_res_ds[idx]
        _high_res_window, high_res_traj, high_res_grid = self.high_res_ds[idx]
        return low_res_window, high_res_traj, high_res_grid


def benchmark_pinn_burgers(
    name: str,
    x_ic: Tensor,
    u_ic: Tensor,
    nu: float,
    domain_x: tuple[float, float],
    domain_t: tuple[float, float],
    target_final: Tensor,
    epochs: int = 2000,
    lr: float = 1e-3,
    device: str = "cpu",
) -> BenchmarkResult:
    """Fit a PINN to ONE Burgers instance (its initial condition), then
    compare its prediction at t=domain_t[1] against the real final-timestep
    ground truth. Not an operator -- see module docstring.
    """
    device_ = torch.device(device)
    model = PINN(in_dim=2, out_dim=1, hidden_dim=64, n_layers=6)

    start = time.perf_counter()
    train_pinn_burgers(
        model, x_ic, u_ic, nu, domain_x, domain_t, epochs=epochs, lr=lr, device=device
    )
    train_time_s = time.perf_counter() - start

    model.to(device_)
    t_final = torch.full_like(x_ic, domain_t[1]).to(device_)
    query = torch.cat([x_ic.to(device_), t_final], dim=-1)

    model.eval()
    with torch.no_grad():
        _synchronize(device_)
        start = time.perf_counter()
        pred = model(query)
        _synchronize(device_)
        inference_latency_ms = (time.perf_counter() - start) * 1000

    target = target_final.reshape(pred.shape).to(device_)
    test_l2_error = trainer.relative_l2_error(
        pred.unsqueeze(0), target.unsqueeze(0)
    ).item()

    return BenchmarkResult(
        name=name,
        n_parameters=count_parameters(model),
        train_time_s=train_time_s,
        inference_latency_ms=inference_latency_ms,
        test_l2_error=test_l2_error,
        superres_l2_error=None,
        note="fit to ONE instance, not an operator -- not directly comparable to the rows above",
    )


def run_burgers_benchmark(
    file_path: str | Path,
    n_train: int = 512,
    n_test: int = 64,
    epochs: int = 20,
    pinn_epochs: int = 2000,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
    hyperparams: BenchmarkHyperparams | None = None,
    skip_pinn: bool = False,
) -> list[BenchmarkResult]:
    device = device or trainer.auto_device()
    hp = hyperparams or BenchmarkHyperparams()

    full_train = Burgers1DDataset(file_path, initial_step=10, split="train")
    full_test = Burgers1DDataset(file_path, initial_step=10, split="test")

    low_res_train = Subset(
        _DownsampledBurgersView(full_train, stride=4), range(n_train)
    )
    low_res_test = Subset(_DownsampledBurgersView(full_test, stride=4), range(n_test))
    full_res_test = Subset(full_test, range(n_test))  # native 1024 points

    results = []

    fno_train_loader = DataLoader(
        low_res_train, batch_size=32, collate_fn=burgers_collate, shuffle=True
    )
    fno_test_loader = DataLoader(
        low_res_test, batch_size=32, collate_fn=burgers_collate
    )
    fno_superres_loader = DataLoader(
        full_res_test, batch_size=32, collate_fn=burgers_collate
    )
    fno_model = FNO1d(
        modes=hp.fno_modes or 16,
        width=hp.fno_width or 64,
        in_channels=11,
        out_channels=1,
        n_layers=hp.fno_n_layers or 4,
    )
    fno_config = _train_config(
        hp, epochs, device, _checkpoint_path(checkpoint_dir, "burgers_fno1d")
    )
    results.append(
        benchmark_operator(
            "FNO1d",
            fno_model,
            fno_train_loader,
            fno_test_loader,
            fno_config,
            fno_superres_loader,
        )
    )

    deeponet_train_loader = DataLoader(
        low_res_train, batch_size=32, collate_fn=burgers_deeponet_collate, shuffle=True
    )
    deeponet_test_loader = DataLoader(
        low_res_test, batch_size=32, collate_fn=burgers_deeponet_collate
    )
    deeponet_superres_data = Subset(
        _DeepONetSuperres1DView(full_test, stride=4), range(n_test)
    )
    deeponet_superres_loader = DataLoader(
        deeponet_superres_data, batch_size=32, collate_fn=_deeponet_superres_collate_1d
    )
    window0, _traj0, _grid0 = low_res_train[0]  # type: ignore[misc]
    deeponet_model = DeepONet(
        branch_input_dim=window0.numel(),
        trunk_input_dim=1,
        hidden_dim=hp.deeponet_hidden_dim or 128,
        latent_dim=hp.deeponet_latent_dim or 64,
        out_channels=1,
    )
    deeponet_config = _train_config(
        hp, epochs, device, _checkpoint_path(checkpoint_dir, "burgers_deeponet")
    )
    results.append(
        benchmark_operator(
            "DeepONet",
            deeponet_model,
            deeponet_train_loader,
            deeponet_test_loader,
            deeponet_config,
            deeponet_superres_loader,
        )
    )

    _pinn_window, pinn_trajectory, pinn_grid = full_test[0]
    if not skip_pinn:
        results.append(
            benchmark_pinn_burgers(
                "PINN (single-instance fit)",
                x_ic=pinn_grid,
                u_ic=pinn_trajectory[0].unsqueeze(-1),
                nu=0.01,  # matches PDEBench's Nu0.01 filename convention
                domain_x=(pinn_grid.min().item(), pinn_grid.max().item()),
                domain_t=(full_test.t.min().item(), full_test.t.max().item()),
                target_final=pinn_trajectory[-1],
                epochs=pinn_epochs,
                lr=hp.lr,
                device=device,
            )
        )

    return results


def benchmark_pinn_navier_stokes(
    name: str,
    force_x_field: Tensor,
    force_y_field: Tensor,
    domain: tuple[float, float, float, float],
    nu: float,
    x_ic: Tensor,
    y_ic: Tensor,
    u_ic: Tensor,
    v_ic: Tensor,
    domain_t: tuple[float, float],
    target_u_final: Tensor,
    target_v_final: Tensor,
    epochs: int = 500,
    lr: float = 1e-3,
    device: str = "cpu",
) -> BenchmarkResult:
    """Fit a PINN to ONE Navier-Stokes instance (its forcing + initial
    condition), then compare its prediction at t=domain_t[1] against the real
    final-timestep velocity field. Not an operator -- see module docstring.
    """
    device_ = torch.device(device)
    model = PINN(in_dim=3, out_dim=2, hidden_dim=64, n_layers=6)

    start = time.perf_counter()
    train_pinn_navier_stokes(
        model,
        force_x_field,
        force_y_field,
        domain,
        nu,
        x_ic,
        y_ic,
        u_ic,
        v_ic,
        domain_t,
        epochs=epochs,
        lr=lr,
        device=device,
    )
    train_time_s = time.perf_counter() - start

    model.to(device_)
    x_query = x_ic.to(device_).requires_grad_(True)
    y_query = y_ic.to(device_).requires_grad_(True)
    t_query = torch.full_like(x_query, domain_t[1])

    start = time.perf_counter()
    coords = torch.cat([x_query, y_query, t_query], dim=-1)
    psi_p = model(coords)
    psi = psi_p[:, 0:1]
    u_pred = torch.autograd.grad(
        psi, y_query, grad_outputs=torch.ones_like(psi), create_graph=False
    )[0]
    v_pred = -torch.autograd.grad(
        psi, x_query, grad_outputs=torch.ones_like(psi), create_graph=False
    )[0]
    _synchronize(device_)
    inference_latency_ms = (time.perf_counter() - start) * 1000

    u_target = target_u_final.reshape(u_pred.shape).to(device_)
    v_target = target_v_final.reshape(v_pred.shape).to(device_)
    pred = torch.cat([u_pred.detach(), v_pred.detach()], dim=-1).unsqueeze(0)
    target = torch.cat([u_target, v_target], dim=-1).unsqueeze(0)
    test_l2_error = trainer.relative_l2_error(pred, target).item()

    return BenchmarkResult(
        name=name,
        n_parameters=count_parameters(model),
        train_time_s=train_time_s,
        inference_latency_ms=inference_latency_ms,
        test_l2_error=test_l2_error,
        superres_l2_error=None,
        note="fit to ONE instance, not an operator -- not directly comparable to the rows above",
    )


def run_navier_stokes_benchmark(
    file_paths: str | Path | Sequence[str | Path],
    epochs: int = 10,
    pinn_epochs: int = 300,
    low_res_stride: int = 16,
    high_res_stride: int = 8,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
    hyperparams: BenchmarkHyperparams | None = None,
    skip_pinn: bool = False,
) -> list[BenchmarkResult]:
    """Only 4 trajectories per downloaded shard, so this uses a (0.5, 0.0, 0.5)
    train/test split by default rather than the usual (0.8, 0.1, 0.1) -- pass
    multiple shard paths for a more meaningful split.
    """
    device = device or trainer.auto_device()
    hp = hyperparams or BenchmarkHyperparams()
    fractions = (0.5, 0.0, 0.5)
    paths: str | Path | list[str | Path] = (
        file_paths if isinstance(file_paths, (str, Path)) else list(file_paths)
    )

    low_res_train = NavierStokes2DDataset(
        paths,
        initial_step=5,
        split="train",
        split_fractions=fractions,
        spatial_stride=low_res_stride,
    )
    low_res_test = NavierStokes2DDataset(
        paths,
        initial_step=5,
        split="test",
        split_fractions=fractions,
        spatial_stride=low_res_stride,
    )
    high_res_test = NavierStokes2DDataset(
        paths,
        initial_step=5,
        split="test",
        split_fractions=fractions,
        spatial_stride=high_res_stride,
    )

    results = []

    fno_train_loader = DataLoader(
        low_res_train, batch_size=1, collate_fn=navier_stokes_collate, shuffle=True
    )
    fno_test_loader = DataLoader(
        low_res_test, batch_size=1, collate_fn=navier_stokes_collate
    )
    fno_superres_loader = DataLoader(
        high_res_test, batch_size=1, collate_fn=navier_stokes_collate
    )
    in_channels = 5 * 2 + 2
    fno_model = FNO2d(
        modes1=hp.fno_modes or 8,
        modes2=hp.fno_modes or 8,
        width=hp.fno_width or 16,
        in_channels=in_channels,
        out_channels=2,
        n_layers=hp.fno_n_layers or 3,
    )
    fno_config = _train_config(
        hp, epochs, device, _checkpoint_path(checkpoint_dir, "navierstokes_fno2d")
    )
    results.append(
        benchmark_operator(
            "FNO2d",
            fno_model,
            fno_train_loader,
            fno_test_loader,
            fno_config,
            fno_superres_loader,
        )
    )

    deeponet_train_loader = DataLoader(
        low_res_train,
        batch_size=1,
        collate_fn=navier_stokes_deeponet_collate,
        shuffle=True,
    )
    deeponet_test_loader = DataLoader(
        low_res_test, batch_size=1, collate_fn=navier_stokes_deeponet_collate
    )
    deeponet_superres_data = _DeepONetNSSuperresView(low_res_test, high_res_test)
    deeponet_superres_loader = DataLoader(
        deeponet_superres_data, batch_size=1, collate_fn=navier_stokes_deeponet_collate
    )
    window0, _traj0, _grid0 = low_res_train[0]
    deeponet_model = DeepONet(
        branch_input_dim=window0.numel(),
        trunk_input_dim=2,
        hidden_dim=hp.deeponet_hidden_dim or 128,
        latent_dim=hp.deeponet_latent_dim or 64,
        out_channels=2,
    )
    deeponet_config = _train_config(
        hp, epochs, device, _checkpoint_path(checkpoint_dir, "navierstokes_deeponet")
    )
    results.append(
        benchmark_operator(
            "DeepONet",
            deeponet_model,
            deeponet_train_loader,
            deeponet_test_loader,
            deeponet_config,
            deeponet_superres_loader,
        )
    )

    grid = high_res_test.grid  # (H, W, 2)
    force = high_res_test.force[0]  # (2, H, W) -- first test trajectory
    velocity0 = high_res_test.velocity[0]  # (T, 2, H, W)
    if not skip_pinn:
        results.append(
            benchmark_pinn_navier_stokes(
                "PINN (single-instance fit)",
                force_x_field=force[0],
                force_y_field=force[1],
                domain=(
                    grid[..., 0].min().item(),
                    grid[..., 0].max().item(),
                    grid[..., 1].min().item(),
                    grid[..., 1].max().item(),
                ),
                nu=0.01,  # assumed placeholder -- not verified against PDEBench's actual generation params
                x_ic=grid[..., 0].reshape(-1, 1),
                y_ic=grid[..., 1].reshape(-1, 1),
                u_ic=velocity0[0, 0].reshape(-1, 1),
                v_ic=velocity0[0, 1].reshape(-1, 1),
                domain_t=(
                    high_res_test.t[0].min().item(),
                    high_res_test.t[0].max().item(),
                ),
                target_u_final=velocity0[-1, 0],
                target_v_final=velocity0[-1, 1],
                epochs=pinn_epochs,
                lr=hp.lr,
                device=device,
            )
        )

    return results


def format_summary(results: list[BenchmarkResult]) -> str:
    header = (
        f"{'Model':<28}{'Params':>10}{'Train (s)':>12}{'Infer (ms)':>12}"
        f"{'Test L2':>10}{'Superres L2':>14}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        superres = (
            f"{r.superres_l2_error:.4f}" if r.superres_l2_error is not None else "n/a"
        )
        lines.append(
            f"{r.name:<28}{r.n_parameters:>10,}{r.train_time_s:>12.2f}"
            f"{r.inference_latency_ms:>12.2f}{r.test_l2_error:>10.4f}{superres:>14}"
        )
        if r.note:
            lines.append(f"    note: {r.note}")
    return "\n".join(lines)


def print_summary(results: list[BenchmarkResult]) -> None:
    print(format_summary(results))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FNO vs DeepONet vs PINN on PDEBench equations."
    )
    parser.add_argument(
        "--equation",
        choices=["darcy", "burgers", "navier-stokes", "all"],
        default="all",
        help="Which equation(s) to benchmark (default: all).",
    )
    parser.add_argument(
        "--darcy-file",
        type=Path,
        default=Path("data/raw/pdebench-darcy2d/2D_DarcyFlow_beta1.0_Train.hdf5"),
        help="Path to a downloaded Darcy HDF5 file (see: uv run fno-download pdebench-darcy2d).",
    )
    parser.add_argument(
        "--burgers-file",
        type=Path,
        default=Path("data/raw/pdebench-burgers1d/1D_Burgers_Sols_Nu0.01.hdf5"),
        help="Path to a downloaded Burgers HDF5 file (see: uv run fno-download pdebench-burgers1d).",
    )
    parser.add_argument(
        "--navier-stokes-file",
        type=Path,
        action="append",
        default=None,
        help=(
            "Path to a downloaded Navier-Stokes shard (repeatable). Defaults to "
            "shard0 (see: uv run fno-download pdebench-navierstokes2d)."
        ),
    )
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument(
        "--epochs", type=int, default=20, help="Epochs for FNO/DeepONet."
    )
    parser.add_argument("--pinn-epochs", type=int, default=500)
    parser.add_argument(
        "--device", default=None, help="Defaults to auto-detected (cuda/mps/cpu)."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for per-model training checkpoints, enabling resume.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Save the combined results (all benchmarked equations) as JSON here.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for FNO/DeepONet/PINN."
    )
    parser.add_argument(
        "--scheduler-step",
        type=int,
        default=None,
        help="StepLR: decay lr every N epochs (unset: constant lr). PDEBench's own FNO baselines use 100.",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.5,
        help="StepLR decay factor, only used if --scheduler-step is set.",
    )
    parser.add_argument(
        "--fno-modes",
        type=int,
        default=None,
        help="Override FNO's Fourier mode count (per-equation default otherwise).",
    )
    parser.add_argument(
        "--fno-width",
        type=int,
        default=None,
        help="Override FNO's channel width (per-equation default otherwise).",
    )
    parser.add_argument(
        "--fno-n-layers",
        type=int,
        default=None,
        help="Override FNO's number of spectral layers (per-equation default otherwise).",
    )
    parser.add_argument(
        "--deeponet-hidden-dim",
        type=int,
        default=None,
        help="Override DeepONet's branch/trunk hidden width (default 128).",
    )
    parser.add_argument(
        "--deeponet-latent-dim",
        type=int,
        default=None,
        help="Override DeepONet's latent (dot-product) dimension (default 64).",
    )
    args = parser.parse_args(argv)

    hyperparams = BenchmarkHyperparams(
        lr=args.lr,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        fno_modes=args.fno_modes,
        fno_width=args.fno_width,
        fno_n_layers=args.fno_n_layers,
        deeponet_hidden_dim=args.deeponet_hidden_dim,
        deeponet_latent_dim=args.deeponet_latent_dim,
    )

    ns_files = args.navier_stokes_file or [
        Path("data/raw/pdebench-navierstokes2d/ns_incom_inhom_2d_512-0.h5")
    ]

    all_results: list[BenchmarkResult] = []

    if args.equation in ("darcy", "all"):
        if not args.darcy_file.exists():
            raise SystemExit(
                f"{args.darcy_file} not found -- run: uv run fno-download pdebench-darcy2d"
            )
        print("=== Darcy Flow ===")
        results = run_darcy_benchmark(
            args.darcy_file,
            n_train=args.n_train,
            n_test=args.n_test,
            epochs=args.epochs,
            pinn_epochs=args.pinn_epochs,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            hyperparams=hyperparams,
        )
        print_summary(results)
        all_results.extend(results)

    if args.equation in ("burgers", "all"):
        if not args.burgers_file.exists():
            raise SystemExit(
                f"{args.burgers_file} not found -- run: uv run fno-download pdebench-burgers1d"
            )
        print("\n=== Burgers' Equation ===")
        results = run_burgers_benchmark(
            args.burgers_file,
            n_train=args.n_train,
            n_test=args.n_test,
            epochs=args.epochs,
            pinn_epochs=args.pinn_epochs,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            hyperparams=hyperparams,
        )
        print_summary(results)
        all_results.extend(results)

    if args.equation in ("navier-stokes", "all"):
        missing = [p for p in ns_files if not p.exists()]
        if missing:
            raise SystemExit(
                f"{missing} not found -- run: uv run fno-download pdebench-navierstokes2d"
            )
        print("\n=== Navier-Stokes 2D ===")
        results = run_navier_stokes_benchmark(
            ns_files,
            epochs=args.epochs,
            pinn_epochs=args.pinn_epochs,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            hyperparams=hyperparams,
        )
        print_summary(results)
        all_results.extend(results)

    if args.results_file:
        save_results(all_results, args.results_file)
        print(f"\nSaved combined results to {args.results_file}")


if __name__ == "__main__":
    main()
