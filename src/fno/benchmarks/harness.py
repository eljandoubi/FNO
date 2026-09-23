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
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from fno.data.pdebench import DarcyFlowDataset
from fno.models.deeponet import DeepONet
from fno.models.fno import FNO2d
from fno.models.pinn import PINN
from fno.training import trainer
from fno.training.collate import darcy_collate, darcy_deeponet_collate
from fno.training.pinn import train_pinn_darcy


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
    device: str = "cpu",
) -> BenchmarkResult:
    """Fit a PINN directly to ONE Darcy instance and compare against its
    ground truth. Not an operator -- see module docstring.
    """
    device_ = torch.device(device)
    model = PINN(in_dim=2, out_dim=1, hidden_dim=64, n_layers=6)

    start = time.perf_counter()
    train_pinn_darcy(
        model, coefficient_field, domain, forcing=forcing, epochs=epochs, device=device
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


def run_darcy_benchmark(
    file_path: str | Path,
    n_train: int = 512,
    n_test: int = 64,
    epochs: int = 20,
    pinn_epochs: int = 500,
    device: str | None = None,
) -> list[BenchmarkResult]:
    device = device or trainer.auto_device()

    full_train = DarcyFlowDataset(file_path, split="train")
    full_test = DarcyFlowDataset(file_path, split="test")

    low_res_train = Subset(_DownsampledDarcyView(full_train, stride=2), range(n_train))
    low_res_test = Subset(_DownsampledDarcyView(full_test, stride=2), range(n_test))
    full_res_test = Subset(
        full_test, range(n_test)
    )  # native 128x128, for the superres check

    config = trainer.TrainConfig(epochs=epochs, device=device, use_wandb=False)

    results = []

    fno_train_loader = DataLoader(
        low_res_train, batch_size=32, collate_fn=darcy_collate, shuffle=True
    )
    fno_test_loader = DataLoader(low_res_test, batch_size=32, collate_fn=darcy_collate)
    fno_superres_loader = DataLoader(
        full_res_test, batch_size=32, collate_fn=darcy_collate
    )
    fno_model = FNO2d(
        modes1=12, modes2=12, width=32, in_channels=3, out_channels=1, n_layers=4
    )
    results.append(
        benchmark_operator(
            "FNO2d",
            fno_model,
            fno_train_loader,
            fno_test_loader,
            config,
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
    field0, _target0, _low_res_grid = low_res_train[0]
    deeponet_model = DeepONet(
        branch_input_dim=field0.numel(),
        trunk_input_dim=2,
        hidden_dim=128,
        latent_dim=64,
        out_channels=1,
    )
    results.append(
        benchmark_operator(
            "DeepONet",
            deeponet_model,
            deeponet_train_loader,
            deeponet_test_loader,
            config,
            deeponet_superres_loader,
        )
    )

    pinn_field, pinn_target, pinn_grid = full_test[
        0
    ]  # same held-out instance as the superres check
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
        description="Benchmark FNO vs DeepONet vs PINN on Darcy Flow."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/raw/pdebench-darcy2d/2D_DarcyFlow_beta1.0_Train.hdf5"),
        help="Path to a downloaded Darcy HDF5 file (see: uv run fno-download pdebench-darcy2d).",
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
    args = parser.parse_args(argv)

    if not args.file.exists():
        raise SystemExit(
            f"{args.file} not found -- run: uv run fno-download pdebench-darcy2d"
        )

    results = run_darcy_benchmark(
        args.file,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
        pinn_epochs=args.pinn_epochs,
        device=args.device,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
