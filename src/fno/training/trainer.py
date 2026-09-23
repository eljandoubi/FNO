"""Generic training loop for operator-learning models (FNO, and later
DeepONet/PINN), decoupled from any specific dataset or architecture."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def relative_l2_error(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Per-sample relative L2 error ||pred - target|| / ||target||, averaged over the batch."""
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    error_norm = torch.linalg.norm(pred - target, dim=1)
    target_norm = torch.linalg.norm(target, dim=1)
    return (error_norm / (target_norm + eps)).mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Callable = nn.functional.mse_loss,
) -> float:
    """One training epoch. Each batch is (*model_inputs, targets) -- e.g. (input,
    targets) for FNO, or (branch_input, trunk_input, targets) for DeepONet.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        *inputs, targets = (t.to(device) for t in batch)
        optimizer.zero_grad()
        preds = model(*inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_error = 0.0
    n_batches = 0
    for batch in dataloader:
        *inputs, targets = (t.to(device) for t in batch)
        preds = model(*inputs)
        total_error += relative_l2_error(preds, targets).item()
        n_batches += 1
    return total_error / n_batches


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str | Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
    map_location: str | None = None,
) -> int:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    device: str = field(default_factory=auto_device)
    use_wandb: bool = False
    wandb_project: str = "fno-benchmark"
    checkpoint_path: str | None = None


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
) -> dict[str, list[float]]:
    """Run a full training loop, optionally logging to W&B. Returns per-epoch history."""
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    run = (
        wandb.init(project=config.wandb_project, config=asdict(config))
        if config.use_wandb
        else None
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_l2_error": []}
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_error = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_l2_error"].append(val_error)
        if run is not None:
            run.log(
                {"epoch": epoch, "train_loss": train_loss, "val_l2_error": val_error}
            )

    if run is not None:
        run.finish()

    if config.checkpoint_path:
        save_checkpoint(model, optimizer, config.epochs, config.checkpoint_path)

    return history
