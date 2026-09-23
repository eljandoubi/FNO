"""Unit tests for fno.training.trainer / fno.training.collate.

W&B network calls are mocked -- these tests never hit the network. The final
test trains on a small real Darcy subset when available, as a mechanics check
(not a rigorous train/val split or an accuracy benchmark).
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from fno.data.pdebench import Burgers1DDataset, DarcyFlowDataset, NavierStokes2DDataset
from fno.models.deeponet import DeepONet
from fno.models.fno import FNO1d, FNO2d
from fno.training import trainer
from fno.training.collate import (
    burgers_collate,
    burgers_deeponet_collate,
    darcy_collate,
    darcy_deeponet_collate,
    navier_stokes_collate,
    navier_stokes_deeponet_collate,
)

DARCY_FILE = Path("data/raw/pdebench-darcy2d/2D_DarcyFlow_beta1.0_Train.hdf5")
BURGERS_FILE = Path("data/raw/pdebench-burgers1d/1D_Burgers_Sols_Nu0.01.hdf5")
NS_FILE = Path("data/raw/pdebench-navierstokes2d/ns_incom_inhom_2d_512-0.h5")


def test_relative_l2_error_zero_for_identical():
    x = torch.randn(4, 3, 8, 8)
    assert trainer.relative_l2_error(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_relative_l2_error_known_value():
    target = torch.tensor([[3.0, 4.0]])  # norm = 5
    pred = torch.zeros_like(target)  # error norm = 5 -> relative error = 1.0
    err = trainer.relative_l2_error(pred, target)
    assert err.item() == pytest.approx(1.0, abs=1e-6)


def test_darcy_collate_shapes():
    field = torch.randn(1, 8, 8)
    target = torch.randn(1, 8, 8)
    grid = torch.randn(8, 8, 2)
    batch = [(field, target, grid), (field, target, grid)]
    inputs, targets = darcy_collate(batch)
    assert inputs.shape == (2, 8, 8, 3)
    assert targets.shape == (2, 8, 8, 1)


def test_darcy_deeponet_collate_shapes():
    field = torch.randn(1, 8, 8)
    target = torch.randn(1, 8, 8)
    grid = torch.randn(8, 8, 2)
    batch = [(field, target, grid), (field, target, grid)]
    branch_input, trunk_input, targets = darcy_deeponet_collate(batch)
    assert branch_input.shape == (2, 64)
    assert trunk_input.shape == (64, 2)
    assert targets.shape == (2, 64, 1)


def test_burgers_collate_shapes():
    window = torch.randn(10, 32)
    trajectory = torch.randn(20, 32)
    grid = torch.randn(32, 1)
    batch = [(window, trajectory, grid), (window, trajectory, grid)]
    inputs, targets = burgers_collate(batch)
    assert inputs.shape == (2, 32, 11)
    assert targets.shape == (2, 32, 1)


def test_navier_stokes_collate_shapes():
    window = torch.randn(5, 2, 16, 16)
    trajectory = torch.randn(20, 2, 16, 16)
    grid = torch.randn(16, 16, 2)
    batch = [(window, trajectory, grid), (window, trajectory, grid)]
    inputs, targets = navier_stokes_collate(batch)
    assert inputs.shape == (2, 16, 16, 5 * 2 + 2)
    assert targets.shape == (2, 16, 16, 2)


def _make_synthetic_loader(n=16, h=16, w=16, c_in=3, c_out=1, batch_size=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, h, w, c_in, generator=g)
    y = x[..., :c_out] * 2.0  # simple learnable target
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_training_reduces_loss():
    torch.manual_seed(0)
    model = FNO2d(
        modes1=4, modes2=4, width=8, in_channels=3, out_channels=1, n_layers=2
    )
    loader = _make_synthetic_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    device = torch.device("cpu")

    first_loss = trainer.train_one_epoch(model, loader, optimizer, device)
    last_loss = first_loss
    for _ in range(20):
        last_loss = trainer.train_one_epoch(model, loader, optimizer, device)

    assert last_loss < first_loss


def test_checkpoint_roundtrip(tmp_path):
    model = FNO2d(
        modes1=4, modes2=4, width=8, in_channels=3, out_channels=1, n_layers=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    history = {"train_loss": [0.5, 0.3], "val_l2_error": [0.9, 0.7]}
    trainer.save_checkpoint(model, optimizer, epoch=5, path=path, history=history)

    new_model = FNO2d(
        modes1=4, modes2=4, width=8, in_channels=3, out_channels=1, n_layers=1
    )
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    epoch, loaded_history = trainer.load_checkpoint(new_model, new_optimizer, path)

    assert epoch == 5
    assert loaded_history == history
    for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_fit_resumes_from_checkpoint_instead_of_restarting(tmp_path):
    torch.manual_seed(0)
    checkpoint_path = tmp_path / "ckpt.pt"
    train_loader = _make_synthetic_loader()
    val_loader = _make_synthetic_loader()

    model = FNO2d(
        modes1=4, modes2=4, width=8, in_channels=3, out_channels=1, n_layers=1
    )
    config = trainer.TrainConfig(
        epochs=3, device="cpu", use_wandb=False, checkpoint_path=str(checkpoint_path)
    )
    first_history = trainer.fit(model, train_loader, val_loader, config)
    assert len(first_history["train_loss"]) == 3

    # Simulate resuming after an interruption: same checkpoint_path, more epochs requested.
    resumed_model = FNO2d(
        modes1=4, modes2=4, width=8, in_channels=3, out_channels=1, n_layers=1
    )
    resumed_config = trainer.TrainConfig(
        epochs=5, device="cpu", use_wandb=False, checkpoint_path=str(checkpoint_path)
    )
    second_history = trainer.fit(
        resumed_model, train_loader, val_loader, resumed_config
    )

    assert len(second_history["train_loss"]) == 5
    # The first 3 entries must be untouched (not recomputed) -- proof it actually resumed.
    assert second_history["train_loss"][:3] == first_history["train_loss"]

    # Calling fit() again with the same (already-reached) epoch count is a no-op.
    noop_history = trainer.fit(resumed_model, train_loader, val_loader, resumed_config)
    assert noop_history == second_history


def test_scheduler_decays_lr_on_schedule():
    torch.manual_seed(0)
    train_loader = _make_synthetic_loader()
    model = FNO2d(
        modes1=4, modes2=4, width=8, in_channels=3, out_channels=1, n_layers=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    lrs_seen = []
    for _ in range(4):
        trainer.train_one_epoch(model, train_loader, optimizer, torch.device("cpu"))
        scheduler.step()
        lrs_seen.append(optimizer.param_groups[0]["lr"])

    assert lrs_seen == pytest.approx([1e-3, 1e-3 * 0.5, 1e-3 * 0.5, 1e-3 * 0.25])


def test_fit_applies_and_resumes_lr_schedule(tmp_path):
    torch.manual_seed(0)
    checkpoint_path = tmp_path / "ckpt.pt"
    train_loader = _make_synthetic_loader()
    val_loader = _make_synthetic_loader()

    model = FNO2d(
        modes1=4, modes2=4, width=8, in_channels=3, out_channels=1, n_layers=1
    )
    config = trainer.TrainConfig(
        epochs=2,
        lr=1e-3,
        scheduler_step=1,
        scheduler_gamma=0.5,
        device="cpu",
        use_wandb=False,
        checkpoint_path=str(checkpoint_path),
    )
    trainer.fit(model, train_loader, val_loader, config)

    checkpoint = torch.load(checkpoint_path)
    assert checkpoint["scheduler_state"]["_last_lr"] == pytest.approx([1e-3 * 0.25])

    # Resume for 2 more epochs -- LR must continue decaying from where it left
    # off (0.25x), not reset to the initial lr.
    resumed_model = FNO2d(
        modes1=4, modes2=4, width=8, in_channels=3, out_channels=1, n_layers=1
    )
    resumed_config = trainer.TrainConfig(
        epochs=4,
        lr=1e-3,
        scheduler_step=1,
        scheduler_gamma=0.5,
        device="cpu",
        use_wandb=False,
        checkpoint_path=str(checkpoint_path),
    )
    trainer.fit(resumed_model, train_loader, val_loader, resumed_config)

    resumed_checkpoint = torch.load(checkpoint_path)
    assert resumed_checkpoint["scheduler_state"]["_last_lr"] == pytest.approx(
        [1e-3 * 0.0625]
    )


class _FakeRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, data):
        self.logged.append(data)

    def finish(self):
        self.finished = True


class _FakeWandb:
    def __init__(self):
        self.init_calls = []
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


def test_fit_logs_to_wandb_when_enabled(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(trainer, "wandb", fake_wandb)

    model = FNO2d(
        modes1=4, modes2=4, width=8, in_channels=3, out_channels=1, n_layers=1
    )
    train_loader = _make_synthetic_loader()
    val_loader = _make_synthetic_loader()
    config = trainer.TrainConfig(epochs=2, device="cpu", use_wandb=True)

    history = trainer.fit(model, train_loader, val_loader, config)

    assert len(fake_wandb.init_calls) == 1
    assert len(fake_wandb.run.logged) == 2
    assert fake_wandb.run.finished is True
    assert len(history["train_loss"]) == 2


def test_fit_skips_wandb_when_disabled(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(trainer, "wandb", fake_wandb)

    model = FNO2d(
        modes1=4, modes2=4, width=8, in_channels=3, out_channels=1, n_layers=1
    )
    train_loader = _make_synthetic_loader()
    val_loader = _make_synthetic_loader()
    config = trainer.TrainConfig(epochs=1, device="cpu", use_wandb=False)

    trainer.fit(model, train_loader, val_loader, config)

    assert fake_wandb.init_calls == []


@pytest.mark.skipif(not DARCY_FILE.exists(), reason="Darcy sample not downloaded")
def test_fit_on_small_real_darcy_subset():
    ds = DarcyFlowDataset(DARCY_FILE, split="train")
    # Mechanics check only -- not a real train/val split or accuracy benchmark.
    train_loader = DataLoader(
        Subset(ds, range(16)), batch_size=4, collate_fn=darcy_collate
    )
    val_loader = DataLoader(
        Subset(ds, range(16, 24)), batch_size=4, collate_fn=darcy_collate
    )

    model = FNO2d(
        modes1=8, modes2=8, width=16, in_channels=3, out_channels=1, n_layers=2
    )
    config = trainer.TrainConfig(epochs=2, lr=1e-3, device="cpu", use_wandb=False)
    history = trainer.fit(model, train_loader, val_loader, config)

    assert len(history["train_loss"]) == 2
    assert all(v >= 0 for v in history["val_l2_error"])


@pytest.mark.skipif(not BURGERS_FILE.exists(), reason="Burgers sample not downloaded")
def test_fit_on_small_real_burgers_subset():
    ds = Burgers1DDataset(BURGERS_FILE, initial_step=10, split="train")
    # Mechanics check only -- not a real train/val split or accuracy benchmark.
    train_loader = DataLoader(
        Subset(ds, range(16)), batch_size=4, collate_fn=burgers_collate
    )
    val_loader = DataLoader(
        Subset(ds, range(16, 24)), batch_size=4, collate_fn=burgers_collate
    )

    model = FNO1d(modes=16, width=16, in_channels=11, out_channels=1, n_layers=2)
    config = trainer.TrainConfig(epochs=2, lr=1e-3, device="cpu", use_wandb=False)
    history = trainer.fit(model, train_loader, val_loader, config)

    assert len(history["train_loss"]) == 2
    assert all(v >= 0 for v in history["val_l2_error"])


@pytest.mark.skipif(not NS_FILE.exists(), reason="NS_incom shard not downloaded")
def test_fit_on_small_real_navier_stokes_subset():
    # Only 4 trajectories per shard -- fractions chosen so train/val are both non-empty.
    # spatial_stride=8 (512 -> 64) keeps this fast without loading full-res arrays.
    fractions = (0.5, 0.25, 0.25)
    train_ds = NavierStokes2DDataset(
        NS_FILE,
        initial_step=5,
        split="train",
        split_fractions=fractions,
        spatial_stride=8,
    )
    val_ds = NavierStokes2DDataset(
        NS_FILE,
        initial_step=5,
        split="val",
        split_fractions=fractions,
        spatial_stride=8,
    )

    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=navier_stokes_collate)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=navier_stokes_collate)

    model = FNO2d(
        modes1=8, modes2=8, width=8, in_channels=5 * 2 + 2, out_channels=2, n_layers=1
    )
    config = trainer.TrainConfig(epochs=1, lr=1e-3, device="cpu", use_wandb=False)
    history = trainer.fit(model, train_loader, val_loader, config)

    assert len(history["train_loss"]) == 1
    assert all(v >= 0 for v in history["val_l2_error"])


@pytest.mark.skipif(not DARCY_FILE.exists(), reason="Darcy sample not downloaded")
def test_fit_deeponet_on_small_real_darcy_subset():
    ds = DarcyFlowDataset(DARCY_FILE, split="train")
    # Mechanics check only -- not a real train/val split or accuracy benchmark.
    train_loader = DataLoader(
        Subset(ds, range(16)), batch_size=4, collate_fn=darcy_deeponet_collate
    )
    val_loader = DataLoader(
        Subset(ds, range(16, 24)), batch_size=4, collate_fn=darcy_deeponet_collate
    )

    model = DeepONet(
        branch_input_dim=128 * 128,
        trunk_input_dim=2,
        hidden_dim=32,
        latent_dim=16,
        out_channels=1,
    )
    config = trainer.TrainConfig(epochs=2, lr=1e-3, device="cpu", use_wandb=False)
    history = trainer.fit(model, train_loader, val_loader, config)

    assert len(history["train_loss"]) == 2
    assert all(v >= 0 for v in history["val_l2_error"])


@pytest.mark.skipif(not BURGERS_FILE.exists(), reason="Burgers sample not downloaded")
def test_fit_deeponet_on_small_real_burgers_subset():
    ds = Burgers1DDataset(BURGERS_FILE, initial_step=10, split="train")
    # Mechanics check only -- not a real train/val split or accuracy benchmark.
    train_loader = DataLoader(
        Subset(ds, range(16)), batch_size=4, collate_fn=burgers_deeponet_collate
    )
    val_loader = DataLoader(
        Subset(ds, range(16, 24)), batch_size=4, collate_fn=burgers_deeponet_collate
    )

    model = DeepONet(
        branch_input_dim=10 * 1024,
        trunk_input_dim=1,
        hidden_dim=32,
        latent_dim=16,
        out_channels=1,
    )
    config = trainer.TrainConfig(epochs=2, lr=1e-3, device="cpu", use_wandb=False)
    history = trainer.fit(model, train_loader, val_loader, config)

    assert len(history["train_loss"]) == 2
    assert all(v >= 0 for v in history["val_l2_error"])


@pytest.mark.skipif(not NS_FILE.exists(), reason="NS_incom shard not downloaded")
def test_fit_deeponet_on_small_real_navier_stokes_subset():
    # Only 4 trajectories per shard -- fractions chosen so train/val are both non-empty.
    # spatial_stride=8 (512 -> 64) keeps this fast without loading full-res arrays.
    fractions = (0.5, 0.25, 0.25)
    train_ds = NavierStokes2DDataset(
        NS_FILE,
        initial_step=5,
        split="train",
        split_fractions=fractions,
        spatial_stride=8,
    )
    val_ds = NavierStokes2DDataset(
        NS_FILE,
        initial_step=5,
        split="val",
        split_fractions=fractions,
        spatial_stride=8,
    )

    train_loader = DataLoader(
        train_ds, batch_size=1, collate_fn=navier_stokes_deeponet_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, collate_fn=navier_stokes_deeponet_collate
    )

    model = DeepONet(
        branch_input_dim=5 * 2 * 64 * 64,
        trunk_input_dim=2,
        hidden_dim=32,
        latent_dim=16,
        out_channels=2,
    )
    config = trainer.TrainConfig(epochs=1, lr=1e-3, device="cpu", use_wandb=False)
    history = trainer.fit(model, train_loader, val_loader, config)

    assert len(history["train_loss"]) == 1
    assert all(v >= 0 for v in history["val_l2_error"])
