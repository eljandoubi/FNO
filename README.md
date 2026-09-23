# FNO

## Executive Summary

This repository provides a high-performance **Fourier Neural Operator (FNO)** implementation in PyTorch designed for fast, mesh-independent surrogate modeling of complex fluid dynamics and non-linear partial differential equations (PDEs). 

Developed as part of a Scientific Machine Learning benchmark (MVA / ENSTA Paris), this project evaluates operator learning architectures (**FNO**, **DeepONet**, and **PINNs**) on benchmark physical systems such as 2D Navier-Stokes and 1D/2D Burgers' equations.

### Key Highlights
- **Resolution Independence:** Learns mapping between continuous function spaces, enabling zero-shot super-resolution (e.g., training on $64 \times 64$ resolution grids and evaluating directly on $256 \times 256$ meshes without fine-tuning).
- **100x Speedup:** Accelerates numerical field evaluation by up to two orders of magnitude compared to traditional Finite Element Method (FEM) solvers, achieving $< 2\%$ relative $L^2$ error.
- **Modern Python Tooling:** Managed via [`uv`](https://github.com/astral-sh/uv) for lightning-fast, reproducible dependency management and environment isolation.

## Getting Started

Requires [`uv`](https://github.com/astral-sh/uv) (Python 3.12 is pinned via `.python-version` and fetched automatically if missing).

```bash
uv sync
```

This creates a `.venv` with all runtime and development dependencies (PyTorch, NumPy, SciPy, h5py, Weights & Biases, pytest, ruff, ...) locked in `uv.lock`.

## Data

Benchmark datasets are fetched on demand via the `fno-download` CLI -- nothing downloads automatically, and everything it creates lives under a single `--data-root` folder (default `data/`) so it can be wiped with one `rm -rf`.

```bash
uv run fno-download --list                                   # available datasets and variants
uv run fno-download pdebench-darcy2d                          # 2D Darcy Flow sample (~1.2 GB)
uv run fno-download pdebench-burgers1d --variant nu0.01       # 1D Burgers sample
uv run fno-download pdebench-navierstokes2d --variant shard0  # 2D Navier-Stokes shard (~9.9 GB)
uv run fno-download airfrans                                  # 2D airfoil RANS CFD (~10 GB)
```

| Key | Source | Notes |
| --- | --- | --- |
| `pdebench-burgers1d` | [PDEBench](https://github.com/pdebench/PDEBench) | 12 viscosities (`--variant nu0.001` ... `nu4.0`) |
| `pdebench-darcy2d` | PDEBench | 5 diffusion coefficients (`--variant beta0.01` ... `beta100.0`) |
| `pdebench-navierstokes2d` | PDEBench | 5 curated shards out of ~275 (`--variant shard0` ... `shard4`) |
| `airfrans` | [AirfRANS](https://github.com/Extrality/AirfRANS) | irregular-geometry airfoil CFD, single archive |

Files are checksum-verified (MD5) against PDEBench's published hashes where available. Use `--variant` (repeatable) or `--all-variants` to pick which files to pull, and `--force` to re-download.

Before writing a loader for a new/unverified file format, inspect its real structure:

```bash
uv run fno-inspect-h5 data/raw/pdebench-navierstokes2d/ns_incom_inhom_2d_512-0.h5
```

### Loading data

`src/fno/data/pdebench.py` provides PyTorch `Dataset` classes for the three target equations, each validated against real downloaded files (schema confirmed via `fno-inspect-h5`, not just assumed from docs):

- `DarcyFlowDataset` -- permeability field (`nu`) -> steady-state pressure field (`tensor`)
- `Burgers1DDataset` -- initial timesteps -> full spatio-temporal trajectory
- `NavierStokes2DDataset` -- initial velocity frames -> full trajectory; accepts multiple shard paths (each shard only holds 4 trajectories) and concatenates them

```python
from fno.data.pdebench import DarcyFlowDataset

ds = DarcyFlowDataset("data/raw/pdebench-darcy2d/2D_DarcyFlow_beta1.0_Train.hdf5", split="train")
input_field, target_field, grid = ds[0]
```

All three accept `split="train"|"val"|"test"` and `split_fractions=(train, val, test)` (default `(0.8, 0.1, 0.1)`) -- validation is kept separate from the held-out test set specifically so hyperparameter tuning never touches test data.

## Models

Three operator-learning/PDE-solving architectures are implemented, mirroring the README's benchmark scope:

- **`fno.models.fno`** (`FNO1d`/`FNO2d`) -- spectral convolutions truncated to a fixed number of Fourier modes, independent of spatial resolution (zero-shot super-resolution). Trained via `fno.training.trainer.fit()` with the `*_collate` adapters in `fno.training.collate`.
- **`fno.models.deeponet`** (`DeepONet`) -- branch net encodes the input function at fixed sensors, trunk net encodes query coordinates, output is their dot product. The trunk can be queried at arbitrary/new points without retraining, but the branch's input size (sensor sampling) is fixed at construction. Trained the same way via `fit()`, using the `*_deeponet_collate` adapters.
- **`fno.models.pinn`** (`PINN`) + **`fno.training.pinn`** -- a coordinate-to-solution MLP trained by minimizing the PDE residual (via autograd) plus an initial-condition data term. Unlike FNO/DeepONet, a PINN is **not an operator**: it solves one fixed PDE instance (one viscosity, one initial condition) and must be retrained for every new instance. Currently covers 1D Burgers' equation only -- Darcy's residual needs a differentiable interpolation of its discretized coefficient field, not yet implemented.

```python
from fno.models.pinn import PINN
from fno.training.pinn import train_pinn_burgers

model = PINN(in_dim=2, out_dim=1)
history = train_pinn_burgers(
    model, x_ic, u_ic, nu=0.01, domain_x=(0.0, 1.0), domain_t=(0.0, 2.0),
)
```

## Development

```bash
uv run pytest -v    # unit tests
uv run ruff check   # lint
```

