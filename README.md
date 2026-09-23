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
uv run fno-download pdebench-navierstokes2d --variant shard0  # 2D Navier-Stokes shard (~8 GB)
uv run fno-download airfrans                                  # 2D airfoil RANS CFD (~10 GB)
```

| Key | Source | Notes |
| --- | --- | --- |
| `pdebench-burgers1d` | [PDEBench](https://github.com/pdebench/PDEBench) | 12 viscosities (`--variant nu0.001` ... `nu4.0`) |
| `pdebench-darcy2d` | PDEBench | 5 diffusion coefficients (`--variant beta0.01` ... `beta100.0`) |
| `pdebench-navierstokes2d` | PDEBench | 5 curated shards out of ~275 (`--variant shard0` ... `shard4`) |
| `airfrans` | [AirfRANS](https://github.com/Extrality/AirfRANS) | irregular-geometry airfoil CFD, single archive |

Files are checksum-verified (MD5) against PDEBench's published hashes where available. Use `--variant` (repeatable) or `--all-variants` to pick which files to pull, and `--force` to re-download.

## Development

```bash
uv run pytest -v    # unit tests
uv run ruff check   # lint
```

