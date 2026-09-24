# FNO project task runner -- see https://github.com/casey/just
# Run `just --list` (or bare `just`) to see all available recipes.

# List available recipes
default:
    @just --list

# Install/sync all locked dependencies into .venv
sync:
    uv sync

# Run the test suite (extra args pass through, e.g. `just test -k darcy`)
test *args:
    uv run pytest -v {{args}}

# Lint with ruff
lint:
    uv run ruff check

# Auto-fix lint issues where possible
fix:
    uv run ruff check --fix

# Reformat code with ruff
fmt:
    uv run ruff format

# Check formatting without modifying files
fmt-check:
    uv run ruff format --check

# Run everything CI runs (lint + tests)
ci: lint test

# Download a dataset, e.g. `just download pdebench-darcy2d`
download *args:
    uv run fno-download {{args}}

# Inspect an HDF5 file's structure, e.g. `just inspect data/raw/.../file.hdf5`
inspect *args:
    uv run fno-inspect-h5 {{args}}

# Train + compare FNO/DeepONet/PINN, e.g. `just benchmark --equation darcy`
benchmark *args:
    uv run fno-benchmark {{args}}

# Grid/random hyperparameter search, e.g. `just search darcy --file ...`
search *args:
    uv run fno-search {{args}}

# Run the full resumable download -> train -> evaluate -> plot pipeline
pipeline *args:
    uv run fno-pipeline {{args}}
