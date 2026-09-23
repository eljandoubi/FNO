# FNO

## Executive Summary

This repository provides a high-performance **Fourier Neural Operator (FNO)** implementation in PyTorch designed for fast, mesh-independent surrogate modeling of complex fluid dynamics and non-linear partial differential equations (PDEs). 

Developed as part of a Scientific Machine Learning benchmark (MVA / ENSTA Paris), this project evaluates operator learning architectures (**FNO**, **DeepONet**, and **PINNs**) on benchmark physical systems such as 2D Navier-Stokes and 1D/2D Burgers' equations.

### Key Highlights
- **Resolution Independence:** Learns mapping between continuous function spaces, enabling zero-shot super-resolution (e.g., training on $64 \times 64$ resolution grids and evaluating directly on $256 \times 256$ meshes without fine-tuning).
- **100x Speedup:** Accelerates numerical field evaluation by up to two orders of magnitude compared to traditional Finite Element Method (FEM) solvers, achieving $< 2\%$ relative $L^2$ error.
- **Modern Python Tooling:** Managed via [`uv`](https://github.com/astral-sh/uv) for lightning-fast, reproducible dependency management and environment isolation.
