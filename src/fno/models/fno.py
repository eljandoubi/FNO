"""Fourier Neural Operator (FNO) building blocks and models.

Reference: Li et al., "Fourier Neural Operator for Parametric Partial
Differential Equations" (https://arxiv.org/abs/2010.08895).

Convention: models take channel-last input (B, *spatial, C_in) -- typically
the raw field(s) concatenated with grid coordinates along the last axis --
and return channel-last output (B, *spatial, C_out). Spectral convolutions
truncate to a fixed number of low-frequency Fourier modes independent of the
spatial resolution, which is what gives FNO its zero-shot super-resolution
property.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv1d(nn.Module):
    """Truncate to `modes` low frequencies, apply a learned complex linear
    map per mode, then transform back."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C_in, X)
        batch_size, _, n = x.shape
        x_ft = torch.fft.rfft(x, norm="ortho")
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :modes], self.weight[:, :, :modes]
        )
        return torch.fft.irfft(out_ft, n=n, norm="ortho")


class SpectralConv2d(nn.Module):
    """Truncate to `modes1` x `modes2` low frequencies (kept separately for
    the positive and negative frequencies of the first spatial axis, since
    `rfft2` only halves the last axis)."""

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C_in, H, W)
        batch_size = x.shape[0]
        height, width = x.shape[-2], x.shape[-1]
        x_ft = torch.fft.rfft2(x, norm="ortho")
        modes1 = min(self.modes1, height // 2)
        modes2 = min(self.modes2, x_ft.shape[-1])
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.shape[-2],
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :modes1, :modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :modes1, :modes2],
            self.weight1[:, :, :modes1, :modes2],
        )
        out_ft[:, :, -modes1:, :modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -modes1:, :modes2],
            self.weight2[:, :, :modes1, :modes2],
        )
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class _FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral_conv(x) + self.pointwise(x))


class _FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral_conv(x) + self.pointwise(x))


class FNO1d(nn.Module):
    """FNO for 1D function-to-function mapping (e.g. Burgers' equation).

    Input/output are channel-last: (B, X, C_in) -> (B, X, C_out). `in_channels`
    typically covers the input field(s) plus a grid-coordinate channel.
    """

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        in_channels: int = 2,
        out_channels: int = 1,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.lifting = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList(
            [_FNOBlock1d(width, modes) for _ in range(n_layers)]
        )
        self.project_in = nn.Linear(width, width * 2)
        self.project_out = nn.Linear(width * 2, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.project_in(x))
        return self.project_out(x)


class FNO2d(nn.Module):
    """FNO for 2D function-to-function mapping (e.g. Darcy Flow).

    Input/output are channel-last: (B, H, W, C_in) -> (B, H, W, C_out).
    `in_channels` typically covers the input field(s) plus two grid-coordinate
    channels (x, y).
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        in_channels: int = 3,
        out_channels: int = 1,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.lifting = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList(
            [_FNOBlock2d(width, modes1, modes2) for _ in range(n_layers)]
        )
        self.project_in = nn.Linear(width, width * 2)
        self.project_out = nn.Linear(width * 2, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        x = x.permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.project_in(x))
        return self.project_out(x)
