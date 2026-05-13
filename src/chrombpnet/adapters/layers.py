"""Low-rank residual adapters for frozen PyTorch layers."""

from __future__ import annotations

import torch


class LowRankConv1dAdapter(torch.nn.Module):
    """Wrap a frozen Conv1d with a trainable low-rank residual branch."""

    adapter_kind = "conv1d"

    def __init__(
        self,
        base: torch.nn.Conv1d,
        rank: int,
        scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base, torch.nn.Conv1d):
            raise TypeError(f"base must be Conv1d, got {type(base)!r}")
        if rank < 1:
            raise ValueError("rank must be >= 1")

        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.rank = int(rank)
        self.scale = float(scale)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.down = torch.nn.Conv1d(
            in_channels=base.in_channels,
            out_channels=self.rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=False,
            padding_mode=base.padding_mode,
        )
        self.up = torch.nn.Conv1d(
            in_channels=self.rank,
            out_channels=base.out_channels,
            kernel_size=1,
            bias=False,
        )
        torch.nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, X):
        return self.base(X) + self.scale * self.up(self.dropout(self.down(X)))

    def manifest(self):
        return {
            "kind": self.adapter_kind,
            "in_channels": self.base.in_channels,
            "out_channels": self.base.out_channels,
            "kernel_size": list(self.base.kernel_size),
            "stride": list(self.base.stride),
            "padding": list(self.base.padding),
            "dilation": list(self.base.dilation),
            "groups": self.base.groups,
            "rank": self.rank,
            "scale": self.scale,
        }


class LowRankLinearAdapter(torch.nn.Module):
    """Wrap a frozen Linear layer with a trainable low-rank residual branch."""

    adapter_kind = "linear"

    def __init__(
        self,
        base: torch.nn.Linear,
        rank: int,
        scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base, torch.nn.Linear):
            raise TypeError(f"base must be Linear, got {type(base)!r}")
        if rank < 1:
            raise ValueError("rank must be >= 1")

        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.rank = int(rank)
        self.scale = float(scale)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.down = torch.nn.Linear(base.in_features, self.rank, bias=False)
        self.up = torch.nn.Linear(self.rank, base.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, X):
        return self.base(X) + self.scale * self.up(self.dropout(self.down(X)))

    def manifest(self):
        return {
            "kind": self.adapter_kind,
            "in_features": self.base.in_features,
            "out_features": self.base.out_features,
            "rank": self.rank,
            "scale": self.scale,
        }
