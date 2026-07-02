"""Llama-style SwiGLU MLP — on-device replacement for amdsharktank
`FFN(ThetaLayer)` (gated SiLU variant).

Differences vs server-side reference:
  * plain `nn.Module` with three `nn.Linear` (gate / up / down)
  * no Theta wrapping, no sharding hooks
  * forward: `down(silu(gate(x)) * up(x))`
"""
import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 2048):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
