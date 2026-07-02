"""Llama-style RMSNorm — on-device replacement for amdsharktank
`RMSNormLayer(ThetaLayer)`.

Differences vs server-side reference:
  * plain `nn.Module` with `nn.Parameter` weight (no Theta wrapping)
  * dtype-stable: cast input to float32 for the rsqrt, return to original
    dtype (matches amdsharktank semantics)
"""
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int = 512, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (self.weight * x32).to(orig_dtype)
