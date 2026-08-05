"""TopK over the last dim of logits — forward-only sampling primitive.

Mirrors the role of amdsharktank's top-k sampling block in
`ServicePagedLlmModelV1.prefill()`, *without* the surrounding control
flow (temperature / repeat penalty / autoregressive loop) that breaks
torch.export tracing.
"""
import torch
from torch import nn


class TopK(nn.Module):
    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.topk(logits, self.k, dim=-1)
