"""Scaled dot-product attention — on-device replacement for amdsharktank
`PagedMHAttention` / `PagedGQAttention`.

Differences vs server-side reference:
  * no KV cache, no paging (forward-only, fixed seq_len)
  * no fused/handwritten kernel — pure standard ops so torch-mlir lowering
    sees `aten.matmul` / `aten.softmax` rather than an opaque op
  * GQA supported via `num_kv_heads` (repeat_interleave on K/V)
"""
import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        causal: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.triu(
                torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1
            )
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out)
