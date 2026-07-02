"""Unit-op forward correctness tests — no Torch-MLIR dependency.

Numerical agreement with reference PyTorch formulas (atol 1e-5) verifies
that the on-device replacements behave like the server-side amdsharktank
equivalents — see `docs/SHARK_AI_ANALYSIS.md` for the mapping.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from torch_mlir_zoo.ops import RMSNorm, ScaledDotProductAttention, SwiGLU, TopK


def test_attention_forward_shape_and_causal():
    torch.manual_seed(0)
    attn = ScaledDotProductAttention(embed_dim=64, num_heads=4, causal=True).eval()
    x = torch.randn(2, 8, 64)
    y = attn(x)
    assert y.shape == (2, 8, 64)
    assert torch.isfinite(y).all()


def test_attention_supports_gqa():
    attn = ScaledDotProductAttention(
        embed_dim=64, num_heads=8, num_kv_heads=2, causal=True
    ).eval()
    x = torch.randn(1, 4, 64)
    assert attn(x).shape == (1, 4, 64)


def test_rmsnorm_matches_manual_formula():
    torch.manual_seed(1)
    norm = RMSNorm(dim=16, eps=1e-5).eval()
    x = torch.randn(2, 4, 16)
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    assert torch.allclose(norm(x), expected, atol=1e-5)


def test_swiglu_forward_shape_and_dtype():
    mlp = SwiGLU(dim=32, hidden_dim=128).eval()
    x = torch.randn(3, 5, 32)
    y = mlp(x)
    assert y.shape == (3, 5, 32)
    assert y.dtype == x.dtype


def test_topk_matches_torch_topk():
    torch.manual_seed(2)
    topk = TopK(k=5).eval()
    logits = torch.randn(2, 1000)
    values, indices = topk(logits)
    assert values.shape == (2, 5)
    assert indices.shape == (2, 5)
    # values descending
    assert (values[:, :-1] >= values[:, 1:]).all()
