"""Embedding quantization + per-channel policy — the footprint half of INT8.

`quantize_linears_` skips tied weights on purpose, and in a tied model the
embedding table *is* the tied weight — so it stays fp32 and remains the largest
tensor. These lock the policy that closes that gap.
"""
import pytest
import torch
import torch.nn as nn

from torch_mlir_zoo.kernels import (
    BlockScaledQ8Embedding,
    BlockScaledQ8Linear,
    quantize_block_scaled_q8,
    quantize_embeddings_,
    quantize_linears_,
)


class _TiedLM(nn.Module):
    """Minimal tied-embedding LM: lm_head shares the embedding table."""

    def __init__(self, vocab=64, dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # tie

    def forward(self, ids):
        return self.lm_head(self.proj(self.embed(ids)))


def test_embedding_lookup_matches_dequantized_table():
    torch.manual_seed(0)
    emb = nn.Embedding(64, 32)
    quant = BlockScaledQ8Embedding.from_embedding(emb, block_size=8)

    ids = torch.tensor([[1, 5, 63], [0, 2, 7]])
    qs, d = quantize_block_scaled_q8(emb.weight.detach(), 8)
    table = (qs.to(torch.float32) * d).reshape(64, 32)

    assert torch.allclose(quant(ids), table[ids], atol=1e-6)
    assert quant(ids).shape == (2, 3, 32)


def test_quantized_embedding_stays_close_to_fp():
    torch.manual_seed(0)
    emb = nn.Embedding(64, 32)
    quant = BlockScaledQ8Embedding.from_embedding(emb)
    ids = torch.arange(64).reshape(1, 64)
    err = (quant(ids) - emb(ids)).abs().max() / emb.weight.abs().max()
    assert err < 0.02  # Q8_0 round-off only


def test_tied_head_shares_the_quantized_table():
    torch.manual_seed(0)
    model = _TiedLM()
    # linear pass alone: the tie is skipped, table still fp32
    quantize_linears_(model)
    assert isinstance(model.lm_head, nn.Linear)
    assert isinstance(model.embed, nn.Embedding)

    quantize_embeddings_(model)
    assert isinstance(model.embed, BlockScaledQ8Embedding)
    assert isinstance(model.lm_head, BlockScaledQ8Linear)
    # one table, two views — the tie survived quantization
    assert model.lm_head.qs.data_ptr() == model.embed.qs.data_ptr()
    assert model.lm_head.d.data_ptr() == model.embed.d.data_ptr()


def test_tied_model_output_survives_quantization():
    torch.manual_seed(0)
    model = _TiedLM().eval()
    ids = torch.arange(16).reshape(1, 16)
    with torch.no_grad():
        ref = model(ids)
    quantize_linears_(model)
    quantize_embeddings_(model)
    with torch.no_grad():
        out = model(ids)
    cosine = torch.nn.functional.cosine_similarity(ref.flatten(), out.flatten(), dim=0)
    assert cosine > 0.99


def test_per_channel_is_block_size_none():
    torch.manual_seed(0)
    weight = torch.randn(8, 32)
    qs, d = quantize_block_scaled_q8(weight, block_size=None)
    assert qs.shape == (8, 1, 32)  # one block per row
    assert d.shape == (8, 1, 1)  # one scale per output channel

    linear = nn.Linear(32, 8, bias=False)
    quantized = BlockScaledQ8Linear.from_linear(linear, block_size=None)
    assert quantized.block_size == 32
    err = (quantized(torch.randn(1, 4, 32)) - linear(torch.randn(1, 4, 32))).abs()
    assert torch.isfinite(err).all()


def test_export_reports_embedding_quantization():
    pytest.importorskip("iree.turbine")
    from torch_mlir_zoo import export_for_npu

    ids = (torch.zeros(1, 8, dtype=torch.long),)
    plain = export_for_npu(_TiedLM(), ids, backend="iree_turbine", quantize="int8")
    full = export_for_npu(
        _TiedLM(), ids, backend="iree_turbine", quantize="int8", quantize_embeddings=True
    )
    # the tied fp32 table is gone from the IR: si8 shows up, and the kernel is used
    assert full.summary["dtypes"].get("si8", 0) > plain.summary["dtypes"].get("si8", 0)
    assert full.ok is True
