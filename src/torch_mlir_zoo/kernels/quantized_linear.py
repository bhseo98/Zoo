"""Model-side API for the target NPU block-scaled INT8 matmul kernel.

Three ways a model calls the kernel (guideline §4):

  1. direct:  ``block_scaled_q8(a, d, qs)``        — the registered op
  2. helper:  ``quantized_matmul(a, d, qs)``       — named wrapper
  3. module:  ``BlockScaledQ8Linear``              — ``nn.Linear`` drop-in

Plus :func:`quantize_block_scaled_q8`, a symmetric per-block Q8_0 quantizer for
building ``(qs, d)`` from a float weight (used by ``from_linear`` and tests).
"""
from __future__ import annotations

import torch
from torch import nn

from .block_scaled_q8 import block_scaled_q8

__all__ = [
    "quantize_block_scaled_q8",
    "quantized_matmul",
    "BlockScaledQ8Linear",
    "BlockScaledQ8Embedding",
    "quantize_linears_",
    "quantize_embeddings_",
]


def resolve_block_size(block_size: int | None, k: int) -> int:
    """``None`` means per-channel — one scale per output row (block = K)."""
    return k if block_size is None else block_size


def quantize_block_scaled_q8(
    weight: torch.Tensor, block_size: int | None = 32, headroom: float = 127.0
):
    """Symmetric per-block Q8_0 quantization of a ``[N, K]`` weight.

    Returns ``(qs, d)`` with ``qs`` int8 ``[N, G, BS]`` and ``d`` float
    ``[N, G, 1]``, where ``K == G * BS``. Levels are ``[-127, 127]`` (Q8_0).
    ``block_size=None`` gives per-channel scaling (``BS == K``, one scale per
    row) — the granularity most NPU INT8 paths support natively.

    ``headroom`` is the divisor that maps a block's max magnitude onto the int8
    range. 127 uses the full range. A smaller value leaves the top of the range
    unused so a downstream accumulator cannot overflow — a device that casts
    each block's partial sum to a narrower float before accumulating needs the
    weights to stay well inside int8, and the amount of slack is a property of
    that accumulator, not of this function.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2d [N,K], got {tuple(weight.shape)}")
    n, k = weight.shape
    block_size = resolve_block_size(block_size, k)
    if k % block_size != 0:
        raise ValueError(f"K={k} not divisible by block_size={block_size}")
    g = k // block_size
    w = weight.reshape(n, g, block_size)
    amax = w.abs().amax(dim=-1, keepdim=True)
    d = (amax / headroom).clamp(min=1e-12)
    qs = torch.round(w / d).clamp(-127, 127).to(torch.int8)
    return qs, d.to(weight.dtype)


def quantized_matmul(a: torch.Tensor, d: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
    """``a[B,M,K] @ dequant(qs, d)^T -> [B,M,N]`` via the target NPU kernel."""
    return block_scaled_q8(a, d, qs)


class BlockScaledQ8Linear(nn.Module):
    """``nn.Linear`` drop-in backed by INT8 block-scaled weight (optional bias).

    The weight is stored as int8 ``qs`` + per-block float scale ``d`` and never
    materialized as a full fp tensor at inference; the kernel fuses dequant into
    the matmul. Accepts inputs of any rank ``[*, K]`` — leading dims are folded
    into the M axis to satisfy the kernel's ``[B, M, K]`` contract.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int | None = 32,
        bias: bool = False,
    ):
        super().__init__()
        block_size = resolve_block_size(block_size, in_features)
        if in_features % block_size != 0:
            raise ValueError(
                f"in_features={in_features} not divisible by block_size={block_size}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        g = in_features // block_size
        self.register_buffer(
            "qs", torch.zeros(out_features, g, block_size, dtype=torch.int8)
        )
        self.register_buffer("d", torch.ones(out_features, g, 1, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead, k = x.shape[:-1], x.shape[-1]
        out = quantized_matmul(x.reshape(1, -1, k), self.d, self.qs)  # [1, prod(lead), N]
        out = out.reshape(*lead, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = 32) -> "BlockScaledQ8Linear":
        """Build from an existing ``nn.Linear`` (quantizes the weight; keeps bias)."""
        mod = cls(
            linear.in_features,
            linear.out_features,
            block_size,
            bias=linear.bias is not None,
        )
        qs, d = quantize_block_scaled_q8(linear.weight.detach(), block_size)
        mod.qs.copy_(qs)
        mod.d.copy_(d)
        if linear.bias is not None:
            mod.bias.copy_(linear.bias.detach())
        return mod


def _tied_weight_ptrs(module: nn.Module) -> set[int]:
    """Data pointers of weights shared by more than one submodule.

    A tied weight (e.g. ``lm_head``/``proj_out`` sharing the decoder
    ``embed_tokens`` table) appears once per holder when duplicates are kept.
    """
    from collections import Counter

    seen = Counter(
        p.data_ptr() for _, p in module.named_parameters(remove_duplicate=False)
    )
    return {ptr for ptr, count in seen.items() if count > 1}


class BlockScaledQ8Embedding(nn.Module):
    """``nn.Embedding`` drop-in with an INT8 block-scaled table.

    The embedding table is the single largest tensor in a small LLM
    (Llama-3.2-1B: 128256 x 2048 fp32 = 1.05 GB, the bulk of the 2.46 GB vmfb),
    and it is *looked up*, not multiplied — so it needs no kernel: rows are
    gathered as int8 and dequantized with standard aten ops (index / mul).

    Layout matches :func:`quantize_block_scaled_q8` exactly (``qs[V,G,BS]`` +
    ``d[V,G,1]``), which is also the ``block_scaled_q8`` weight layout — so a
    tied ``lm_head`` can share these very buffers instead of storing a copy.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, block_size: int | None = 32):
        super().__init__()
        block_size = resolve_block_size(block_size, embedding_dim)
        if embedding_dim % block_size != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} not divisible by block_size={block_size}"
            )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.groups = embedding_dim // block_size
        self.register_buffer(
            "qs", torch.zeros(num_embeddings, self.groups, block_size, dtype=torch.int8)
        )
        self.register_buffer(
            "d", torch.ones(num_embeddings, self.groups, 1, dtype=torch.float32)
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1)
        rows = self.qs.reshape(self.num_embeddings, self.embedding_dim)[flat]
        scales = self.d.reshape(self.num_embeddings, self.groups)[flat]
        out = rows.reshape(-1, self.groups, self.block_size).to(scales.dtype) * scales.unsqueeze(-1)
        return out.reshape(*ids.shape, self.embedding_dim)

    @classmethod
    def from_embedding(
        cls, embedding: nn.Embedding, block_size: int | None = 32
    ) -> "BlockScaledQ8Embedding":
        mod = cls(embedding.num_embeddings, embedding.embedding_dim, block_size)
        qs, d = quantize_block_scaled_q8(embedding.weight.detach(), block_size)
        mod.qs.copy_(qs)
        mod.d.copy_(d.to(torch.float32))
        return mod


def _linear_sharing(module: nn.Module, weight: torch.Tensor) -> list[tuple[nn.Module, str]]:
    """``(parent, attr)`` of every ``nn.Linear`` tied to ``weight``."""
    ptr = weight.data_ptr()
    return [
        (parent, name)
        for parent in module.modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.Linear) and child.weight.data_ptr() == ptr
    ]


def quantize_embeddings_(
    module: nn.Module, block_size: int | None = 32, share_tied: bool = True
) -> nn.Module:
    """In-place: replace every ``nn.Embedding`` with :class:`BlockScaledQ8Embedding`.

    Embeddings are the footprint item ``quantize_linears_`` deliberately leaves
    alone: it skips tied weights, and in a tied model (Llama, Qwen) the table is
    exactly the tied one — so a linear-only pass leaves the biggest tensor fp32.

    ``share_tied`` closes that: an ``nn.Linear`` tied to the table becomes a
    :class:`BlockScaledQ8Linear` that *shares* the quantized buffers, so the tie
    survives quantization and the table is stored once, not twice.
    """
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if not isinstance(child, nn.Embedding):
                continue
            tied_linears = _linear_sharing(module, child.weight) if share_tied else []
            quantized = BlockScaledQ8Embedding.from_embedding(child, block_size)
            setattr(parent, name, quantized)
            for lin_parent, lin_name in tied_linears:
                linear = getattr(lin_parent, lin_name)
                shared = BlockScaledQ8Linear(
                    linear.in_features,
                    linear.out_features,
                    quantized.block_size,
                    bias=linear.bias is not None,
                )
                shared.qs = quantized.qs  # one table, two views
                shared.d = quantized.d
                if linear.bias is not None:
                    shared.bias.copy_(linear.bias.detach())
                setattr(lin_parent, lin_name, shared)
    return module


def quantize_linears_(
    module: nn.Module, block_size: int | None = 32, _tied: set[int] | None = None
) -> nn.Module:
    """In-place: replace every quantizable ``nn.Linear`` (``K %% block_size == 0``)
    in ``module`` with :class:`BlockScaledQ8Linear`. Returns ``module`` for chaining.

    This is the model-side "apply to the framework" hook — drop any PyTorch model
    in and its Linear layers run on the target NPU block-scaled INT8 kernel.

    Linears whose weight is *tied* to another module (e.g. a ``proj_out``/``lm_head``
    sharing the fp32 ``embed_tokens`` table) are skipped: quantizing one half of a
    tie breaks it and stores a redundant INT8 copy of a weight meant to stay full
    precision. ``_tied`` is computed once on the top-level call.

    A bare top-level ``nn.Linear`` has no children to walk, so it is replaced
    directly and the replacement is *returned* (there is no parent to ``setattr``
    on) — callers must use the return value, e.g. ``m = quantize_linears_(m)``.
    """
    def _quantizable(layer: nn.Linear) -> bool:
        bs = resolve_block_size(block_size, layer.in_features)
        return layer.in_features % bs == 0 and layer.weight.data_ptr() not in _tied

    top_level = _tied is None
    if _tied is None:
        _tied = _tied_weight_ptrs(module)
    if top_level and isinstance(module, nn.Linear) and _quantizable(module):
        return BlockScaledQ8Linear.from_linear(module, block_size)
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and _quantizable(child):
            setattr(module, name, BlockScaledQ8Linear.from_linear(child, block_size))
        else:
            quantize_linears_(child, block_size, _tied)
    return module
