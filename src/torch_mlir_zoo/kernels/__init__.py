"""target NPU CustomOp kernels (design scaffold).

Backend-independent kernel layer for the target NPU path: the CustomOp registration,
shape contract (``select``), model-backend call functions, and a *portable*
standard-linalg microkernel. The target-specific intrinsic microkernel is a
documented swap-in point pending the target NPU library/runtime handoff.

See ``docs/kernel-design/01-guideline-and-plan.md``.
"""
from .block_scaled_q8 import block_scaled_q8
from .fused_rmsnorm import FusedRMSNorm, fused_rmsnorm
from .quantized_linear import (
    BlockScaledQ8Embedding,
    BlockScaledQ8Linear,
    quantize_block_scaled_q8,
    quantize_embeddings_,
    quantize_linears_,
    quantized_matmul,
)

__all__ = [
    "block_scaled_q8",
    "quantized_matmul",
    "quantize_block_scaled_q8",
    "BlockScaledQ8Linear",
    "BlockScaledQ8Embedding",
    "quantize_linears_",
    "quantize_embeddings_",
    "fused_rmsnorm",
    "FusedRMSNorm",
]
