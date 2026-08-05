"""MLIR text → JSON summary (op-count, dtype, dynamic-dim, server-side hits)."""
from .ir_summary import SERVER_SIDE_HINTS, summarize
from .op_audit import FUSED_ATEN_OPS, SUPPORTED_ATEN_OPS, audit

__all__ = [
    "summarize",
    "SERVER_SIDE_HINTS",
    "audit",
    "SUPPORTED_ATEN_OPS",
    "FUSED_ATEN_OPS",
]
