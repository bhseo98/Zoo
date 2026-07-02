"""MLIR text → JSON summary (op-count, dtype, dynamic-dim, server-side hits)."""
from .ir_summary import SERVER_SIDE_HINTS, summarize

__all__ = ["summarize", "SERVER_SIDE_HINTS"]
