"""Lightweight MLIR text analyzer — regex-based, no MLIR parser dependency.

Produces the JSON summary used by `IRSummaryAnalyzer` stage and by the
SHARK-AI comparison (`docs/SHARK_AI_ANALYSIS.md`):

  * `op_counts` — `torch.aten.*` op frequency
  * `dtypes`    — f16/f32/i64/... distribution
  * `has_dynamic_dim` — `?` symbol in vtensor shapes
  * `server_side_op_hits` — paged_attention / kv_cache / vllm tokens
    (target = 0 for on-device, ≥1 in the SHARK reference)
"""
from __future__ import annotations

import re
from collections import Counter

SERVER_SIDE_HINTS = (
    "paged_attention",
    "paged_kv_cache",
    "kv_cache",
    "flash_attention",
    "vllm",
    "tensor_parallel",
    "device_affinity",
)

_ATEN_OP = re.compile(r"torch\.aten\.([a-zA-Z0-9_]+)")
_DTYPE = re.compile(r"!torch\.vtensor<[^>]*?,\s*([a-z0-9]+)>")
_MODULE = re.compile(r"module\s+(?:attributes\s+\{[^}]*\})?\s*\{")


def _count_aten_ops(text: str) -> dict[str, int]:
    return dict(Counter(_ATEN_OP.findall(text)))


def _scan_dtypes(text: str) -> dict[str, int]:
    return dict(Counter(_DTYPE.findall(text)))


def _scan_server_side(text: str) -> dict[str, int]:
    low = text.lower()
    return {hint: low.count(hint) for hint in SERVER_SIDE_HINTS if hint in low}


def _extract_module_name(text: str) -> str | None:
    m = re.search(r'torch\.debug_module_name\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else None


def summarize(mlir_text: str) -> dict:
    return {
        "n_lines": len(mlir_text.splitlines()),
        "module_name": _extract_module_name(mlir_text),
        "op_counts": _count_aten_ops(mlir_text),
        "dtypes": _scan_dtypes(mlir_text),
        "has_dynamic_dim": "vtensor<?" in mlir_text or re.search(r"\?\s*x", mlir_text) is not None,
        "server_side_op_hits": _scan_server_side(mlir_text),
    }
