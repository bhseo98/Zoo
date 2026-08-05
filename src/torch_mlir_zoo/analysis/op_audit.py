"""Op-support audit — the on-device allowlist gate.

Follows SiMa ``model_surgery_guard`` in *mechanism only*: bucket every op in an
exported IR into supported / unknown / server-side, and fail (non-zero) when
anything is not on-device suitable. The allowlist is **our own** — the set of
``torch.aten.*`` ops we have verified lower cleanly (seeded from measured zoo
exports), never a hardware datasheet.

Two axes, kept distinct:
  * server-side  — paged-attention / KV-cache / vLLM leak (a *denylist*, from
    ``SERVER_SIDE_HINTS``). Any hit means the model is not on-device.
  * unknown      — an aten op not yet in ``SUPPORTED_ATEN_OPS`` (an *allowlist*).
    New/unlisted ops we have not certified for the join contract.

``audit(mlir, profile)`` returns a report; ``ok`` is True only when there are no
server-side hits and no unknown ops. The allowlist is profile-dependent: the
fused export profile legitimately carries whole ops (``rms_norm``,
``scaled_dot_product_attention``) that the analysis profile would decompose, and
gating both against one list would reject our own handoff IR.
"""
from __future__ import annotations

from .ir_summary import _count_aten_ops, _scan_server_side

# Allowlist — aten ops verified to lower cleanly through the zoo exports
# (attention / rmsnorm / swiglu / topk / llama-on-device). Grow this as new
# ops are certified; the audit flags anything outside it as "unknown".
#
# Certification is a compile, not a guess: `scripts/certify_ops.py` exports a
# tiny probe per op family and compiles it to a CPU vmfb (see
# logs/op-certification.json). The one exception is noted inline.
SUPPORTED_ATEN_OPS = frozenset(
    {
        # elementwise / activation
        "add", "sub", "mul", "div", "neg", "pow", "rsqrt", "sqrt", "exp",
        "silu", "gelu", "relu", "sigmoid", "tanh", "erf",
        # reductions / norm
        "mean", "sum", "max", "softmax", "_softmax", "amax",
        # matmul family
        "mm", "bmm", "matmul", "linear", "addmm",
        # shape / movement
        "view", "_unsafe_view", "reshape", "transpose", "permute", "expand",
        "expand_as", "unsqueeze", "squeeze", "flatten", "cat", "slice",
        "select", "broadcast_to", "contiguous", "to", "type_as", "clone",
        # tensor construction (lowering byproducts)
        "ones", "zeros", "full", "scalar_tensor", "empty",
        # indexing / attention support
        "embedding", "index", "index_select", "gather", "masked_fill",
        "triu", "tril", "arange", "where", "repeat_interleave",
        # selection
        "topk", "argmax", "sort",
        # positional / misc
        "cos", "sin", "stack", "split",
        # certified by scripts/certify_ops.py (probe -> CPU vmfb), 2026-07-27:
        # LayerNorm family (gpt2 / bert / opt / whisper)
        "var_mean", "rsub",
        # mask + buffer byproducts of HF attention masks
        "gt", "eq", "fill", "zeros_like", "empty_like", "copy", "slice_scatter",
        # audio front end / position ids / broadcasting
        "convolution", "cumsum", "repeat",
        # functional dtype cast — only visible when decompositions are skipped,
        # so its probe runs under the fused profile
        "_to_copy",
        # certified instead by the whisper-tiny INT8 end-to-end run (WM3:
        # full vmfb executed on IREE-CPU, rel 0.8%) — no probe reproduces it
        "empty_strided",
    }
)


# Ops that exist only when decompositions are skipped (the fused export
# profile). They are absent from the analysis allowlist on purpose: under the
# default profile their presence would mean a decomposition silently failed.
#
# dropout / layer_norm / conv1d were the gap that made the fused audit fail on
# every HF checkpoint: run_decompositions dissolves them before the analysis
# audit ever sees them, so they had never been certified. Each is certified the
# same way as the rest — a probe that exports and compiles to a CPU vmfb
# (scripts/certify_ops.py, fused profile).
#
# dropout is identity in eval and the exporter only ever sees eval models, so
# it costs nothing downstream; layer_norm and conv1d are real work, and their
# presence here means "lowers cleanly", not "the target has a kernel for it".
FUSED_ATEN_OPS = frozenset(
    {
        "rms_norm",
        "scaled_dot_product_attention",
        "dropout",
        "layer_norm",
        "conv1d",
    }
)


def audit(mlir_text: str, profile: str = "analysis") -> dict:
    """Bucket ops in ``mlir_text`` and decide on-device fitness.

    Args:
        mlir_text: exported torch-dialect MLIR.
        profile: the export profile the IR came from. Anything other than
            ``"analysis"`` is a fused profile, where the whole-op forms in
            :data:`FUSED_ATEN_OPS` are expected rather than suspicious.

    Returns a report dict:
        supported / unknown  — {op: count}
        server_side          — {hint: count}
        ok                   — True when no server-side hits and no unknown ops
    """
    op_counts = _count_aten_ops(mlir_text)
    server_side = _scan_server_side(mlir_text)
    allowed = SUPPORTED_ATEN_OPS if profile == "analysis" else SUPPORTED_ATEN_OPS | FUSED_ATEN_OPS

    supported: dict[str, int] = {}
    unknown: dict[str, int] = {}
    for op, n in op_counts.items():
        (supported if op in allowed else unknown)[op] = n

    return {
        "supported": supported,
        "unknown": unknown,
        "server_side": server_side,
        "unique_ops": len(op_counts),
        "unsupported_count": len(unknown),
        "server_side_count": len(server_side),
        "ok": not server_side and not unknown,
    }
