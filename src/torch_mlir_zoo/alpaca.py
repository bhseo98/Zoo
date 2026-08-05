"""Alpaca SDK — public compiler facade.

One entry point turns a PyTorch model into clean top-level torch-dialect MLIR
that an on-device NPU runtime compiler can consume:

    from torch_mlir_zoo import export_for_npu

    result = export_for_npu(model, example_args, quantize="int8")
    result.mlir       # torch-dialect MLIR text
    result.summary    # {"server_side_op_hits": {}, "op_counts": {...}, ...}
    result.ok         # True when no server-side ops leaked (on-device suitable)
    result.save("model.mlir")

This is a thin facade over the already-verified building blocks
(``exporters`` / ``analysis`` / ``kernels``); it adds no new lowering logic.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .capture import ExportError, diagnose, drop_context_markers, prepare

_BACKENDS = ("torch_mlir", "iree_turbine")

# Export profiles — what "good IR" means depends on who consumes it.
#
#   analysis     maximally decomposed: every op is a standard aten primitive, so
#                any backend can lower it and the op audit is meaningful.
#   fused        maximally fused: the torch-to-npu pattern matcher consumes
#                whole ops (aten.linear / rms_norm / scaled_dot_product_attention),
#                because a fused op is what maps onto one hardware kernel.
#
# They are opposites on purpose. Measured on one Llama decoder layer: the analysis
# profile emits 98 ops in 25 kinds, the fused profile 37 in 12.
_PROFILES = {
    "analysis": (False, True),
    "fused": (True, False),
}

# Capture strategies tried in order by the iree_turbine backend. Non-strict is
# the amdsharktank default; strict (dynamo) is the fallback for models whose
# forward defeats the non-strict tracer — measured on Qwen2.5 / Llama-family HF.
_CAPTURE_LADDER = ("nonstrict", "strict")


@dataclass
class ExportResult:
    """The IR, its analysis, and the join contract of one export."""

    mlir: str
    summary: dict
    backend: str
    accuracy: dict | None = None
    rewrites: list[str] = field(default_factory=list)
    capture: str | None = None
    profile: str = "analysis"

    @property
    def ok(self) -> bool:
        """True when the IR is on-device suitable — no server-side ops."""
        return self.summary.get("server_side_op_hits", {}) == {}

    def save(self, path: str | Path) -> Path:
        """Write the MLIR text to ``path`` and return the resolved path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.mlir)
        return p


def _accuracy(orig: Any, quant: Any, example_args: tuple) -> dict:
    """Compare fp vs quantized model on ``example_args``; report error metrics.

    Output-level (rel-error / cosine / argmax-match) plus per-Linear weight
    quantization error — the accuracy safety net for INT8 lowering.
    """
    import torch
    import torch.nn as nn

    from .kernels.quantized_linear import BlockScaledQ8Linear

    with torch.no_grad():
        fp = orig.eval()(*example_args)
        qz = quant.eval()(*example_args)
    ff, qf = fp.float(), qz.float()
    diff = (ff - qf).abs()
    scale = ff.abs().max().clamp_min(1e-8)
    report: dict = {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff.max() / scale).item(),
        "cosine": torch.nn.functional.cosine_similarity(ff.flatten(), qf.flatten(), dim=0).item(),
    }
    if ff.shape[-1] > 1:
        report["argmax_match"] = (ff.argmax(-1) == qf.argmax(-1)).float().mean().item()

    quant_mods = dict(quant.named_modules())
    layers = []
    for name, m in orig.named_modules():
        qm = quant_mods.get(name)
        if isinstance(m, nn.Linear) and isinstance(qm, BlockScaledQ8Linear):
            wq = (qm.qs.to(torch.float32) * qm.d).reshape(m.weight.shape)
            w = m.weight.detach().float()
            err = (wq - w).abs()
            wscale = w.abs().max().clamp_min(1e-8)
            layers.append(
                {
                    "name": name or "<root>",
                    "max_rel": (err.max() / wscale).item(),
                    "mean_rel": (err.mean() / wscale).item(),
                }
            )
    report["layers"] = layers
    return report


def _decomposition_ctx(run_decompositions: bool):
    """Turbine's decomposition table, or an empty one.

    An empty table makes turbine skip ``run_decompositions`` entirely, so the
    pre-dispatch aten ops (``linear`` / ``rms_norm`` / ``scaled_dot_product_
    attention``) reach the importer whole instead of being broken into
    mm / pow-mean-rsqrt / bmm-softmax.
    """
    if run_decompositions:
        import contextlib

        return contextlib.nullcontext()

    from iree.turbine.aot import extend_aot_decompositions

    return extend_aot_decompositions(from_current=False)


def _run_export(
    model: Any, example_args: tuple, backend: str, run_decompositions: bool = True
) -> tuple[str, str | None]:
    """Export through ``backend``; walk the capture ladder and diagnose failures.

    Returns ``(mlir_text, capture_strategy)`` — the strategy is ``None`` for the
    torch_mlir backend, which has a single capture path.
    """
    if backend == "torch_mlir":
        from .exporters import export_top_level_torch_dialect

        try:
            return export_top_level_torch_dialect(model, example_args), None
        except ModuleNotFoundError:
            raise  # missing toolchain, not a model problem
        except Exception as e:
            raise ExportError(diagnose(e), e) from e

    from .exporters import export_via_iree_turbine

    failure: BaseException | None = None
    with drop_context_markers(), _decomposition_ctx(run_decompositions):
        for capture in _CAPTURE_LADDER:
            try:
                mlir = export_via_iree_turbine(
                    model, example_args, strict=(capture == "strict")
                )
                return mlir, capture
            except ModuleNotFoundError:
                raise
            except Exception as e:
                failure = e
                if diagnose(e).retry != "strict":
                    break
    raise ExportError(diagnose(failure), failure) from failure


def export_for_npu(
    model: Any,
    example_args: tuple,
    *,
    backend: str = "iree_turbine",
    quantize: str | None = None,
    block_size: int | None = 32,
    quantize_embeddings: bool = False,
    verify: bool = False,
    rewrite: bool = True,
    arg_names: tuple[str, ...] | None = None,
    profile: str = "analysis",
) -> ExportResult:
    """Compile a PyTorch model to top-level torch-dialect MLIR.

    Args:
        model: a ``torch.nn.Module``.
        example_args: tuple of example positional inputs that fix the traced
            shape (dynamic shapes are intentionally not used on-device).
        backend: ``"iree_turbine"`` (default) or ``"torch_mlir"``. Both wrap the
            same fx_importer; what a fused ``aten.linear`` depends on is
            ``profile``, not the backend. ``torch_mlir`` is kept for
            completeness but is currently uninstallable — its published wheel
            index stopped at 2024-01 and pins a torch nightly that no longer
            resolves, so every call raises ``ModuleNotFoundError``.
        quantize: ``None`` or ``"int8"`` (``block_scaled_q8``, applied last —
            only after lowering is correct).
        block_size: INT8 block size when ``quantize="int8"``; ``None`` means
            per-channel (one scale per output row).
        quantize_embeddings: also quantize ``nn.Embedding`` tables, sharing the
            buffers with any tied ``lm_head``. Off by default because it changes
            the numerics of the largest tensor in the model — turn it on when
            footprint is the binding constraint.
        verify: when ``True`` and ``quantize="int8"``, run the fp and quantized
            model on ``example_args`` and attach an accuracy report
            (:attr:`ExportResult.accuracy`) — output rel-error / cosine /
            argmax-match plus per-layer weight quantization error.
        rewrite: apply the capture rewrites (eager attention, forward-only
            wrapper, eval mode) before tracing and report them in
            :attr:`ExportResult.rewrites`. Switching attention to eager edits the
            model's config in place — same math, traceable form — so pass
            ``False`` to export a model exactly as handed in.
        arg_names: bind ``example_args`` to these keyword names when the model's
            leading positionals are not its inputs (Whisper takes
            ``(input_features, attention_mask, decoder_input_ids)``). Positional
            binding otherwise.
        profile: ``"analysis"`` — decomposed IR for the op audit and IREE-CPU
            execution — or ``"fused"``, which keeps ops fused for the
            downstream pattern matcher (see :data:`_PROFILES`).

    Returns:
        An :class:`ExportResult` (``.mlir`` / ``.summary`` / ``.ok`` / ``.save``).

    Raises:
        ValueError: on an unknown ``backend`` or ``quantize`` value.
        ModuleNotFoundError: if the chosen backend's toolchain is not installed
            (``torch_mlir`` for the join backend, ``iree.turbine`` otherwise).
        ExportError: when the model cannot be captured — carries a
            :class:`~torch_mlir_zoo.capture.Diagnosis` naming the failing layer
            (load / capture / lowering) and the fix.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {list(_BACKENDS)}")
    if quantize not in (None, "int8"):
        raise ValueError(f"quantize must be None or 'int8', got {quantize!r}")
    if profile not in _PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {list(_PROFILES)}")
    if profile == "fused" and quantize is not None:
        # Measured: quantizing replaces all 7 aten.linear of a decoder layer with
        # block_scaled_q8 calls, and a matcher looking for fused linear then finds
        # nothing. Quantization belongs downstream when the consumer does its own
        # weight/activation lowering. Drop this guard if that changes.
        raise ValueError(
            "profile='fused' must stay unquantized — a downstream pass that does "
            "its own w8a8 matches fused aten.linear. Use profile='analysis' for "
            "the INT8 path in this repo."
        )
    keep_fused, run_decompositions = _PROFILES[profile]

    rewrites: list[str] = []
    if rewrite:
        model, rewrites = prepare(model, arg_names=arg_names, keep_fused=keep_fused)

    accuracy = None
    if quantize == "int8":
        orig = model  # keep the fp reference for optional verification
        model = copy.deepcopy(orig)  # never mutate the caller's weights
        from .kernels import quantize_embeddings_, quantize_linears_

        # Linears first (they skip tied weights), then embeddings — which is
        # where the tied table gets handled, buffers shared with its lm_head.
        model = quantize_linears_(model, block_size=block_size)
        if quantize_embeddings:
            model = quantize_embeddings_(model, block_size=block_size)
        if verify:
            accuracy = _accuracy(orig, model, example_args)

    model = model.eval()
    mlir, capture = _run_export(model, example_args, backend, run_decompositions)

    from .analysis import summarize

    return ExportResult(
        mlir=mlir,
        summary=summarize(mlir),
        backend=backend,
        accuracy=accuracy,
        rewrites=rewrites,
        capture=capture,
        profile=profile,
    )


__all__ = ["export_for_npu", "ExportResult", "ExportError"]
