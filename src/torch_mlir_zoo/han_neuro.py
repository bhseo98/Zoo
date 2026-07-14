"""Han-Neuro SDK — public compiler facade.

One entry point turns a PyTorch model into clean top-level torch-dialect MLIR
that an on-device NPU runtime compiler can consume:

    from torch_mlir_zoo import export_for_npu

    result = export_for_npu(model, example_args, backend="torch_mlir", quantize="int8")
    result.mlir       # torch-dialect MLIR text
    result.summary    # {"server_side_op_hits": {}, "op_counts": {...}, ...}
    result.ok         # True when no server-side ops leaked (on-device suitable)
    result.save("model.mlir")

This is a thin facade over the already-verified building blocks
(``exporters`` / ``analysis`` / ``kernels``); it adds no new lowering logic.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BACKENDS = ("torch_mlir", "iree_turbine")


@dataclass
class ExportResult:
    """The IR, its analysis, and the join contract of one export."""

    mlir: str
    summary: dict
    backend: str

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


def export_for_npu(
    model: Any,
    example_args: tuple,
    *,
    backend: str = "torch_mlir",
    quantize: str | None = None,
    block_size: int = 32,
) -> ExportResult:
    """Compile a PyTorch model to top-level torch-dialect MLIR.

    Args:
        model: a ``torch.nn.Module``.
        example_args: tuple of example positional inputs that fix the traced
            shape (dynamic shapes are intentionally not used on-device).
        backend: ``"torch_mlir"`` — the join backend, preserves fused
            ``aten.linear`` for the NPU runtime pass — or ``"iree_turbine"``,
            which decomposes linear into mm/bmm (analysis / IREE-native path).
        quantize: ``None`` or ``"int8"`` (``block_scaled_q8``, applied last —
            only after lowering is correct).
        block_size: INT8 block size when ``quantize="int8"``.

    Returns:
        An :class:`ExportResult` (``.mlir`` / ``.summary`` / ``.ok`` / ``.save``).

    Raises:
        ValueError: on an unknown ``backend`` or ``quantize`` value.
        ModuleNotFoundError: if the chosen backend's toolchain is not installed
            (``torch_mlir`` for the join backend, ``iree.turbine`` otherwise).
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {list(_BACKENDS)}")
    if quantize not in (None, "int8"):
        raise ValueError(f"quantize must be None or 'int8', got {quantize!r}")

    if quantize == "int8":
        model = copy.deepcopy(model)  # never mutate the caller's weights
        from .kernels import quantize_linears_

        quantize_linears_(model, block_size=block_size)

    model = model.eval()

    if backend == "torch_mlir":
        from .exporters import export_top_level_torch_dialect

        mlir = export_top_level_torch_dialect(model, example_args)
    else:
        from .exporters import export_via_iree_turbine

        mlir = export_via_iree_turbine(model, example_args)

    from .analysis import summarize

    return ExportResult(mlir=mlir, summary=summarize(mlir), backend=backend)


__all__ = ["export_for_npu", "ExportResult"]
