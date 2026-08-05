"""Top-level torch-dialect MLIR export — the single toolchain dependency.

Lowering depth is intentionally `OutputType.TORCH` (top-level torch
dialect, not Linalg / Tosa / Stablehlo). Step 5: "모델의 MLIR 결과를
IR dumping을 통해 분석" — analysis happens on this representation, not on
the deeper backend-specific dialects.
"""
from __future__ import annotations

from typing import Any


def export_top_level_torch_dialect(module: Any, example_args: tuple) -> str:
    """PyTorch nn.Module → top-level torch dialect MLIR text.

    Args:
        module: torch.nn.Module in eval mode.
        example_args: tuple of example positional inputs that fix the
            traced shape. Dynamic shapes are intentionally not used —
            on-device targets prefer fixed shapes.

    Returns:
        MLIR text as a single string (newline-separated).

    Raises:
        ModuleNotFoundError: if `torch_mlir` is not installed. Callers
            running unit checks without the toolchain should skip via
            `pytest.importorskip("torch_mlir")`.
    """
    import torch_mlir

    compiled = torch_mlir.compile(
        module,
        example_args,
        output_type=torch_mlir.OutputType.TORCH,
        use_tracing=False,
    )
    return str(compiled)
