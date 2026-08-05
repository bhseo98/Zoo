"""IREE-Turbine export — sibling backend to `torch_mlir.compile`.

Same shape as `export_top_level_torch_dialect(module, args) -> str` so the rest
of the zoo (stages, analyzers, tests) only sees a backend swap. Backend uses
`iree.turbine.aot.FxProgramsBuilder` + `aot.export`, the path that AMD-SHARK-AI
uses internally. Reference: `amdsharktank/models/t5/export.py:50-89`,
`amdsharktank/utils/export.py:214-234`.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any


def export_via_iree_turbine(
    module: Any,
    example_args: tuple,
    *,
    func_name: str = "forward",
    strict: bool = False,
) -> str:
    """PyTorch nn.Module → MLIR text via iree.turbine.aot.

    Args:
        module: torch.nn.Module in eval mode.
        example_args: tuple of example positional inputs (fix the traced shape).
        func_name: entry-point function name in the produced MLIR.
        strict: passed to `@fxb.export_program(strict=...)`. False is the
            common amdsharktank setting for forward-only export.

    Returns:
        MLIR text (newline-separated).

    Raises:
        ModuleNotFoundError: if `iree.turbine` is not installed. Activate
            venv-shark or `pip install -e .[shark]`.
    """
    try:
        from iree.turbine.aot import FxProgramsBuilder, export
    except ImportError as e:
        raise ModuleNotFoundError(
            "iree.turbine not installed. Activate venv-shark or "
            "`pip install -e .[shark]` first."
        ) from e

    fxb = FxProgramsBuilder(module)

    @fxb.export_program(name=func_name, args=example_args, strict=strict)
    def _entry(model, *runtime_args):
        return model(*runtime_args)

    output = export(fxb)

    fd, tmp_path = tempfile.mkstemp(suffix=".mlir")
    os.close(fd)
    try:
        output.save_mlir(tmp_path)
        with open(tmp_path) as f:
            return f.read()
    finally:
        os.unlink(tmp_path)
