"""Torch-MLIR Model Zoo — PyTorch-only on-device building blocks toward
target NPU compiler stack (Step 3-5).

This package supplies the *capability* layer between framework core
(`npu_harness_framework`) and the target NPU/IREE compiler stack: it exposes
PyTorch-only model implementations (attention / RMSNorm / MLP / TopK /
Llama-3.2-1B on-device) that pass cleanly through `torch_mlir.compile`,
and the harness stages that wrap loader → exporter → analyzer into a
profiled `Pipeline`.

Layout:
    ops/        — unit ops (Attention, RMSNorm, SwiGLU, TopK)
    models/     — composed models (LlamaOnDevice)
    exporters/  — torch-mlir invocation (single toolchain dependency)
    analysis/   — MLIR text → JSON summary (op-count, dtype, server-side hints)
    stages.py   — BaseStage subclasses; importing this package registers them

Plugin registration is side-effect: `import torch_mlir_zoo` also triggers
`from . import stages` so the harness can `build(...)` them via YAML.
"""
from . import stages  # noqa: F401  (side-effect: registers stages)
from .han_neuro import ExportResult, export_for_npu

__version__ = "0.1.0"

__all__ = ["export_for_npu", "ExportResult"]
