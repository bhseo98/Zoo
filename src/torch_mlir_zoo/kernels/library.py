"""The single torch library every target NPU CustomOp registers into.

``def_library`` may only be called once per name, so the library object and the
Jinja microkernel loader live here and every kernel module imports them. Each
kernel keeps its **own** ``MICROKERNEL_BACKEND`` constant: the emission target is
a per-kernel decision (a hand-written target NPU intrinsic can land for one kernel
while another still uses the portable path).
"""
from __future__ import annotations

from iree.turbine.runtime.op_reg import def_library
from iree.turbine.runtime.op_reg.impl_helper import JinjaTemplateLoader

LIBRARY = def_library("npu")
TEMPLATES = JinjaTemplateLoader("torch_mlir_zoo.kernels")

__all__ = ["LIBRARY", "TEMPLATES"]
