"""Fused RMSNorm — a CustomOp because the standard path will not stay fused.

Measured on this stack: a hand-written RMSNorm *and* torch's own
``F.rms_norm`` both come out of ``iree.turbine`` export as five separate ops
(pow / mean / add / rsqrt / mul). The target NPU join contract wants the norm as one
op, and no amount of model rewriting keeps it whole — that is precisely the case
a CustomOp exists for (an op with no standard representation that survives).

Same two halves as :mod:`~torch_mlir_zoo.kernels.block_scaled_q8`:

  * ``select`` — the shape/dtype contract handed to the runtime integrator, stable
    across microkernel swaps.
  * ``generate`` — emits the portable standard-linalg microkernel today; the
    target NPU intrinsic replaces it by flipping ``MICROKERNEL_BACKEND``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from iree.turbine.runtime.op_reg import CustomOp, KernelBuilder, KernelSelection
from iree.turbine.runtime.op_reg.impl_helper import call_function
from iree.turbine.support.ir_imports import RankedTensorType

from .library import LIBRARY, TEMPLATES

# Emission target — the single swap point of this kernel. ``generate`` emits
# ``templates/fused_rmsnorm_{MICROKERNEL_BACKEND}.mlir``. See
# ``docs/kernel-design/02-lowering-layer.md`` §4.
MICROKERNEL_BACKEND = "standard"

__all__ = ["FusedRMSNorm", "fused_rmsnorm"]


@CustomOp.register(library=LIBRARY)
class fused_rmsnorm(CustomOp):  # noqa: N801 — name becomes the op callable
    """RMS normalization kept as one op::

        x:   [B, M, D]   float
        w:   [D]         float   (learned scale)
        eps: float                (specialized into the microkernel)
        out: [B, M, D]   = x * rsqrt(mean(x^2, dim=-1) + eps) * w

    Specialized on D, eps and the element type; B and M stay dynamic.
    """

    signature = "fused_rmsnorm(Tensor x, Tensor w, float eps) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        x_desc = ksel.arg_tensor(0)
        w_desc = ksel.arg_tensor(1)
        ksel.attr_float(2)

        *batch_dims, x_m, x_d = x_desc.t.shape
        torch._check(
            x_desc.t.dtype.is_floating_point,
            lambda: f"fused_rmsnorm arg 'x': expected floating point (got {x_desc.t.dtype})",
        )
        torch._check(
            len(batch_dims) == 1,
            lambda: f"fused_rmsnorm arg 'x': expected 3d [B,M,D] (got {tuple(x_desc.t.shape)})",
        )
        torch._check(
            tuple(w_desc.t.shape) == (x_d,) and w_desc.t.dtype == x_desc.t.dtype,
            lambda: (
                f"fused_rmsnorm arg 'w': expected [{x_d}] of {x_desc.t.dtype} "
                f"(got {tuple(w_desc.t.shape)} of {w_desc.t.dtype})"
            ),
        )

        # Specialize the normalized dim (and all of w); keep B, M dynamic.
        x_desc.specialize_dims(-1)
        w_desc.specialize_all_dims()

        out_desc = ksel.return_new_tensor(
            list(batch_dims) + [x_m, x_d], dtype=x_desc.t.dtype
        )
        out_desc.specialize_dims(-1)

    def eager_execute(self, x, w, eps):
        # Pure-PyTorch reference (backend-independent). Runs today, no IREE.
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * scale * w

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        x_type = RankedTensorType(kb.arg_value(0).type)
        d = x_type.get_dim_size(x_type.rank - 1)
        elem_type = str(x_type.element_type)
        eps = ksel.arg_descs[2].v

        fn_name = f"npu_fused_rmsnorm_{d}_{elem_type}"
        template_name = f"fused_rmsnorm_{MICROKERNEL_BACKEND}"  # swap point
        func_op = TEMPLATES.inline_template_function(
            kb,
            template_name,
            fn_name,
            d=d,
            elem_type=elem_type,
            d_literal=f"{float(d):e}",
            eps_literal=f"{float(eps):e}",
        )
        # arg 2 is an attribute, not a value — only the tensors are bound.
        kb.yield_results(*call_function(func_op, *kb.arg_bindings[0:2]))


class FusedRMSNorm(nn.Module):
    """Drop-in RMSNorm whose IR is one ``npu.fused_rmsnorm`` call.

    Weight layout and semantics match ``torch_mlir_zoo.ops.RMSNorm``, so the two
    are swappable: use the zoo op to see the decomposed aten form, this one to
    hold the fused join contract.
    """

    def __init__(self, dim: int = 512, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)
