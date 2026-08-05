"""NPU INT8 block-scaled matmul — CustomOp scaffold.

Kernel-design skeleton for the target NPU path (see
``docs/kernel-design/01-guideline-and-plan.md``). Mirrors amdsharktank's
``mmt_block_scaled_q8`` but:

  * uses iree.turbine's upstream ``op_reg`` + ``JinjaTemplateLoader`` (no
    amdsharktank dependency), and
  * emits a *portable* standard-linalg microkernel (no ``iree_linalg_ext``) that
    lowers through IREE for CPU today. The target-specific intrinsic is a
    documented swap-in point in :meth:`generate` / the ``.mlir`` template.

A CustomOp has two halves:

  * Python (``signature`` / ``select`` / ``generate``) = **compile-time
    metacode** — it does no numeric work, only decides *which* MLIR to emit and
    with what shapes.
  * the ``.mlir`` microkernel = the **actual computation** IREE compiles.

``select`` (the shape/dtype contract) is the stable interface we hand to the
runtime integrator; only ``generate``'s emission target changes when target NPU IR lands.
"""
from __future__ import annotations

import torch

from iree.turbine.runtime.op_reg import CustomOp, KernelBuilder, KernelSelection
from iree.turbine.runtime.op_reg.impl_helper import call_function
from iree.turbine.support.ir_imports import RankedTensorType

from .library import LIBRARY, TEMPLATES as _templates

# Microkernel backend — the single swap point of the lowering layer. ``generate``
# emits ``templates/block_scaled_q8_{MICROKERNEL_BACKEND}.mlir``. "standard" is the
# portable standard-linalg microkernel that lowers on IREE-CPU today; swap to
# "npu" (and add ``block_scaled_q8_npu.mlir``) once the target NPU library/runtime
# IR is handed over. ``select`` (the shape contract), ``eager_execute`` and the
# model classes stay unchanged across the swap. See
# ``docs/kernel-design/02-lowering-layer.md`` §4.
MICROKERNEL_BACKEND = "standard"

__all__ = ["block_scaled_q8"]


@CustomOp.register(library=LIBRARY)
class block_scaled_q8(CustomOp):  # noqa: N801 — name becomes the op callable
    """INT8 block-scaled matmul with transposed RHS (GGUF Q8_0-style).

    Planar block-scaled layout::

        a:   [B, M, K]          float   (activation / LHS)
        d:   [N, K // BS, 1]    float   (per-block scale)
        qs:  [N, K // BS, BS]   int8    (quantized weight)
        out: [B, M, N]          = a @ dequant(qs, d)^T

    Specialized on N, K, BS and the LHS element type; B and M stay dynamic.
    """

    signature = "block_scaled_q8(Tensor a, Tensor d, Tensor qs) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        a_desc = ksel.arg_tensor(0)
        d_desc = ksel.arg_tensor(1)
        qs_desc = ksel.arg_tensor(2)

        *batch_dims, a_m, a_k = a_desc.t.shape
        torch._check(
            a_desc.t.dtype.is_floating_point,
            lambda: f"block_scaled_q8 arg 'a': expected floating point (got {a_desc.t.dtype})",
        )
        torch._check(
            len(batch_dims) == 1,
            lambda: f"block_scaled_q8 arg 'a': expected 3d [B,M,K] (got {tuple(a_desc.t.shape)})",
        )

        qs_n, qs_group0, qs_bs, *rest = qs_desc.t.shape
        torch._check(
            len(rest) == 0 and (qs_group0 * qs_bs) == a_k,
            lambda: f"block_scaled_q8 arg 'qs': expected [N,K//BS,BS] with K=={a_k} (got {tuple(qs_desc.t.shape)})",
        )

        d_n, d_group0, d_one, *rest = d_desc.t.shape
        torch._check(
            len(rest) == 0 and (d_group0 * qs_bs) == a_k and d_one == 1 and d_n == qs_n,
            lambda: f"block_scaled_q8 arg 'd': expected [N,K//BS,1] matching qs (got {tuple(d_desc.t.shape)})",
        )

        # Specialize K (LHS) and all of qs/d; keep B, M dynamic.
        a_desc.specialize_dims(-1)
        qs_desc.specialize_all_dims()
        d_desc.specialize_all_dims()

        c_desc = ksel.return_new_tensor(
            list(batch_dims) + [a_m, d_n], dtype=a_desc.t.dtype
        )
        c_desc.specialize_dims(-1)

    def eager_execute(self, a, d, qs):
        # Pure-PyTorch reference (backend-independent). Runs today, no IREE.
        n, group0, bs = qs.shape
        w = (qs.to(a.dtype) * d).reshape(n, group0 * bs)  # dequant -> [N, K]
        return torch.matmul(a, w.transpose(-1, -2))  # [B, M, N]

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        a_type = RankedTensorType(kb.arg_value(0).type)
        d_type = RankedTensorType(kb.arg_value(1).type)
        qs_type = RankedTensorType(kb.arg_value(2).type)

        k = a_type.get_dim_size(a_type.rank - 1)
        n, group0, bs = qs_type.shape
        a_dtype = str(a_type.element_type)
        scale_dtype = str(d_type.element_type)

        fn_name = f"npu_mmt_block_scaled_q8_{n}_{k}_{bs}_{a_dtype}"
        template_name = f"block_scaled_q8_{MICROKERNEL_BACKEND}"  # swap point
        func_op = _templates.inline_template_function(
            kb,
            template_name,
            fn_name,
            n=n,
            k=k,
            bs=bs,
            group0=group0,
            a_type=a_dtype,
            scale_type=scale_dtype,
        )
        kb.yield_results(*call_function(func_op, *kb.arg_bindings))
