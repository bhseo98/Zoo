"""Fused RMSNorm CustomOp — the contract kernel that keeps the norm as one op.

Two layers, matching test_block_scaled_q8.py:
  * eager — registration + shape contract + numeric parity with the zoo op.
  * export smoke — ``generate()`` splices the portable standard-linalg
    microkernel and the aten decomposition is gone. Requires venv-shark.
"""
import pytest
import torch

from torch_mlir_zoo.kernels import FusedRMSNorm, fused_rmsnorm
from torch_mlir_zoo.ops import RMSNorm


def test_eager_matches_the_zoo_rmsnorm():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 64)
    zoo = RMSNorm(dim=64, eps=1e-6)
    with torch.no_grad():
        zoo.weight.copy_(torch.rand(64) + 0.5)

    out = fused_rmsnorm(x, zoo.weight, zoo.eps)
    ref = zoo(x)
    assert torch.allclose(out, ref, atol=1e-6)


def test_module_wraps_the_op():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32)
    module = FusedRMSNorm(dim=32, eps=1e-6)
    assert torch.allclose(module(x), RMSNorm(dim=32, eps=1e-6)(x), atol=1e-6)


def test_contract_rejects_bad_shapes_at_export_time():
    # `select` — the contract we hand the runtime integrator — runs during tracing,
    # not in eager mode, so the shape rules are checked at export.
    pytest.importorskip("iree.turbine")
    from torch_mlir_zoo import export_for_npu

    class _WrongWeight(torch.nn.Module):
        def forward(self, x):
            return fused_rmsnorm(x, torch.ones(16), 1e-6)

    class _TwoDim(torch.nn.Module):
        def forward(self, x):
            return fused_rmsnorm(x, torch.ones(32), 1e-6)

    with pytest.raises(Exception):
        export_for_npu(_WrongWeight(), (torch.randn(1, 4, 32),), backend="iree_turbine")
    with pytest.raises(Exception):
        export_for_npu(_TwoDim(), (torch.randn(4, 32),), backend="iree_turbine")


def test_export_keeps_one_fused_op_instead_of_the_decomposition():
    pytest.importorskip("iree.turbine")
    from torch_mlir_zoo import export_for_npu

    args = (torch.randn(1, 8, 64),)
    fused = export_for_npu(FusedRMSNorm(dim=64), args, backend="iree_turbine")
    plain = export_for_npu(RMSNorm(dim=64), args, backend="iree_turbine")

    # The zoo op decomposes; the CustomOp does not.
    assert {"pow", "mean", "rsqrt"} <= set(plain.summary["op_counts"])
    assert "pow" not in fused.summary["op_counts"]
    assert "rsqrt" not in fused.summary["op_counts"]

    # ...and what replaced it is our microkernel, portable (no IREE-only dialect).
    assert "npu_fused_rmsnorm" in fused.mlir
    assert "iree_linalg_ext" not in fused.mlir
    assert fused.ok is True
