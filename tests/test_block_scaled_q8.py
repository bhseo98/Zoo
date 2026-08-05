"""Tests for the target NPU block-scaled INT8 matmul kernel scaffold.

Two layers, matching the design (docs/kernel-design/01-guideline-and-plan.md):
  * eager (pure-torch reference) — runs without IREE, validates registration +
    shape contract + numerics.
  * export smoke — validates ``generate()`` splices the portable standard-linalg
    microkernel (no ``iree_linalg_ext``). Requires venv-shark (iree.turbine).
"""
import pytest
import torch

from torch_mlir_zoo.kernels import (
    BlockScaledQ8Linear,
    quantize_block_scaled_q8,
    quantized_matmul,
)


def _ref_dequant_matmul(a, d, qs):
    n, g, bs = qs.shape
    w = (qs.to(a.dtype) * d).reshape(n, g * bs)
    return a @ w.transpose(-1, -2)


def _rel_err(out, ref):
    return (out - ref).norm() / ref.norm().clamp(min=1e-12)


def test_eager_matches_dequant_matmul():
    """Registered op (eager) == manual dequant+matmul, exactly."""
    torch.manual_seed(0)
    B, M, K, N, BS = 1, 4, 64, 8, 32
    a = torch.randn(B, M, K)
    qs, d = quantize_block_scaled_q8(torch.randn(N, K), BS)

    out = quantized_matmul(a, d, qs)
    assert out.shape == (B, M, N)
    torch.testing.assert_close(out, _ref_dequant_matmul(a, d, qs), rtol=0, atol=0)


def test_quantization_close_to_float_matmul():
    """Q8_0 kernel result tracks the fp32 matmul within quant error."""
    torch.manual_seed(0)
    B, M, K, N, BS = 2, 3, 128, 16, 32
    a = torch.randn(B, M, K)
    weight = torch.randn(N, K)
    qs, d = quantize_block_scaled_q8(weight, BS)

    out = quantized_matmul(a, d, qs)
    ref_float = a @ weight.transpose(-1, -2)
    assert _rel_err(out, ref_float) < 0.05


def test_block_scaled_q8_linear_module():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 8, bias=False)
    x = torch.randn(1, 5, 64)

    q = BlockScaledQ8Linear.from_linear(lin)
    out = q(x)
    assert out.shape == (1, 5, 8)
    assert _rel_err(out, lin(x)) < 0.05


def test_quantize_linears_swaps_and_matches():
    """quantize_linears_ recursively swaps Linears (bias preserved) and the INT8
    model tracks the fp32 model within quant error."""
    from torch_mlir_zoo.kernels import BlockScaledQ8Linear, quantize_linears_

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 96, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 32, bias=False),
    )
    x = torch.randn(2, 5, 64)  # arbitrary leading dims
    ref = model(x)

    quantize_linears_(model)
    assert not any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert any(isinstance(m, BlockScaledQ8Linear) for m in model.modules())

    out = model(x)
    assert out.shape == ref.shape
    assert _rel_err(out, ref) < 0.05  # bias kept, quant error small


def test_shape_contract_rejects_bad_qs():
    """K mismatch (group0*BS != K) must fail, not silently miscompute."""
    a = torch.randn(1, 2, 64)
    bad_qs = torch.zeros(8, 2, 16, dtype=torch.int8)  # 2*16=32 != 64
    bad_d = torch.ones(8, 2, 1)
    with pytest.raises(Exception):
        quantized_matmul(a, bad_d, bad_qs)


def test_export_smoke_portable_microkernel():
    """generate() splices the standard-linalg microkernel; no iree_linalg_ext."""
    pytest.importorskip("iree.turbine")
    from torch_mlir_zoo.analysis import summarize
    from torch_mlir_zoo.exporters import export_via_iree_turbine

    B, M, K, N, BS = 1, 4, 64, 8, 32
    qs, d = quantize_block_scaled_q8(torch.randn(N, K), BS)

    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("d", d)
            self.register_buffer("qs", qs)

        def forward(self, a):
            return quantized_matmul(a, self.d, self.qs)

    mlir = export_via_iree_turbine(Mod(), (torch.randn(B, M, K),))

    assert "npu_mmt_block_scaled_q8" in mlir  # microkernel spliced
    assert "iree_linalg_ext" not in mlir  # portable: standard linalg only
    assert summarize(mlir)["server_side_op_hits"] == {}


def test_iree_cpu_execution_matches_reference():
    """M1 gate: the microkernel actually COMPILES + RUNS on IREE-CPU and its
    output matches the torch reference (not just string-matching the IR)."""
    pytest.importorskip("iree.runtime")
    import numpy as np
    from iree.turbine.aot import FxProgramsBuilder, export
    import iree.runtime as rt

    torch.manual_seed(0)
    B, M, K, N, BS = 1, 8, 128, 32, 32
    weight = torch.randn(N, K)
    qs, d = quantize_block_scaled_q8(weight, BS)
    a = torch.randn(B, M, K)

    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("d", d)
            self.register_buffer("qs", qs)

        def forward(self, x):
            return quantized_matmul(x, self.d, self.qs)

    fxb = FxProgramsBuilder(Mod())

    @fxb.export_program(name="forward", args=(a,), strict=False)
    def _entry(model, *args):
        return model(*args)

    vmfb = bytes(
        export(fxb).compile(save_to=None, target_backends=("llvm-cpu",)).map_memory()
    )

    config = rt.Config("local-task")
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(rt.VmModule.copy_buffer(config.vm_instance, vmfb))
    out_iree = np.asarray(ctx.modules.module.forward(a.numpy())).astype(np.float32)

    out_eager = quantized_matmul(a, d, qs).numpy()
    out_float = (a @ weight.transpose(-1, -2)).numpy()

    assert out_iree.shape == (B, M, N)
    assert np.abs(out_iree - out_eager).max() < 1e-4  # matches torch reference
    assert (
        np.linalg.norm(out_iree - out_float) / np.linalg.norm(out_float) < 0.05
    )  # tracks fp32 within quant error
