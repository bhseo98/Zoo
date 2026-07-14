"""Han-Neuro SDK facade — real export smoke test + input validation."""
import pytest
import torch

from torch_mlir_zoo import ExportResult, export_for_npu
from torch_mlir_zoo.ops import RMSNorm


def _args():
    return (torch.randn(1, 8, 512),)


def test_bad_backend_rejected():
    with pytest.raises(ValueError):
        export_for_npu(RMSNorm(), _args(), backend="nope")


def test_bad_quantize_rejected():
    with pytest.raises(ValueError):
        export_for_npu(RMSNorm(), _args(), quantize="fp4")


def test_turbine_export_rmsnorm_ok(tmp_path):
    pytest.importorskip("iree.turbine")
    r = export_for_npu(RMSNorm(), _args(), backend="iree_turbine")
    assert isinstance(r, ExportResult)
    assert r.backend == "iree_turbine"
    assert "module" in r.mlir
    assert r.summary["server_side_op_hits"] == {}
    assert r.ok is True
    out = r.save(tmp_path / "sub" / "rmsnorm.mlir")
    assert out.read_text() == r.mlir


def test_quantize_int8_does_not_mutate_caller():
    m = torch.nn.Linear(64, 64)
    before = m.weight.detach().clone()
    try:
        export_for_npu(m, (torch.randn(1, 64),), backend="iree_turbine", quantize="int8")
    except ModuleNotFoundError:
        pytest.skip("iree.turbine not installed")
    assert torch.equal(m.weight.detach(), before)  # deepcopy protected caller
