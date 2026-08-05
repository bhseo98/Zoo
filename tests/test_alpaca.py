"""Alpaca SDK facade — real export smoke test + input validation."""
import pytest
import torch

from torch_mlir_zoo import ExportResult, export_for_npu
from torch_mlir_zoo.ops import RMSNorm


def _args():
    return (torch.randn(1, 8, 512),)


def test_default_backend_actually_runs():
    # Regression: the default used to be torch_mlir, whose wheel index is dead,
    # so the documented one-liner raised ModuleNotFoundError on every call.
    pytest.importorskip("iree.turbine")
    r = export_for_npu(RMSNorm(), _args())
    assert r.backend == "iree_turbine"
    assert r.ok is True


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


class _TinyMLP(torch.nn.Module):
    """Two child Linears — the realistic case (Linear wrapped in a parent)."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_quantize_int8_applies_and_does_not_mutate_caller():
    m = _TinyMLP()
    before = m.fc1.weight.detach().clone()
    try:
        r = export_for_npu(
            m, (torch.randn(1, 64),), backend="iree_turbine", quantize="int8"
        )
    except ModuleNotFoundError:
        pytest.skip("iree.turbine not installed")
    assert torch.equal(m.fc1.weight.detach(), before)  # deepcopy protected caller
    assert "block_scaled_q8" in r.mlir  # int8 kernel actually emitted (not a no-op)
    assert r.ok is True


def test_quantize_int8_handles_bare_top_level_linear():
    # A bare top-level nn.Linear has no children to walk; quantize_linears_ must
    # still replace it (regression lock for the named_children()-only bug).
    m = torch.nn.Linear(64, 64)
    before = m.weight.detach().clone()
    try:
        r = export_for_npu(
            m, (torch.randn(1, 64),), backend="iree_turbine", quantize="int8"
        )
    except ModuleNotFoundError:
        pytest.skip("iree.turbine not installed")
    assert torch.equal(m.weight.detach(), before)  # caller untouched
    assert "block_scaled_q8" in r.mlir  # root Linear was quantized


def test_verify_reports_quantization_accuracy():
    # verify=True attaches an accuracy report (fp vs int8), the safety net.
    m = _TinyMLP()
    try:
        r = export_for_npu(
            m, (torch.randn(4, 64),),
            backend="iree_turbine", quantize="int8", verify=True,
        )
    except ModuleNotFoundError:
        pytest.skip("iree.turbine not installed")
    assert r.accuracy is not None
    assert 0.0 <= r.accuracy["max_rel"] < 0.3       # block-Q8 output stays close
    assert r.accuracy["cosine"] > 0.98
    assert len(r.accuracy["layers"]) == 2           # fc1, fc2 both quantized
    assert all(layer["mean_rel"] < 0.05 for layer in r.accuracy["layers"])


def test_verify_off_by_default():
    # default path carries no accuracy overhead
    try:
        r = export_for_npu(_TinyMLP(), (torch.randn(1, 64),), backend="iree_turbine", quantize="int8")
    except ModuleNotFoundError:
        pytest.skip("iree.turbine not installed")
    assert r.accuracy is None
