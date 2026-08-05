"""Op-support audit gate — allowlist / unknown / server-side bucketing."""
from torch_mlir_zoo.analysis import audit


def test_clean_rmsnorm_ir_passes():
    ir = "%0 = torch.aten.pow %a\n%1 = torch.aten.mean %0\n%2 = torch.aten.rsqrt %1\n%3 = torch.aten.mul %2"
    rep = audit(ir)
    assert rep["ok"] is True
    assert rep["unknown"] == {}
    assert rep["server_side"] == {}
    assert set(rep["supported"]) == {"pow", "mean", "rsqrt", "mul"}


def test_unknown_op_fails():
    ir = "%0 = torch.aten.mm %a\n%1 = torch.aten.some_exotic_op %0"
    rep = audit(ir)
    assert rep["ok"] is False
    assert rep["unknown"] == {"some_exotic_op": 1}
    assert rep["unsupported_count"] == 1


def test_server_side_leak_fails():
    ir = "func private @paged_attention()\n%0 = torch.aten.mm %a"
    rep = audit(ir)
    assert rep["ok"] is False
    assert "paged_attention" in rep["server_side"]
