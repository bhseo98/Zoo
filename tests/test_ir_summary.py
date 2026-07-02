"""Unit tests for `analysis.ir_summary.summarize` — no toolchain dependency.

The `server_side_op_hits == {}` invariant is the zoo's central on-device
claim, asserted across export tests. Those tests need torch-mlir and skip on
most hosts, so the *detector itself* was never exercised negatively. These
positive/negative controls verify it actually fires (and stays quiet) on
known MLIR text, with no torch-mlir / iree.turbine import.
"""
from torch_mlir_zoo.analysis.ir_summary import summarize

_CLEAN = """\
module attributes {torch.debug_module_name = "OnDevice"} {
  func.func @forward(%arg0: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[1,4],f32> {
    %0 = torch.aten.matmul %arg0, %arg0 : !torch.vtensor<[1,4],f32>
    %1 = torch.aten.softmax %0 : !torch.vtensor<[1,4],f32>
    return %1 : !torch.vtensor<[1,4],f32>
  }
}
"""

# Same shape, but carrying server-side tokens the detector must catch.
_SERVER = _CLEAN.replace(
    "%1 = torch.aten.softmax %0",
    "%1 = torch.aten.paged_attention %0  // kv_cache + tensor_parallel",
)


def test_clean_ir_has_no_server_side_hits():
    s = summarize(_CLEAN)
    assert s["server_side_op_hits"] == {}, "on-device IR must show zero hits"


def test_server_side_tokens_are_detected():
    hits = summarize(_SERVER)["server_side_op_hits"]
    # positive control: detector must fire on the injected tokens
    assert hits.get("paged_attention", 0) >= 1
    assert hits.get("kv_cache", 0) >= 1
    assert hits.get("tensor_parallel", 0) >= 1


def test_op_counts_and_dtypes_parse():
    s = summarize(_CLEAN)
    assert s["op_counts"].get("matmul") == 1
    assert s["op_counts"].get("softmax") == 1
    assert s["dtypes"].get("f32", 0) >= 1
    assert s["module_name"] == "OnDevice"
    assert s["has_dynamic_dim"] is False
