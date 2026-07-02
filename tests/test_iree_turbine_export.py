"""iree.turbine export tests — sibling of test_zoo_export.py.

Verifies that the additive `iree_turbine` exporter produces legal MLIR text
for the four unit ops and TINY LlamaOnDevice, without any HF download. Skipped
when iree.turbine isn't installed (most local dev boxes — install via
`pip install -e .[shark]` inside venv-shark).
"""
import json
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip("iree.turbine")

from npu_harness_framework import Pipeline, build  # noqa: E402
import torch_mlir_zoo  # noqa: E402, F401  (side-effect: registers stages)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "zoo"
OPS = ["attention", "rmsnorm", "mlp", "topk"]


def _run_config(yaml_name: str, tmp_path: Path):
    cfg = yaml.safe_load((CONFIG_DIR / yaml_name).read_text())
    for stage in cfg["stages"]:
        if "out_path" in stage["config"]:
            stage["config"]["out_path"] = str(tmp_path / Path(stage["config"]["out_path"]).name)
    stages = [(s["name"], build(s["stage"], s["config"])) for s in cfg["stages"]]
    log_path = tmp_path / "profile.jsonl"
    return Pipeline(stages, log_path=str(log_path)).run(None), tmp_path, log_path


@pytest.mark.parametrize("op_name", OPS)
def test_iree_turbine_emits_legal_mlir(op_name, tmp_path):
    summary, out_dir, _ = _run_config(f"{op_name}_iree_turbine.yaml", tmp_path)
    mlir_file = next(out_dir.glob("*.mlir"))
    text = mlir_file.read_text()
    assert "func.func" in text or "module" in text
    # iree.turbine 의 IR 가 매우 압축적 (topk = ~10 lines). 8 줄 이상이면 trivial 모듈 이상.
    assert len(text.splitlines()) > 8
    assert summary["n_lines"] == len(text.splitlines())
    assert summary["op_counts"], "expected at least one op in IR"
    assert summary["server_side_op_hits"] == {}, "on-device op should not emit server-side hints"


def test_iree_turbine_profiler_records_three_lines(tmp_path):
    _, _, log_path = _run_config("rmsnorm_iree_turbine.yaml", tmp_path)
    lines = log_path.read_text().splitlines()
    assert len(lines) == 3
    stage_names = [json.loads(line)["stage"] for line in lines]
    assert stage_names == ["load", "export", "analyze"]


def test_iree_turbine_tiny_llama_export(tmp_path):
    """TINY LlamaOnDevice → iree.turbine MLIR (no HF download, no config file)."""
    from torch_mlir_zoo.models import LlamaConfig, LlamaOnDevice

    TINY = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=512,
        max_position_embeddings=64,
    )
    model = LlamaOnDevice(TINY).eval()
    input_ids = torch.randint(0, TINY.vocab_size, (1, 16))

    class _DirectModel:
        def __call__(self, _ignored):
            return (model, (input_ids,))

    exporter = build("exporter", {"type": "iree_turbine", "out_path": str(tmp_path / "tiny.mlir")})
    analyzer = build("analyzer", {"type": "ir_summary", "out_path": str(tmp_path / "tiny.summary.json")})

    p = Pipeline(
        [("load_model", _DirectModel()), ("export", exporter), ("analyze", analyzer)],
        log_path=str(tmp_path / "profile.jsonl"),
    )
    summary = p.run(None)
    assert summary["op_counts"], "expected at least one op in TINY Llama IR"
    assert summary["server_side_op_hits"] == {}
    assert (tmp_path / "tiny.mlir").exists()
