"""Torch-MLIR export + IR-summary tests — skipped without torch-mlir.

These verify that the Pipeline (loader → exporter → analyzer) actually
produces legal MLIR text and a non-empty summary JSON for each unit op.
On hosts without torch-mlir (most local dev boxes), the suite is skipped
via `pytest.importorskip`; the docker image installs torch-mlir nightly
and runs them.
"""
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch_mlir")

from npu_harness_framework import Pipeline, build  # noqa: E402
import torch_mlir_zoo  # noqa: E402, F401  (side-effect: registers stages)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "zoo"
OPS = ["attention", "rmsnorm", "mlp", "topk"]


def _run(op_name: str, tmp_path: Path):
    cfg = yaml.safe_load((CONFIG_DIR / f"{op_name}.yaml").read_text())
    # Redirect artifact paths under tmp_path so tests don't litter the repo.
    for stage in cfg["stages"]:
        if "out_path" in stage["config"]:
            stage["config"]["out_path"] = str(tmp_path / Path(stage["config"]["out_path"]).name)
    stages = [(s["name"], build(s["stage"], s["config"])) for s in cfg["stages"]]
    log_path = tmp_path / "profile.jsonl"
    return Pipeline(stages, log_path=str(log_path)).run(None), tmp_path, log_path


@pytest.mark.parametrize("op_name", OPS)
def test_export_emits_legal_mlir_and_summary(op_name, tmp_path):
    summary, out_dir, _ = _run(op_name, tmp_path)
    mlir_file = next(out_dir.glob("*.mlir"))
    text = mlir_file.read_text()
    # Legal top-level MLIR tokens
    assert "func.func" in text or "module" in text
    assert len(text.splitlines()) > 10
    # Summary structure
    assert summary["n_lines"] == len(text.splitlines())
    assert summary["op_counts"], "expected at least one torch.aten.* op"
    assert summary["server_side_op_hits"] == {}, "on-device op should not emit server-side hints"


def test_profiler_records_three_lines_per_op(tmp_path):
    _, _, log_path = _run("rmsnorm", tmp_path)
    lines = log_path.read_text().splitlines()
    assert len(lines) == 3
    stage_names = [json.loads(line)["stage"] for line in lines]
    assert stage_names == ["load", "export", "analyze"]
