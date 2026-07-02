"""LlamaOnDevice forward + 4-stage application tests.

Light tests run with tiny config (a few layers, small hidden_size) so
they don't need HF download or Llama-3.2-1B weights. The real-weights
e2e is gated behind `@pytest.mark.slow` + `RUN_HEAVY=1`.
"""
import json
import os
from pathlib import Path

import pytest
import torch

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


def test_llama_on_device_forward_shape():
    torch.manual_seed(0)
    model = LlamaOnDevice(TINY).eval()
    input_ids = torch.randint(0, TINY.vocab_size, (1, 16))
    logits = model(input_ids)
    assert logits.shape == (1, 16, TINY.vocab_size)
    assert torch.isfinite(logits).all()


def test_llama_on_device_gqa_kv_repeat():
    """num_kv_heads=2 should be repeated to num_heads=4 for attention."""
    model = LlamaOnDevice(TINY).eval()
    # Smoke test: attention forward without exception.
    input_ids = torch.randint(0, TINY.vocab_size, (2, 8))
    assert model(input_ids).shape == (2, 8, TINY.vocab_size)


def test_llama_hf_key_strip_logic_matches_expected_names():
    """Schema check: our state_dict keys are the HF keys with `model.` stripped.

    Catches drift if someone renames a sub-module — guards against the
    HF→on-device weight load silently dropping a layer.
    """
    model = LlamaOnDevice(TINY).eval()
    keys = set(model.state_dict().keys())
    must_have = {
        "embed_tokens.weight",
        "norm.weight",
        "lm_head.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
    }
    assert must_have.issubset(keys), f"missing: {must_have - keys}"


def test_llama_pipeline_dummy_runs_without_hf(tmp_path, monkeypatch):
    """4-stage Pipeline with tiny config + no HF download.

    Builds the Pipeline manually (skipping the YAML which references the
    real Llama-3.2 id) so the test runs on hosts without HF access.
    """
    pytest.importorskip("torch_mlir")
    from npu_harness_framework import Pipeline, build
    import torch_mlir_zoo  # noqa: F401

    model = LlamaOnDevice(TINY).eval()
    input_ids = torch.randint(0, TINY.vocab_size, (1, 16))

    class _DirectInput:
        def __call__(self, _): return input_ids
    class _DirectModel:
        def __call__(self, ids): return (model, (ids,))

    exporter = build("exporter", {"type": "torch_mlir_dialect", "out_path": str(tmp_path / "tiny.mlir")})
    analyzer = build("analyzer", {"type": "ir_summary", "out_path": str(tmp_path / "tiny.summary.json")})

    p = Pipeline(
        [("tokenize", _DirectInput()), ("load_model", _DirectModel()),
         ("export", exporter), ("analyze", analyzer)],
        log_path=str(tmp_path / "profile.jsonl"),
    )
    summary = p.run(None)
    assert summary["op_counts"], "expected at least one torch.aten.* op"
    assert summary["server_side_op_hits"] == {}
    assert (tmp_path / "tiny.mlir").exists()
    lines = (tmp_path / "profile.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert [json.loads(l)["stage"] for l in lines] == ["tokenize", "load_model", "export", "analyze"]


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_HEAVY") != "1", reason="set RUN_HEAVY=1 to run")
def test_llama_e2e_with_real_weights(tmp_path):
    """Full Llama-3.2-1B download + 4-stage Pipeline + IR dump."""
    pytest.importorskip("torch_mlir")
    pytest.importorskip("transformers")
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN required for gated Llama-3.2-1B")

    import yaml

    from npu_harness_framework import Pipeline, build
    import torch_mlir_zoo  # noqa: F401

    cfg = yaml.safe_load(
        Path("configs/zoo/llama_on_device.yaml").read_text()
    )
    for s in cfg["stages"]:
        if "out_path" in s["config"]:
            s["config"]["out_path"] = str(tmp_path / Path(s["config"]["out_path"]).name)
    stages = [(s["name"], build(s["stage"], s["config"])) for s in cfg["stages"]]
    p = Pipeline(stages, log_path=str(tmp_path / "profile.jsonl"))
    summary = p.run("Hello")
    assert summary["op_counts"]
    assert summary["server_side_op_hits"] == {}
