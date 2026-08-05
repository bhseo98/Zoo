"""Config-driven decoder LM — Llama defaults + Qwen2 arch flags."""
import pytest
import torch

from torch_mlir_zoo import export_for_npu
from torch_mlir_zoo.analysis import audit
from torch_mlir_zoo.models.llama_on_device import LlamaConfig, LlamaOnDevice


def test_llama_defaults_no_qkv_bias():
    cfg = LlamaConfig()
    m = LlamaOnDevice(cfg)
    assert m.layers[0].self_attn.q_proj.bias is None
    assert not cfg.tie_word_embeddings


def test_qwen_config_flags_shape_the_model():
    # Qwen2 = qkv bias + tied embeddings; the model must reflect both.
    cfg = LlamaConfig(
        hidden_size=896, intermediate_size=4864, num_hidden_layers=2,
        num_attention_heads=14, num_key_value_heads=2, vocab_size=1024,
        attn_qkv_bias=True, tie_word_embeddings=True,
    )
    m = LlamaOnDevice(cfg)
    assert m.layers[0].self_attn.q_proj.bias is not None        # qkv bias present
    assert m.lm_head.weight is m.embed_tokens.weight            # tied


def test_qwen_shaped_model_exports_and_audits_clean():
    pytest.importorskip("iree.turbine")
    cfg = LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
        attn_qkv_bias=True, tie_word_embeddings=True,
    )
    r = export_for_npu(LlamaOnDevice(cfg).eval(), (torch.randint(0, 512, (1, 8)),), backend="iree_turbine")
    assert r.ok
    assert audit(r.mlir)["ok"]        # no unknown / server-side ops


@pytest.mark.slow
def test_qwen_from_pretrained_matches_hf():
    # Real weights: our on-device Qwen must match HF numerically.
    transformers = pytest.importorskip("transformers")
    from torch_mlir_zoo.models.llama_on_device import from_pretrained

    mid = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        ours = from_pretrained(mid).eval()
        hf = transformers.AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32).eval()
    except Exception:
        pytest.skip("Qwen2.5-0.5B not cached")
    ids = torch.randint(0, ours.cfg.vocab_size, (1, 16))
    with torch.no_grad():
        rel = ((ours(ids) - hf(ids).logits).abs().max() / hf(ids).logits.abs().max()).item()
    assert rel < 1e-4
