#!/usr/bin/env python3
"""HF-zoo MLIR lowering — 여러 Llama-similar HF 모델을 같은 export pipeline 으로.

candidate set (전부 ungated, small):
  - distilgpt2          (82M)   classic decoder, no SDPA
  - gpt2                (124M)  classic decoder, LayerNorm + GELU
  - opt-125m            (125M)  pre-RoPE Meta decoder, learned pos
  - pythia-160m         (162M)  EleutherAI, GPT-NeoX style
  - qwen2.5-0.5b        (494M)  Llama-family (RMSNorm/SwiGLU/RoPE/GQA)
  - tinyllama-1.1b      (1.1B)  Llama architecture, 22 layers
  - bert-base-uncased   (110M)  encoder-only — 비교 control

reference: Llama-3.2-1B (이미 step5 에서 export 했음)

각 모델별로:
  1. AutoModel*.from_pretrained(model_id, attn_implementation="eager", torch_dtype=fp32)
  2. ForwardOnly wrapper 로 감쌈 — (input_ids,) -> logits/last_hidden
  3. export_via_iree_turbine(...)
  4. summarize(mlir_text)
  5. logs/hf-zoo/{model}.mlir + results.json

attn_implementation="eager" 가 핵심 — 이게 없으면 transformers 가
F.scaled_dot_product_attention 을 호출해 _scaled_dot_product_flash_attention_for_cpu
opaque op 가 박힌다 (이전 turn arbitrary_model 테스트에서 확인됨).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from torch_mlir_zoo.exporters import export_via_iree_turbine
from torch_mlir_zoo.analysis.ir_summary import summarize


# ============================================================================
# Wrappers — convert HF model to forward(input_ids) -> tensor
# ============================================================================

class CausalLMWrapper(nn.Module):
    """HF AutoModelForCausalLM → forward(input_ids) returns logits tensor."""
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids):
        # return_dict=False → tuple; [0] = logits
        # use_cache=False → no past_key_values branching
        return self.model(input_ids, use_cache=False, return_dict=False)[0]


class EncoderWrapper(nn.Module):
    """HF AutoModel (BERT-style encoder) → forward(input_ids) returns last_hidden."""
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids):
        return self.model(input_ids, return_dict=False)[0]


# ============================================================================
# Candidates
# ============================================================================

CANDIDATES = [
    # (name, hf_id, family, wrapper, seq_len)
    ("distilgpt2",       "distilgpt2",                            "gpt2",     "causal",  32),
    ("gpt2-small",       "gpt2",                                  "gpt2",     "causal",  32),
    ("opt-125m",         "facebook/opt-125m",                     "opt",      "causal",  32),
    ("pythia-160m",      "EleutherAI/pythia-160m",                "gpt-neox", "causal",  32),
    ("qwen2.5-0.5b",     "Qwen/Qwen2.5-0.5B-Instruct",            "llama",    "causal",  32),
    ("tinyllama-1.1b",   "TinyLlama/TinyLlama-1.1B-Chat-v1.0",    "llama",    "causal",  32),
    ("bert-base",        "bert-base-uncased",                     "bert",     "encoder", 32),
]


def _build(hf_id: str, kind: str) -> tuple[nn.Module, tuple, int]:
    """Load HF model with eager attention; wrap; return (module, args, n_params)."""
    from transformers import AutoModel, AutoModelForCausalLM

    common = dict(torch_dtype=torch.float32, attn_implementation="eager")
    # use_safetensors prevents torch.load security path when only safetensors weights are present.
    # Some old checkpoints (opt-125m on HF) ship only .bin; try safetensors first, fall back.
    load_kwargs_list = [dict(use_safetensors=True), dict()]
    last_err = None
    for extra in load_kwargs_list:
        try:
            if kind == "causal":
                m = AutoModelForCausalLM.from_pretrained(hf_id, **common, **extra)
            elif kind == "encoder":
                m = AutoModel.from_pretrained(hf_id, **common, **extra)
            else:
                raise ValueError(kind)
            break
        except Exception as e:
            last_err = e
    else:
        raise last_err
    m.eval()
    wrapper = CausalLMWrapper(m) if kind == "causal" else EncoderWrapper(m)
    return wrapper, (torch.zeros(1, 32, dtype=torch.long),), sum(p.numel() for p in m.parameters())


def run_one(name: str, hf_id: str, family: str, kind: str, seq_len: int,
            out_dir: Path) -> dict:
    print(f"\n=== {name}  ({hf_id})")
    t_load0 = time.perf_counter()
    try:
        module, args, n_params = _build(hf_id, kind)
        load_ms = (time.perf_counter() - t_load0) * 1000
    except Exception as e:
        return {"name": name, "hf_id": hf_id, "family": family, "status": "load_fail",
                "error_type": type(e).__name__, "error": str(e)[:300]}

    print(f"  loaded ({load_ms:.0f} ms) — {n_params:,} params")

    t_exp0 = time.perf_counter()
    try:
        mlir = export_via_iree_turbine(module, args, func_name="forward")
        export_ms = (time.perf_counter() - t_exp0) * 1000
    except Exception as e:
        return {"name": name, "hf_id": hf_id, "family": family, "status": "export_fail",
                "n_params": n_params, "load_ms": round(load_ms, 1),
                "error_type": type(e).__name__, "error": str(e)[:500],
                "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-6:])}

    (out_dir / f"{name}.mlir").write_text(mlir)
    s = summarize(mlir)
    print(f"  exported ({export_ms:.0f} ms) — {s['n_lines']} lines, "
          f"{sum(s['op_counts'].values())} ops, srv_hits={sum(s['server_side_op_hits'].values())}")
    return {
        "name": name, "hf_id": hf_id, "family": family, "status": "ok",
        "n_params": n_params,
        "load_ms": round(load_ms, 1),
        "export_ms": round(export_ms, 1),
        "mlir_lines": s["n_lines"],
        "module_name": s["module_name"],
        "n_aten_ops": sum(s["op_counts"].values()),
        "unique_aten_ops": len(s["op_counts"]),
        "top_ops": dict(sorted(s["op_counts"].items(), key=lambda kv: -kv[1])[:15]),
        "dtypes": s["dtypes"],
        "has_dynamic_dim": s["has_dynamic_dim"],
        "server_side_op_hits": s["server_side_op_hits"],
    }


def main() -> int:
    out_dir = ROOT / "logs" / "hf-zoo"
    out_dir.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] if len(sys.argv) > 1 else None

    results = []
    for cand in CANDIDATES:
        if only and cand[0] not in only:
            continue
        results.append(run_one(*cand, out_dir=out_dir))

    (out_dir / "results.json").write_text(
        json.dumps({"results": results}, indent=2, ensure_ascii=False)
    )

    print("\n=== SUMMARY ===")
    print(f"{'model':<18} {'family':<10} {'status':<13} {'params':>12} "
          f"{'lines':>7} {'ops':>5} {'uniq':>5} {'srv':>4} {'load_s':>7} {'exp_s':>6}")
    for r in results:
        if r["status"] != "ok":
            print(f"{r['name']:<18} {r['family']:<10} {r['status']:<13} {'-':>12} "
                  f"{'-':>7} {'-':>5} {'-':>5} {'-':>4} "
                  f"{r.get('error_type','?')}: {(r.get('error','') or '')[:80]}")
        else:
            srv = sum(r['server_side_op_hits'].values())
            print(f"{r['name']:<18} {r['family']:<10} {r['status']:<13} "
                  f"{r['n_params']:>12,} {r['mlir_lines']:>7,} "
                  f"{r['n_aten_ops']:>5} {r['unique_aten_ops']:>5} {srv:>4} "
                  f"{r['load_ms']/1000:>7.1f} {r['export_ms']/1000:>6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
