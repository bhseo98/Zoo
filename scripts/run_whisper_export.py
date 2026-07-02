#!/usr/bin/env python3
"""Whisper-tiny MLIR lowering — encoder-decoder zoo 확장 (PDF Step 6).

PDF Step 6 의 확장 타깃 `Whisper-tiny-INT8`(rhasspy/faster-whisper-tiny-int8)은
CTranslate2 `WhisperSpec` 바이너리라 torch.export 가 추적할 nn.Module 이 없다.
→ 아키텍처가 동일한 PyTorch `openai/whisper-tiny`(FP32)를 같은 export pipeline
(run_hf_models_export.py 와 동일: eager attention + forward-only wrapper +
export_via_iree_turbine + summarize)으로 lowering 해서 *어떤 kernel 이 문제가
되고 어떤 wrapper 클래스로 성공하는지* 를 Llama 와 똑같이 분석한다. 양자화는 마지막.

Whisper = encoder-decoder. Llama(decoder-only) 대비 새 surface:
  - Conv1d 프런트엔드 ×2 (mel → hidden)         ← Llama 에 없음
  - cross-attention (decoder ↔ encoder output)  ← Llama 에 없음
  - learned positional embedding (RoPE 아님)     ← Llama 의 RoPE custom kernel 자체가 없음
  - LayerNorm / GELU (RMSNorm / SwiGLU 아님)
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


class WhisperForwardOnly(nn.Module):
    """HF WhisperForConditionalGeneration → 단일 forward.

    (input_features, decoder_input_ids) -> logits.
    use_cache=False  → KV cache state arg / paged 분기 제거
    return_dict=False → tuple 반환 (graph 친화)
    """

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_features, decoder_input_ids):
        return self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=False,
        )[0]


def build(model_id: str = "openai/whisper-tiny"):
    from transformers import WhisperForConditionalGeneration

    m = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = m.config
    # encoder input: [B, num_mel_bins, 2*max_source_positions] (conv stride2 → max_source_positions)
    feat = torch.zeros(1, cfg.num_mel_bins, 2 * cfg.max_source_positions)
    dec = torch.zeros(1, 8, dtype=torch.long)  # fixed decoder length
    return WhisperForwardOnly(m), (feat, dec), m


def main() -> int:
    out_dir = ROOT / "logs" / "whisper"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = sys.argv[1] if len(sys.argv) > 1 else "openai/whisper-tiny"

    print(f"=== {model_id} ===")
    t0 = time.perf_counter()
    module, args, raw = build(model_id)
    n_params = sum(p.numel() for p in raw.parameters())
    print(f"  loaded ({(time.perf_counter()-t0)*1000:.0f} ms) — {n_params:,} params, "
          f"enc_in={tuple(args[0].shape)}, dec_in={tuple(args[1].shape)}")

    t1 = time.perf_counter()
    try:
        mlir = export_via_iree_turbine(module, args, func_name="forward")
    except Exception as e:
        print("  EXPORT FAIL:", type(e).__name__, str(e)[:400])
        print("\n".join(traceback.format_exc().splitlines()[-8:]))
        return 1
    export_ms = (time.perf_counter() - t1) * 1000

    (out_dir / "whisper-tiny.mlir").write_text(mlir)
    s = summarize(mlir)
    print(f"  exported ({export_ms:.0f} ms) — {s['n_lines']} lines, "
          f"{sum(s['op_counts'].values())} aten ops, "
          f"srv_hits={sum(s['server_side_op_hits'].values())}, "
          f"dynamic={s['has_dynamic_dim']}")

    # ── kernel surface 분석 (Whisper 고유) ──
    top = dict(sorted(s["op_counts"].items(), key=lambda kv: -kv[1])[:25])
    interesting = {k: s["op_counts"].get(k, 0) for k in [
        "convolution", "conv1d", "_convolution",
        "scaled_dot_product_attention",
        "_scaled_dot_product_flash_attention_for_cpu",
        "bmm", "baddbmm", "matmul", "softmax", "_softmax",
        "native_layer_norm", "layer_norm", "gelu", "embedding",
    ] if s["op_counts"].get(k, 0)}

    result = {
        "model_id": model_id, "status": "ok", "n_params": n_params,
        "enc_in": list(args[0].shape), "dec_in": list(args[1].shape),
        "mlir_lines": s["n_lines"], "module_name": s["module_name"],
        "n_aten_ops": sum(s["op_counts"].values()),
        "unique_aten_ops": len(s["op_counts"]),
        "top_ops": top,
        "whisper_surface_ops": interesting,
        "dtypes": s["dtypes"], "has_dynamic_dim": s["has_dynamic_dim"],
        "server_side_op_hits": s["server_side_op_hits"],
        "export_ms": round(export_ms, 1),
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n  -- Whisper surface ops --")
    for k, v in interesting.items():
        print(f"     {k:<48} {v}")
    print("\n  -- top aten ops --")
    for k, v in top.items():
        print(f"     {k:<48} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
