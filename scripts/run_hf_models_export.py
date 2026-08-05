#!/usr/bin/env python3
"""HF-zoo lowering sweep — arbitrary HuggingFace checkpoints through one path.

Every model goes through the same two layers the SDK uses, so the sweep measures
the *framework*, not per-model hand-holding:

    models.hf.load_hf_model   L0 — materialize weights (tied-embedding repair)
    alpaca.export_for_npu     L1/L2 — capture rewrites, strict fallback, audit

Per model it records the capture strategy that won, the rewrites applied, the
aten histogram, the audit verdict, and — on failure — the layer diagnosis
instead of a raw traceback.

The profile matters for the kernel-coverage question. ``analysis`` decomposes, so
its histogram answers "which primitives appear"; ``fused`` keeps ops whole, which
is the shape a downstream pattern matcher consumes and therefore the only one
whose gaps are real. Results land in a per-profile file so the two never mix.

Run (inside a dedicated venv):
    python scripts/run_hf_models_export.py                       # whole sweep
    python scripts/run_hf_models_export.py --model qwen2.5-0.5b  # one model
    python scripts/run_hf_models_export.py --profile fused       # the fused shape
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from torch_mlir_zoo import ExportError, export_for_npu  # noqa: E402
from torch_mlir_zoo.analysis import audit  # noqa: E402
from torch_mlir_zoo.models import TASKS, example_args, load_hf_model  # noqa: E402

# (name, hf_id, family, task) — all locally cached; KO models included because
# the target application is Korean.
CANDIDATES = [
    ("distilgpt2", "distilgpt2", "gpt2", "causal_lm"),
    ("gpt2-small", "gpt2", "gpt2", "causal_lm"),
    ("opt-125m", "facebook/opt-125m", "opt", "causal_lm"),
    ("pythia-160m", "EleutherAI/pythia-160m", "gpt-neox", "causal_lm"),
    ("bert-base", "bert-base-uncased", "bert", "encoder"),
    ("bert-kor-base", "kykim/bert-kor-base", "bert", "encoder"),
    ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "llama", "causal_lm"),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "llama", "causal_lm"),
    ("tinyllama-1.1b", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", "causal_lm"),
    ("llama-3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", "llama", "causal_lm"),
    ("exaone-4.0-1.2b", "LGAI-EXAONE/EXAONE-4.0-1.2B", "llama", "causal_lm"),
    ("hyperclovax-1.5b", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B", "llama", "causal_lm"),
    ("whisper-tiny", "openai/whisper-tiny", "whisper", "speech_seq2seq"),
]

SEQ_LEN = 8
MAX_MLIR_BYTES = 500_000_000  # above this the .mlir is weights, not readable IR


def run_one(name: str, hf_id: str, family: str, task: str, out_dir: Path,
            profile: str = "analysis") -> dict:
    print(f"\n=== {name}  ({hf_id})")
    row = {"name": name, "hf_id": hf_id, "family": family, "task": task}

    t0 = time.perf_counter()
    try:
        model = load_hf_model(hf_id, task)
        args = example_args(model, task, SEQ_LEN)
    except ExportError as e:
        print(f"  LOAD FAIL — {e.diagnosis}")
        return {**row, "status": "load_fail", "layer": e.diagnosis.layer,
                "cause": e.diagnosis.cause, "hint": e.diagnosis.hint}
    except Exception as e:  # toolchain / network / gated repo
        print(f"  LOAD FAIL — {type(e).__name__}: {str(e)[:160]}")
        return {**row, "status": "load_fail", "layer": "L0-load",
                "cause": f"{type(e).__name__}: {str(e)[:300]}"}
    row["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    row["n_params"] = sum(p.numel() for p in model.parameters())
    print(f"  loaded ({row['load_ms']:.0f} ms) — {row['n_params']:,} params")

    t0 = time.perf_counter()
    try:
        r = export_for_npu(
            model, args, backend="iree_turbine", profile=profile,
            arg_names=TASKS[task][1]
        )
    except ExportError as e:
        print(f"  EXPORT FAIL — {e.diagnosis}")
        return {**row, "status": "export_fail", "layer": e.diagnosis.layer,
                "cause": e.diagnosis.cause, "hint": e.diagnosis.hint}
    row["export_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Turbine inlines weights as dense text, so a 1B model's MLIR is ~10 GB.
    # The histogram is what the sweep is for — only keep IR small enough to read.
    if len(r.mlir) <= MAX_MLIR_BYTES:
        (out_dir / f"{name}.mlir").write_text(r.mlir)
    else:
        print(f"  (IR not saved — {len(r.mlir) / 1e9:.1f} GB of inlined weights)")
    verdict = audit(r.mlir, profile=profile)
    print(f"  exported ({row['export_ms']:.0f} ms) — {r.summary['n_lines']} lines, "
          f"capture={r.capture}, rewrites={r.rewrites}, "
          f"unknown={list(verdict['unknown'])}, ssoh={r.summary['server_side_op_hits']}")
    return {
        **row,
        "status": "ok",
        "capture": r.capture,
        "rewrites": r.rewrites,
        "mlir_lines": r.summary["n_lines"],
        "n_aten_ops": sum(r.summary["op_counts"].values()),
        "unique_aten_ops": len(r.summary["op_counts"]),
        "profile": profile,
        "op_counts": r.summary["op_counts"],
        "dtypes": r.summary["dtypes"],
        "server_side_op_hits": r.summary["server_side_op_hits"],
        "unknown_ops": verdict["unknown"],
        "on_device_ok": verdict["ok"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[],
                    help="sweep only these (repeatable)")
    ap.add_argument("--profile", default="analysis",
                    choices=["analysis", "fused"],
                    help="analysis decomposes; fused is what a pattern matcher sees")
    args = ap.parse_args()

    out_dir = ROOT / "logs" / "hf-zoo"
    out_dir.mkdir(parents=True, exist_ok=True)
    only = args.model
    todo = [c for c in CANDIDATES if not only or c[0] in only]

    results = [run_one(*c, out_dir=out_dir, profile=args.profile) for c in todo]
    name = "results.json" if args.profile == "analysis" else "results-fused.json"
    (out_dir / name).write_text(
        json.dumps({"results": results}, indent=2, ensure_ascii=False)
    )

    print("\n=== SUMMARY ===")
    print(f"{'model':<18} {'family':<9} {'status':<12} {'params':>13} {'lines':>7} "
          f"{'uniq':>5} {'capture':>9}  unknown/diagnosis")
    for r in results:
        if r["status"] != "ok":
            print(f"{r['name']:<18} {r['family']:<9} {r['status']:<12} {'-':>13} {'-':>7} "
                  f"{'-':>5} {'-':>9}  [{r.get('layer')}] {r.get('cause', '')[:60]}")
        else:
            print(f"{r['name']:<18} {r['family']:<9} {r['status']:<12} {r['n_params']:>13,} "
                  f"{r['mlir_lines']:>7,} {r['unique_aten_ops']:>5} {r['capture']:>9}  "
                  f"{list(r['unknown_ops']) or 'none'}")
    ok = sum(r["status"] == "ok" for r in results)
    print(f"\nexported {ok}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
