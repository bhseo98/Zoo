#!/usr/bin/env python3
"""Step 5 — Llama-3.2-1B on-device end-to-end MLIR export application.

vision-app's 4-stage Pipeline pattern: tokenize → load_model → export →
analyze. Drops `artifacts/llama-3.2-1b-on-device.mlir` (top-level torch
dialect) and `artifacts/llama-3.2-1b-on-device.summary.json` (op counts,
server-side-op hits should be 0 vs amdsharktank's ≥1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from npu_harness_framework import Pipeline, build
import torch_mlir_zoo  # noqa: F401  (side-effect: registers stages)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "zoo" / "llama_on_device.yaml"
LOG_PATH = Path("logs/profile.jsonl")


def main() -> int:
    if not os.environ.get("HF_TOKEN"):
        print(
            "HF_TOKEN is not set. Llama-3.2-1B-Instruct is a gated repo; export\n"
            "it before re-running:  export HF_TOKEN=hf_..."
        )
        return 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("prompt", nargs="?", default="Hello, world.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    prompt = args.prompt
    stages = [(s["name"], build(s["stage"], s["config"])) for s in cfg["stages"]]
    # budget_mb=2048: 설계 검토 §2 "2GB 넘지 않는 모델" enforce.
    # Llama-3.2-1B FP32 ≈ 4GB → load_model stage 에서 ⚠ 경고가 떠야 정상 (PDF
    # Step 6 의 Q8_0 양자화 후 ≈ 1.2GB 가 budget fit 의 target).
    pipeline = Pipeline(stages, log_path=str(LOG_PATH), budget_mb=2048)

    summary = pipeline.run(prompt)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2)[:600])

    hits = summary.get("server_side_op_hits", {})
    if hits:
        print(f"\n⚠  server_side_op_hits not empty: {hits}")
        print("   The on-device Llama should not emit paged_attention/kv_cache/vllm.")
        return 3
    print("\n✓ server_side_op_hits = {} — on-device suitability verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
