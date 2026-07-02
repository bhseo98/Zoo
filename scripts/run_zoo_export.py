#!/usr/bin/env python3
"""Step 4 — drive the unit-op Pipeline(s) end-to-end.

Loads `configs/zoo/<op>.yaml`, builds the loader → exporter → analyzer
stages, and runs them through the framework `Pipeline`. Output artifacts:
`artifacts/zoo/<op>.mlir` and `artifacts/zoo/<op>.summary.json`.

Usage:
    python scripts/run_zoo_export.py            # all four ops
    python scripts/run_zoo_export.py attention  # one op
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from npu_harness_framework import Pipeline, build
import torch_mlir_zoo  # noqa: F401  (side-effect: registers stages)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "zoo"
LOG_PATH = Path("logs/profile.jsonl")

ALL_OPS = ["attention", "rmsnorm", "mlp", "topk"]


def run_one(op_name: str) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / f"{op_name}.yaml").read_text())
    stages = [
        (s["name"], build(s["stage"], s["config"]))
        for s in cfg["stages"]
    ]
    # budget_mb=2048: 설계 검토 §2 "2GB 넘지 않는 모델" enforce.
    # 단위 op forward 단독은 보통 fit
    # 하지만 invariant 를 코드에 박아둔다 (Llama e2e 와 일관성).
    pipeline = Pipeline(stages, log_path=str(LOG_PATH), budget_mb=2048)
    print(f"\n=== {op_name} ===")
    summary = pipeline.run(None)
    print(json.dumps(summary, indent=2)[:400])
    return summary


def main() -> int:
    ops = [sys.argv[1]] if len(sys.argv) > 1 else ALL_OPS
    for op in ops:
        if op not in ALL_OPS:
            print(f"unknown op '{op}'. known: {ALL_OPS}")
            return 2
        run_one(op)
    return 0


if __name__ == "__main__":
    sys.exit(main())
