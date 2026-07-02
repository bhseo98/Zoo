#!/usr/bin/env python3
"""Sibling driver to run_zoo_export.py — same 4 ops, iree.turbine backend.

Loads `configs/zoo/<op>_iree_turbine.yaml`, runs the Pipeline, and prints
a diff vs the torch_mlir_dialect summary (if present in `artifacts/zoo/`).

Usage:
    python scripts/run_iree_turbine_export.py            # all four ops
    python scripts/run_iree_turbine_export.py attention  # one op
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from npu_harness_framework import Pipeline, build
import torch_mlir_zoo  # noqa: F401  (side-effect: registers stages)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "zoo"
LOG_PATH = Path("logs/profile-iree.jsonl")
ALL_OPS = ["attention", "rmsnorm", "mlp", "topk"]


def _diff_against_torch_mlir(op_name: str, iree_summary: dict) -> None:
    """If the `torch_mlir_dialect` summary exists alongside, print top-op diff."""
    other = Path("artifacts/zoo") / f"{op_name}.summary.json"
    if not other.exists():
        print(f"   (no torch_mlir_dialect baseline at {other} — skip diff)")
        return
    baseline = json.loads(other.read_text())
    a, b = baseline.get("op_counts", {}), iree_summary.get("op_counts", {})
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    print(f"   torch_mlir-only ops: {only_a[:8]}{'…' if len(only_a) > 8 else ''}")
    print(f"   iree_turbine-only ops: {only_b[:8]}{'…' if len(only_b) > 8 else ''}")
    print(f"   torch_mlir n_lines={baseline.get('n_lines')} vs iree_turbine n_lines={iree_summary.get('n_lines')}")


def run_one(op_name: str) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / f"{op_name}_iree_turbine.yaml").read_text())
    stages = [(s["name"], build(s["stage"], s["config"])) for s in cfg["stages"]]
    pipeline = Pipeline(stages, log_path=str(LOG_PATH), budget_mb=2048)
    print(f"\n=== {op_name} (iree.turbine) ===")
    summary = pipeline.run(None)
    print(json.dumps(summary, indent=2)[:400])
    _diff_against_torch_mlir(op_name, summary)
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
