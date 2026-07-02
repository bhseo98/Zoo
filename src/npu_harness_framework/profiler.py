"""Per-stage latency + memory profiler.

Writes one JSON line per measurement to `log_path`, and emits a short
human-readable summary to stdout. If `budget_mb` is set, warns when RSS
exceeds it (the 2GB DRAM constraint of the embedded target).
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import psutil


def _gpu_peak_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except ImportError:
        pass
    return 0.0


def _gpu_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


@contextmanager
def measure(
    stage: str,
    log_path: Path | str | None = None,
    budget_mb: int | None = None,
):
    proc = psutil.Process()
    ram_before = proc.memory_info().rss / (1024 ** 2)
    _gpu_reset()
    t0 = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ram_after = proc.memory_info().rss / (1024 ** 2)
    gpu_peak = _gpu_peak_mb()

    record = {
        "stage": stage,
        "elapsed_ms": round(elapsed_ms, 2),
        "ram_after_mb": round(ram_after, 1),
        "ram_delta_mb": round(ram_after - ram_before, 1),
        "gpu_peak_mb": round(gpu_peak, 1),
    }

    print(
        f"  ⏱  [{stage:<4}] {elapsed_ms:7.1f} ms  |  "
        f"RAM {ram_after:6.0f} MB (Δ {record['ram_delta_mb']:+.0f})  |  "
        f"GPU peak {gpu_peak:6.0f} MB"
    )

    if budget_mb is not None and ram_after > budget_mb:
        print(
            f"  ⚠  RAM {ram_after:.0f} MB exceeds budget {budget_mb} MB "
            f"— this model would not fit on the embedded target."
        )

    if log_path:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(record) + "\n")
