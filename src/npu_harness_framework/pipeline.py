"""Linear, stage-list Pipeline — modality-agnostic core.

Each stage is wrapped in a profiler context manager so latency / RAM / GPU
peak get appended to the JSONL log without extra plumbing. Non-linear
topologies (DAG, training loop) are out of scope for this core — introduce
as separate Pipeline subclasses when a second domain demands them. See
`docs/ARCHITECTURE.md` §10.3 (Phase B).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .interfaces import BaseStage
from .profiler import measure


class Pipeline:
    def __init__(
        self,
        stages: list[tuple[str, BaseStage]],
        profiler_enabled: bool = True,
        log_path: str | Path | None = None,
        budget_mb: int | None = None,
    ):
        self._stages = list(stages)
        self._profile = profiler_enabled
        self._log = log_path
        self._budget = budget_mb

    def run(self, payload: Any) -> Any:
        log = self._log if self._profile else None
        budget = self._budget if self._profile else None
        out = payload
        for name, stage in self._stages:
            with measure(name, log, budget):
                out = stage(out)
        return out
