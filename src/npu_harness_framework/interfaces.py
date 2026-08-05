"""Abstract base for any pipeline stage — modality-agnostic core.

A single marker ABC. Concrete stages (encoder · classifier · decoder ·
generator · ...) inherit and supply their primary method through
`__call__`. The framework core (registry, build, profiler, Pipeline)
operates on `BaseStage` instances without knowing the modality.

See `docs/ARCHITECTURE.md` §3 (contract) and §10 (Phase A generalization).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStage(ABC):
    """Marker ABC. A stage is anything callable that transforms a payload."""

    @abstractmethod
    def __call__(self, payload: Any) -> Any:
        ...
