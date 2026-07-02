"""NPU Harness Framework — modality-agnostic evaluation harness core.

The framework primitives: a config-driven registry for swapping stage
implementations, a per-stage profiler (latency / RAM / GPU), a marker ABC
for stages, and a linear Pipeline that chains stages with profiler
measurement.

Concrete plugins (voice STT/LLM/TTS, vision classifier, ...) live on
separate branches. Design intent and the Phase A/B/C generalization path
live in `docs/ARCHITECTURE.md`.
"""
from .interfaces import BaseStage
from .pipeline import Pipeline
from .profiler import measure
from .registry import build, register, registered

__version__ = "0.2.0"

__all__ = ["BaseStage", "Pipeline", "measure", "build", "register", "registered"]
