"""Framework-core tests — registry, build, generic Pipeline, profiler.

Modality-agnostic. Dummy stages are pure string transformers — no model
loads, no torch, no audio. Runs in well under 1s on CPU.
"""
from __future__ import annotations

import json

import pytest

from npu_harness_framework import BaseStage, Pipeline, build, register, registered


class _Upper(BaseStage):
    def __init__(self, suffix: str = ""):
        self._suffix = suffix

    def __call__(self, payload: str) -> str:
        return payload.upper() + self._suffix


class _Repeat(BaseStage):
    def __init__(self, times: int = 2):
        self._times = times

    def __call__(self, payload: str) -> str:
        return payload * self._times


def test_register_decorator_and_build():
    @register("step", "_test_upper")
    class _D(_Upper):
        pass

    obj = build("step", {"type": "_test_upper", "suffix": "!"})
    assert obj("hi") == "HI!"


def test_build_rejects_unknown_type():
    @register("step", "_test_present")
    class _P(BaseStage):
        def __call__(self, payload):
            return payload

    with pytest.raises(KeyError):
        build("step", {"type": "_test_does_not_exist"})


def test_build_rejects_missing_type():
    @register("step", "_test_present2")
    class _P(BaseStage):
        def __call__(self, payload):
            return payload

    with pytest.raises(KeyError):
        build("step", {"suffix": "!"})


def test_build_does_not_mutate_caller_config():
    @register("step", "_test_noop")
    class _N(BaseStage):
        def __call__(self, payload):
            return payload

    cfg = {"type": "_test_noop"}
    build("step", cfg)
    assert "type" in cfg


def test_registered_returns_known_stages():
    @register("_test_ns", "x")
    class _X(BaseStage):
        def __call__(self, payload):
            return payload

    out = registered()
    assert isinstance(out, dict)
    assert "_test_ns" in out


def test_pipeline_chains_stages_in_order():
    p = Pipeline(
        [("upper", _Upper()), ("repeat", _Repeat(times=3))],
        profiler_enabled=False,
    )
    assert p.run("ab") == "ABABAB"


def test_pipeline_profiler_records_per_stage(tmp_path):
    log = tmp_path / "profile.jsonl"
    p = Pipeline(
        [("upper", _Upper()), ("repeat", _Repeat())],
        log_path=str(log),
    )
    p.run("xy")

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    stages = [json.loads(line)["stage"] for line in lines]
    assert stages == ["upper", "repeat"]
