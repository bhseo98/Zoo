"""Harness stages for the Torch-MLIR Model Zoo.

Same plug-in pattern as `src/vision_app/stages.py` (vision-app branch):
each stage inherits `BaseStage` from the framework core, registers via
`@register`, and exposes a single `__call__`. The framework core remains
unchanged.

Three stages cover the loader → exporter → analyzer chain shared by all
unit-op pipelines (Step 4) and reused by the Llama application (Step 5).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from npu_harness_framework import BaseStage, register

from .ops import RMSNorm, ScaledDotProductAttention, SwiGLU, TopK

_OP_REGISTRY = {
    "attention": ScaledDotProductAttention,
    "rmsnorm": RMSNorm,
    "mlp": SwiGLU,
    "topk": TopK,
}


@register("loader", "zoo_op")
class ZooOpLoader(BaseStage):
    """Instantiate a unit op + dummy input. Returns `(module, args)`."""

    def __init__(self, op_name: str, dummy_shape: list[int], seed: int = 0):
        if op_name not in _OP_REGISTRY:
            raise KeyError(
                f"Unknown zoo op '{op_name}'. Known: {list(_OP_REGISTRY)}"
            )
        self._op_name = op_name
        self._shape = tuple(dummy_shape)
        self._seed = seed

    def __call__(self, _payload=None):
        torch.manual_seed(self._seed)
        module = _OP_REGISTRY[self._op_name]().eval()
        args = (torch.randn(*self._shape),)
        return (module, args)


@register("exporter", "torch_mlir_dialect")
class TorchMLIRDialectExporter(BaseStage):
    """`(module, args)` → top-level torch dialect MLIR text file."""

    def __init__(self, out_path: str):
        self._out = Path(out_path)

    def __call__(self, payload):
        from .exporters import export_top_level_torch_dialect

        module, args = payload
        text = export_top_level_torch_dialect(module, args)
        self._out.parent.mkdir(parents=True, exist_ok=True)
        self._out.write_text(text)
        return self._out


@register("exporter", "iree_turbine")
class IREETurbineExporter(BaseStage):
    """`(module, args)` → IREE-Turbine MLIR text file.

    Sibling of `torch_mlir_dialect` — same input/output shape, backend swap.
    Lets analyzer/configs/scripts compare both paths.
    """

    def __init__(self, out_path: str, func_name: str = "forward", strict: bool = False):
        self._out = Path(out_path)
        self._func_name = func_name
        self._strict = strict

    def __call__(self, payload):
        from .exporters import export_via_iree_turbine

        module, args = payload
        text = export_via_iree_turbine(
            module, args, func_name=self._func_name, strict=self._strict
        )
        self._out.parent.mkdir(parents=True, exist_ok=True)
        self._out.write_text(text)
        return self._out


@register("analyzer", "ir_summary")
class IRSummaryAnalyzer(BaseStage):
    """`.mlir` path → JSON summary (op_counts, dtypes, server_side_op_hits)."""

    def __init__(self, out_path: str):
        self._out = Path(out_path)

    def __call__(self, mlir_path):
        from .analysis import summarize

        text = Path(mlir_path).read_text()
        summary = summarize(text)
        self._out.parent.mkdir(parents=True, exist_ok=True)
        self._out.write_text(json.dumps(summary, indent=2))
        return summary


# ---------------------------------------------------------------------------
# Step 5 — Llama application stages (Tokenizer + Model)
# ---------------------------------------------------------------------------


@register("tokenizer", "hf_llama")
class HFLlamaTokenizer(BaseStage):
    """Prompt (str) → input_ids tensor `(1, max_length)` from HF tokenizer."""

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        max_length: int = 64,
    ):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._max = max_length

    def __call__(self, prompt: str) -> torch.Tensor:
        out = self._tok(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self._max,
            truncation=True,
        )
        return out.input_ids


@register("model", "llama_on_device")
class LlamaOnDeviceLoader(BaseStage):
    """Build LlamaOnDevice (optionally load HF weights). Returns `(model, args)`."""

    def __init__(
        self,
        hf_model_id: str | None = None,
        load_weights: bool = True,
        config_overrides: dict | None = None,
    ):
        from .models import LlamaConfig, LlamaOnDevice, load_hf_weights

        cfg = LlamaConfig(**(config_overrides or {}))
        self._model = LlamaOnDevice(cfg).eval()
        if load_weights:
            if not hf_model_id:
                raise ValueError("load_weights=True requires hf_model_id")
            load_hf_weights(self._model, hf_model_id, token=os.environ.get("HF_TOKEN"))

    def __call__(self, input_ids: torch.Tensor):
        return (self._model, (input_ids,))
