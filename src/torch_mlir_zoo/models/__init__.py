"""On-device PyTorch-only model assemblies (Llama-3.2-1B, ...).

These compose the unit ops from `torch_mlir_zoo.ops` into complete model
forward passes that pass cleanly through `torch_mlir.compile`. They are
the on-device counterparts to amdsharktank's server-side
`PagedLlmModelV1` (see `docs/SHARK_AI_ANALYSIS.md` §2.4 / §3).
"""
from .llama_on_device import LlamaConfig, LlamaOnDevice, load_hf_weights

__all__ = ["LlamaConfig", "LlamaOnDevice", "load_hf_weights"]
