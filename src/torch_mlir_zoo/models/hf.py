"""HuggingFace checkpoints → capture-ready modules (the L0 loader layer).

Loading is its own failure layer. A checkpoint that stores only one side of a
tied embedding pair (``facebook/opt-125m`` ships ``lm_head.weight`` and no
``decoder.embed_tokens.weight``) comes back with the other side left on the meta
device: no export can recover that, and the error it eventually throws blames
the exporter. :func:`load_hf_model` repairs the tie at load time instead.

The example-args table lives here too, because the input signature is a property
of the checkpoint's task, not of the exporter.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..capture import Diagnosis, ExportError, ForwardOnly, meta_params

__all__ = ["TASKS", "example_args", "load_hf_model"]

# task -> (transformers auto-class name, positional arg names for ForwardOnly)
TASKS: dict[str, tuple[str, tuple[str, ...] | None]] = {
    "causal_lm": ("AutoModelForCausalLM", None),
    "encoder": ("AutoModel", None),
    # Whisper's second positional is attention_mask, so bind by name.
    "speech_seq2seq": (
        "AutoModelForSpeechSeq2Seq",
        ("input_features", "decoder_input_ids"),
    ),
}


def _auto_class(task: str):
    import transformers

    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; known: {list(TASKS)}")
    return getattr(transformers, TASKS[task][0])


def _name_of(model: nn.Module, tensor: torch.Tensor) -> str | None:
    for name, param in model.named_parameters(remove_duplicate=False):
        if param is tensor:
            return name
    return None


def _repair_tied_embeddings(auto_cls, model_id: str, kwargs: dict) -> nn.Module:
    """Reload untied so the stored copy materializes, then restore the tie.

    Which side the checkpoint actually stores differs per model, so the missing
    key reported by the loader decides the copy direction.
    """
    model, info = auto_cls.from_pretrained(
        model_id, tie_word_embeddings=False, output_loading_info=True, **kwargs
    )
    missing = set(info.get("missing_keys", ()))
    inp, out = model.get_input_embeddings(), model.get_output_embeddings()
    if inp is not None and out is not None:
        in_name, out_name = _name_of(model, inp.weight), _name_of(model, out.weight)
        if in_name in missing and out_name not in missing:
            inp.weight = out.weight
        elif out_name in missing and in_name not in missing:
            out.weight = inp.weight

    unloaded = meta_params(model)
    if unloaded:
        raise ExportError(
            Diagnosis(
                "L0-load",
                f"{model_id}: {len(unloaded)} tensor(s) still unmaterialized after "
                f"tie repair: {unloaded[:4]}",
                "the checkpoint does not contain these weights — check the repo "
                "files, or load the variant that ships them",
            )
        )
    return model


def load_hf_model(
    model_id: str,
    task: str = "causal_lm",
    *,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Load ``model_id`` with every weight materialized, in eval mode.

    Attention is requested eager at construction so the model never builds the
    SDPA path (the capture layer would force it anyway, this just avoids the
    detour).
    """
    auto_cls = _auto_class(task)
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "attn_implementation": "eager"}

    # Safetensors first: transformers refuses .bin checkpoints on torch < 2.6.
    # Whichever kwargs load is the one the repair path has to reuse.
    try:
        kwargs["use_safetensors"] = True
        model = auto_cls.from_pretrained(model_id, **kwargs)
    except Exception:
        kwargs.pop("use_safetensors")
        model = auto_cls.from_pretrained(model_id, **kwargs)

    if meta_params(model):
        model = _repair_tied_embeddings(auto_cls, model_id, kwargs)
    return model.eval()


def example_args(model: nn.Module, task: str, seq_len: int = 32) -> tuple:
    """Fixed-shape example inputs for ``task`` — the traced shape of the export."""
    if task in ("causal_lm", "encoder"):
        return (torch.zeros(1, seq_len, dtype=torch.long),)
    if task == "speech_seq2seq":
        config = model.config
        frames = config.max_source_positions * 2  # hop-length halves the frames
        return (
            torch.zeros(1, config.num_mel_bins, frames),
            torch.zeros(1, seq_len, dtype=torch.long),
        )
    raise KeyError(f"unknown task {task!r}; known: {list(TASKS)}")


def forward_only(model: nn.Module, task: str) -> ForwardOnly:
    """Wrap ``model`` for tracing, binding args by name when the task needs it."""
    return ForwardOnly(model, arg_names=TASKS[task][1])
