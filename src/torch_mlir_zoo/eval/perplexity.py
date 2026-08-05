"""Token-level perplexity via a single forward pass (forward-only contract)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(model: Any, input_ids: torch.Tensor) -> float:
    """Perplexity of ``model`` over ``input_ids`` (shape ``[1, T]``).

    Uses one forward pass: logits[..., :-1] predict tokens[..., 1:]. Works for
    any model returning ``[B, T, vocab]`` logits — including the on-device
    rewrites, whose forward-only recompute needs no KV cache.
    """
    model = model.eval()
    logits = model(input_ids)
    if not isinstance(logits, torch.Tensor):
        logits = logits.logits  # HF ModelOutput
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
    shift_labels = input_ids[:, 1:].reshape(-1)
    ce = F.cross_entropy(shift_logits, shift_labels)
    return torch.exp(ce).item()


def perplexity_delta(
    fp_model: Any, quant_model: Any, input_ids: torch.Tensor
) -> dict:
    """Compare fp vs quantized perplexity on the same tokens.

    Returns ``{fp, quant, abs, rel}`` — the accuracy impact of quantization
    measured at the language-model level (not just per-weight error).
    """
    fp = perplexity(fp_model, input_ids)
    qz = perplexity(quant_model, input_ids)
    return {"fp": fp, "quant": qz, "abs": qz - fp, "rel": (qz - fp) / fp}
