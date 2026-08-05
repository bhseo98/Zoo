"""Host-side evaluation — perplexity on the forward-only contract.

The on-device models recompute every position (no KV cache), so a single
forward over a token window yields the logits needed for token-level
cross-entropy — the same ``[B, T, vocab]`` contract lm-evaluation-harness'
``_model_call`` expects. This lets us measure, on the host with no board,
whether lowering / INT8 quantization preserves language-model quality.
"""
from .perplexity import perplexity, perplexity_delta

__all__ = ["perplexity", "perplexity_delta"]
