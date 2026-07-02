"""On-device PyTorch-only Llama (forward-only, no KV cache, no sampling).

Default config matches Llama-3.2-1B-Instruct
(<https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>). The
implementation only uses standard `torch.aten.*` ops so
`torch_mlir.compile(..., OutputType.TORCH)` produces a clean top-level
graph for the target NPU compiler stack (Step 5).

What's intentionally absent vs amdsharktank's `PagedLlmModelV1`
(`docs/SHARK_AI_ANALYSIS.md` §2.4):
  * KV cache + paging → forward recomputes every position
  * `prefill()` / `decode()` split → single `forward(input_ids)`
  * sampling / temperature / repeat penalty → `lm_head` returns raw logits
  * `iree.turbine.aot.DeviceAffinity` → single-NPU, no sharding hooks
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..ops import RMSNorm, SwiGLU


@dataclass
class LlamaConfig:
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 128256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 2048   # forward fixed; full HF default is 131072

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def _build_rope_cache(head_dim: int, max_seq_len: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)              # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)       # (T, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D) → broadcast to (1, 1, T, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class LlamaAttention(nn.Module):
    """Llama attention with RoPE + GQA, no KV cache."""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        h, kv = cfg.num_attention_heads, cfg.num_key_value_heads
        d = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, h * d, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv * d, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv * d, bias=False)
        self.o_proj = nn.Linear(h * d, cfg.hidden_size, bias=False)
        cos, sin = _build_rope_cache(d, cfg.max_position_embeddings, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h, kv, d = self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim
        q = self.q_proj(x).view(b, t, h, d).transpose(1, 2)
        k = self.k_proj(x).view(b, t, kv, d).transpose(1, 2)
        v = self.v_proj(x).view(b, t, kv, d).transpose(1, 2)
        cos = self.rope_cos[:t]
        sin = self.rope_sin[:t]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if kv != h:
            rep = h // kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, h * d)
        return self.o_proj(out)


class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = LlamaAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaOnDevice(nn.Module):
    """`input_ids: (B, T) → logits: (B, T, vocab_size)`."""

    def __init__(self, cfg: LlamaConfig | None = None):
        super().__init__()
        self.cfg = cfg or LlamaConfig()
        self.embed_tokens = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaBlock(self.cfg) for _ in range(self.cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(self.cfg.hidden_size, self.cfg.rms_norm_eps)
        self.lm_head = nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def load_hf_weights(
    model: LlamaOnDevice,
    hf_model_id: str,
    token: str | None = None,
) -> LlamaOnDevice:
    """Load weights from a HF `LlamaForCausalLM` checkpoint into LlamaOnDevice.

    HF state-dict keys are prefixed with `model.` (e.g.
    `model.layers.0.self_attn.q_proj.weight`); `lm_head.weight` is
    top-level. We strip the prefix and keep RoPE buffers (`rope_cos` /
    `rope_sin`) intact, since HF stores them per-attention as
    `inv_freq` and recomputes — ours are precomputed.
    """
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        hf_model_id, torch_dtype=torch.float32, token=token
    )
    hf_sd = hf.state_dict()
    target_sd = {}
    for k, v in hf_sd.items():
        if k.startswith("model."):
            target_sd[k[len("model.") :]] = v
        else:
            target_sd[k] = v
        # HF stores inv_freq buffers; ours uses precomputed cos/sin so we drop them.
        if k.endswith(".inv_freq") or k.endswith(".rotary_emb.inv_freq"):
            target_sd.pop(k.removeprefix("model."), None)
    missing, unexpected = model.load_state_dict(target_sd, strict=False)
    # Allow our rope_cos / rope_sin buffers to be 'missing' — they're persistent=False.
    unexpected = [k for k in unexpected if "inv_freq" not in k]
    if unexpected:
        raise RuntimeError(
            f"Unexpected HF keys not consumed by LlamaOnDevice: {unexpected}"
        )
    return model
