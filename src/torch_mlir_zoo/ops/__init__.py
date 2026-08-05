"""On-device unit ops for Llama-style transformers.

Each op is a standard `nn.Module` composed only of `torch.aten.*` ops that
`torch_mlir.compile(..., OutputType.TORCH)` can lower without custom
kernels or fused decoders. See `docs/SHARK_AI_ANALYSIS.md` for the
server-side amdsharktank patterns these are replacing.
"""
from .attention import ScaledDotProductAttention
from .mlp import SwiGLU
from .rmsnorm import RMSNorm
from .topk import TopK

__all__ = ["ScaledDotProductAttention", "RMSNorm", "SwiGLU", "TopK"]
