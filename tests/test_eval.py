"""Host-side perplexity eval (forward-only contract)."""
import torch

from torch_mlir_zoo.eval import perplexity, perplexity_delta
from torch_mlir_zoo.models.llama_on_device import LlamaConfig, LlamaOnDevice


def _tiny():
    cfg = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
    )
    return LlamaOnDevice(cfg).eval()


def test_perplexity_is_positive_finite():
    m = _tiny()
    ids = torch.randint(0, 256, (1, 16))
    ppl = perplexity(m, ids)
    assert ppl > 0 and torch.isfinite(torch.tensor(ppl))


def test_perplexity_delta_reports_quant_impact():
    import copy

    from torch_mlir_zoo.kernels import quantize_linears_

    m = _tiny()
    ids = torch.randint(0, 256, (1, 16))
    d = perplexity_delta(m, quantize_linears_(copy.deepcopy(m)), ids)
    assert set(d) == {"fp", "quant", "abs", "rel"}
    assert d["fp"] > 0 and d["quant"] > 0
