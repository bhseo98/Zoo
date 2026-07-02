# syntax=docker/dockerfile:1.7
# torch-mlir-zoo — PyTorch-only on-device building blocks + Torch-MLIR top-level
# IR dump. CPU-only. python:3.11-slim + torch CPU wheel + torch-mlir nightly +
# (Step 5) transformers for HF tokenizer / Llama weight loading.
# Framework core (npu_harness_framework) is unchanged.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

# torch CPU wheel + torch-mlir nightly + framework / app deps.
# torch-mlir is shipped as pre-releases on the LLVM index, not on PyPI.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 \
 && pip install --pre torch-mlir \
        -f https://llvm.github.io/torch-mlir/package-index/ \
 && pip install \
        transformers sentencepiece huggingface_hub \
        pyyaml psutil pytest

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs

RUN pip install -e ".[dev,llama]"

# HF_TOKEN is injected at runtime (`docker run -e HF_TOKEN=$HF_TOKEN ...`),
# never baked into the image.

CMD ["pytest", "tests/", "-v"]
