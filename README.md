<div align="center">

# 🦈 Zoo — Torch-MLIR Model Zoo

**서버측 패턴 없이 깨끗한 top-level torch-dialect MLIR로 내려가는
온디바이스 PyTorch 모델 주(zoo).**

paged-attention · KV-cache · vLLM 없이, 온디바이스 NPU 런타임 컴파일러가 소비할 수
있는 표준 IR을 만든다 — amdsharktank를 **온디바이스로 뒤집은** 프론트엔드.

`forward-only` · `server_side_op_hits = 0` · `모델 swap = config 한 줄`

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torch-MLIR](https://img.shields.io/badge/torch--mlir-LLVM-blueviolet)](https://github.com/llvm/torch-mlir)
[![IREE](https://img.shields.io/badge/IREE-turbine-orange)](https://github.com/iree-org/iree)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[**시작하기**](docs/GETTING-STARTED.md) ·
[**아키텍처**](docs/ARCHITECTURE.md) ·
[**레시피**](docs/RECIPES.md) ·
[**가이드라인**](docs/GUIDELINES.md) ·
[**범위**](docs/SCOPE.md)

</div>

---

## Zoo란

모델들이 **깨끗한 top-level torch-dialect MLIR**로 내려가도록 작성된 모델 주(zoo)다.
서버측 패턴(paged attention, KV-cache op, vLLM)이 전혀 없다.

산출물은 IR 파일 하나가 아니라 **IR + 그 IR의 성질에 대한 검증된 진술**이다 —
`server_side_op_hits == {}`, op allowlist 통과, 프로파일별 op 히스토그램. 매
export마다 기계로 확인한다.

이것은 커스텀 온디바이스 스택의 **프론트엔드**(PyTorch Model Zoo → Torch-MLIR
export)다. 경계 아래(런타임 컴파일러·마이크로커널·디바이스 레이아웃)는 하드웨어
소유자의 것이고 이 저장소에 없다 — [SCOPE.md](docs/SCOPE.md).

### 무엇을 빼고, 무엇을 남기나

- **뺀 것 = PagedAttention** (+ KV-cache / paging). 서버향 `PagedMHAttention` /
  `PagedGQAttention`을 온디바이스로 porting하며 제거.
- **남긴 것 = Attention · RMSNorm · MLP · Top-K.** Attention은 제거가 아니라 표준
  **SDPA(+GQA, causal)** 로 구현.

> ⚠ 흔한 오해: 뺀 건 **Paged**Attention(KV-cache/paging)이지 attention 자체가 아니다.

---

## 빠른 시작

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .[shark]              # 전용 venv 필수, 시스템 Python 금지

pytest                               # 86개 중 3 skip

alpaca models                        # export 가능한 zoo op 목록
alpaca export rmsnorm -o rmsnorm.mlir
alpaca audit rmsnorm.mlir            # allowlist 게이트 (유출 시 exit 2)

alpaca export rmsnorm --profile fused -o fused.mlir   # 융합 op 유지
alpaca audit fused.mlir --profile fused               # 감사도 프로파일별

python scripts/run_llama_export.py                     # Llama-3.2-1B → MLIR + summary.json
python scripts/run_hf_models_export.py --profile fused # 임의 HF 체크포인트 스윕
python scripts/certify_ops.py                          # allowlist 인증 (export + CPU vmfb)
```

SDK로 직접:

```python
from torch_mlir_zoo import export_for_npu, ExportError

try:
    r = export_for_npu(model, (example_input,), profile="fused")
except ExportError as e:
    print(e.diagnosis)   # [L0-load] / [L1-capture] / [L2-lowering] + 구체적 처방
else:
    r.ok                 # 서버측 op 유출 없음 = 온디바이스 적합
    r.capture, r.rewrites, r.summary["op_counts"]
```

전체 사용법은 [GETTING-STARTED.md](docs/GETTING-STARTED.md).

---

## 융합을 결정하는 것은 백엔드가 아니라 프로파일이다

하위 런타임 컴파일러는 보통 **융합 op**을 패턴 매치한다 — 융합 op 하나가 하드웨어
커널 하나에 대응하기 때문이다. 그래서 export 프로파일이 둘이고, 둘은 의도적으로
반대다:

| 프로파일 | decomposition | 한 decoder layer | 용도 |
|---|---|---|---|
| `analysis` (기본) | turbine 기본 테이블 | 98 ops / 25종 (`linear`→`mm`, `rms_norm`→pow·mean·rsqrt, SDPA→bmm·softmax) | op audit · IREE-CPU 실행 검증 |
| `fused` | **없음** | **37 ops / 12종** (`linear` 7 · `rms_norm` 2 · SDPA 1 유지) | 융합 op을 매치하는 하위 소비자 |

`analysis`가 "더 나쁜 IR"인 게 아니라 **소비자가 다르다.** 융합을 깨는 주체는
importer가 아니라 export 후에 도는 `run_decompositions()`이므로, 그것만 건너뛰면
pre-dispatch aten이 그대로 남는다.

```python
export_for_npu(model, args, profile="fused")
```

> **정정.** 이 저장소의 예전 문서는 "융합을 보존하려면 torch-mlir 백엔드를 써야
> 하고 turbine은 못 쓴다"고 적었다. 그 진술은 틀렸다. 게다가 `torch_mlir` 백엔드는
> 배포 wheel 인덱스가 2024-01에 멈춰 **설치되지 않는다** — 실질 백엔드는
> `iree_turbine` 하나이고, 프로파일이 융합을 정하므로 문제되지 않는다.

---

## 지원 모델

임의의 HuggingFace 체크포인트를 같은 진입점에 넣어 실측 — **두 프로파일 모두
12/13 export 성공**, 성공한 12개 전부 `server_side_op_hits == 0`:

| 계열 | 모델 | capture | 비고 |
|---|---|---|---|
| llama | Llama-3.2-1B · Qwen2.5-0.5B/1.5B · TinyLlama-1.1B · HyperCLOVAX-1.5B | strict | `forward` 안 `nn.ModuleList` 슬라이싱 → strict 자동 재시도 |
| gpt2 / neox | distilgpt2 · gpt2 · pythia-160m | nonstrict | |
| opt | opt-125m | nonstrict | 체크포인트가 tied 사본 한쪽만 저장 → 로더가 tie 복원 |
| bert | bert-base-uncased · bert-kor-base | nonstrict | encoder-only 대조군 |
| whisper | whisper-tiny | nonstrict | encoder-decoder, forward-only |
| — | EXAONE-4.0-1.2B | 실패 | transformers가 `exaone4` 아키텍처 미인식 — export가 아니라 **load** 실패(L0-load로 진단) |

`fused` 프로파일에서 llama 계열 5종과 pythia는 오래 막혀 있었다. 원인은 커널
갭이 아니라 캡처 계층의 갭이었다 — HF `forward` 안의 `with torch.autocast(...)` /
`with torch.no_grad()`를 `torch.export`가 higher-order op으로 감싸고 fx_importer가
거부했다. `capture.drop_context_markers()`가 그 마커를 HOP이 생기기 전에 지운다.
`analysis` 프로파일 IR은 적용 전후 **바이트 동일**이다.

### 두 게이트 다 초록

| 게이트 | `analysis` | `fused` |
|---|---|---|
| export | 12/13 | 12/13 |
| `server_side_op_hits == 0` | 12/12 | 12/12 |
| op allowlist | 12/12 통과 | 12/12 통과 |

`fused` allowlist는 한동안 **0/12**였다. `dropout`(12개 전부) · `layer_norm`(7개) ·
`conv1d`(whisper)가 미등재였기 때문이다. 서버측 op 유출이 아니라, 이 op들은
`analysis`에서 `run_decompositions`가 없애버려 감사에 아예 나타나지 않아 **"분해를
건너뛸 때만 나타나는 op"에 대해 allowlist가 한 번도 인증된 적이 없었던** 것이다.

셋 다 다른 op과 같은 기준으로 인증했다 — export + CPU vmfb 컴파일
([certify_ops.py](scripts/certify_ops.py)의 fused 프로파일 probe). `dropout`은 eval
모드에서 항등이라 실행 비용이 없고, `layer_norm`·`conv1d`는 실제 연산이다. 등재는
**"깨끗하게 내려간다"는 뜻이지 "타깃에 커널이 있다"는 뜻이 아니다** — 커널 커버리지는
별개 질문이다.

자체 구현 모델(`LlamaOnDevice`, 단위 op 4종)은 별도로 export된다. op 커버리지
행렬은 [op-coverage.md](docs/generated/op-coverage.md) — `scripts/build_op_coverage.py`가
실측에서 자동 생성.

allowlist 등재 기준은 문서가 아니라 **컴파일**이다. `scripts/certify_ops.py`가
op별 최소 probe를 export하고 CPU vmfb까지 컴파일해야 등재된다.

---

## amdsharktank와의 관계

발상은 같고(모델 zoo + PyTorch→MLIR→IREE), **방향은 반대**다 — amdsharktank는
데이터센터 GPU 서빙 스택, Zoo는 그것을 온디바이스 NPU용으로 뒤집은 프론트엔드.

| 항목 | amdsharktank | Zoo (torch-mlir-zoo) |
|---|---|---|
| 타깃 | AMD GPU (ROCm), multi-GPU | 온디바이스 NPU, single |
| 서버측 패턴 | **기반으로 삼음** (PagedAttention, KV-cache, ThetaLayer, sharding) | **의도적으로 제거** (`server_side_op_hits == 0`) |
| 실행 모델 | prefill/decode 분리, KV cache, sampling | forward-only, 매 position 재계산 |
| export | 항상 분해 (GPU linalg codegen) | 프로파일로 선택 — 분해 또는 융합 유지 |
| 규모 | 70B~405B, sharding | 1B급, sharding 없음 |
| 역할 | 모델 라이브러리 + 서빙 스택(전체) | 경계까지 책임지는 프론트엔드 |

**핵심 3가지:**
1. **추상화 방향 반대** — sharktank는 서버측 추상화 위에, Zoo는 그걸 벗겨 표준 aten만
   남긴다. `ops/`는 `PagedMHAttention` / `RMSNormLayer(ThetaLayer)` / `FFN`의 온디바이스 대체물.
2. **융합 선택권** — 하위 패스가 융합 `aten.linear`를 매치한다면 `fused` 프로파일로
   보존한다. sharktank에는 이 선택지가 없다(GPU codegen은 분해가 전제).
3. **완제품 vs 프론트엔드** — sharktank는 모델+서빙 전체, Zoo는 경계까지.

자세히: [docs/SHARK_AI_ANALYSIS.md](docs/SHARK_AI_ANALYSIS.md).

---

## 구성

| 패키지 / 경로 | 내용 |
|---|---|
| `npu_harness_framework` | 도메인 중립 코어 197 LOC — `interfaces`(BaseStage) · `registry` · `pipeline` · `profiler`. MLIR도 torch도 모르므로 export와 무관한 파이프라인에도 쓴다 → [사용법](docs/RECIPES.md#0-프레임워크-코어--stage--registry--pipeline--profiler) |
| `torch_mlir_zoo.alpaca` | SDK 파사드 — `export_for_npu(model, args)` 한 줄 진입점 (+ `alpaca` CLI) |
| `torch_mlir_zoo.capture` | **L1 계층** — 실패를 층(L0 load / L1 capture / L2 lowering)으로 진단, 알려진 rewrite 자동 적용, export 시점 컨텍스트 마커 제거 |
| `torch_mlir_zoo.ops` | 단위 op 4종 — `ScaledDotProductAttention` · `RMSNorm` · `SwiGLU` · `TopK` (순수 표준-aten `nn.Module`) |
| `torch_mlir_zoo.models` | `LlamaOnDevice`(forward-only) · `WhisperForwardOnly` + `hf` 로더 — 임의 HF 체크포인트를 가중치 전부 materialize된 상태로 로드 |
| `torch_mlir_zoo.kernels` | INT8 `block_scaled_q8` · `fused_rmsnorm` CustomOp + 이식 가능한 linalg 마이크로커널, 임베딩/tied-head 양자화 |
| `torch_mlir_zoo.exporters` | 두 export 백엔드, 동일 시그니처 — `torch_mlir_export` · `iree_turbine_export` |
| `torch_mlir_zoo.analysis` | `ir_summary`(op 히스토그램 + `server_side_op_hits`) · `op_audit`(프로파일별 allowlist 게이트) |
| `torch_mlir_zoo.eval` | forward-only perplexity — 양자화 전후 언어모델 수준 정확도 |
| `configs/zoo/*.yaml` · `scripts/` | op·모델 × 백엔드 config(`type` 한 줄 swap) + export 드라이버 · op 인증 |

---

## 특징

- **설계상 온디바이스** — 모든 모델이 `server_side_op_hits == 0`(paged-attention /
  KV-cache / vLLM op 없음), export마다 검증.
- **실패를 진단한다** — traceback이 아니라 `Diagnosis(layer, cause, hint)`. 층이 곧
  처방이다.
- **프로파일로 융합 제어** — 같은 모델, 두 소비자.
- **모델 swap 프레임워크** — 좁은 `BaseStage` + config-driven registry. 모델·백엔드
  추가가 프레임워크 코어를 건드리지 않는다(additive plugin).
- **allowlist는 컴파일로 인증** — 문서에 적는 것으로는 등재되지 않는다.

---

## 문서

| 문서 | 내용 |
|---|---|
| [GETTING-STARTED.md](docs/GETTING-STARTED.md) | 설치 → 첫 export → 프로파일 선택 → 실패 진단 |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | system / software / lowering 3-view + 다이어그램 |
| [RECIPES.md](docs/RECIPES.md) | 재현 가능한 명령 모음 |
| [GUIDELINES.md](docs/GUIDELINES.md) | 모델·op·백엔드 추가 규칙 + 엔지니어링 규율 |
| [SCOPE.md](docs/SCOPE.md) | 공개/비공개 경계 — 무엇을 보장하고 무엇은 보장하지 않나 |
| [SHARK_AI_ANALYSIS.md](docs/SHARK_AI_ANALYSIS.md) | amdsharktank 서버향 → 온디바이스 포팅 분석 |

---

## 기여

전체 규칙: [GUIDELINES.md](docs/GUIDELINES.md). 요약:

- **모델:** 표준 `torch.aten.*`만 — 커스텀 fused 커널·서버측 추상화 금지.
  lowering이 `server_side_op_hits == {}`를 보여야 함.
- **새 op / 모델 / 백엔드:** 모듈 + `@register` 한 줄 + YAML. 프레임워크 코어 fork 금지.
- **새 op의 allowlist 등재:** `scripts/certify_ops.py`가 CPU vmfb까지 컴파일해야 함.
- **공개 경계:** SKU명·마이크로커널명·디바이스 바이트 레이아웃·사이클 수 반입 금지
  ([SCOPE.md](docs/SCOPE.md)).

---

## 라이선스

[MIT](LICENSE).
