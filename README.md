<div align="center">

# 🦈 Zoo — Torch-MLIR Model Zoo

**서버측 패턴 없이 깨끗한 top-level torch-dialect MLIR로 내려가는
온디바이스 PyTorch 모델 주(zoo).**

paged-attention · KV-cache · vLLM 없이, 온디바이스 NPU 런타임 컴파일러가 그대로
소비할 수 있는 IR을 만든다 — amdsharktank를 **온디바이스로 뒤집은** 프론트엔드.

`forward-only` · `server_side_op_hits = 0` · `모델 swap = config 한 줄`

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torch-MLIR](https://img.shields.io/badge/torch--mlir-LLVM-blueviolet)](https://github.com/llvm/torch-mlir)
[![IREE](https://img.shields.io/badge/IREE-turbine-orange)](https://github.com/iree-org/iree)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[**빠른 시작**](#빠른-시작) ·
[**아키텍처**](docs/ARCHITECTURE.md) ·
[**레시피**](docs/RECIPES.md) ·
[**가이드라인**](docs/GUIDELINES.md)

</div>

---

## Zoo란

모델들이 **깨끗한 top-level torch-dialect MLIR**로 내려가도록 작성된 모델 주(zoo)다.
서버측 패턴(paged attention, KV-cache op, vLLM)이 전혀 없어, 온디바이스 NPU 런타임
컴파일러가 그대로 소비한다.

이것은 커스텀 온디바이스 스택의 **프론트엔드**(PyTorch Model Zoo → Torch-MLIR export)로,
하위 NPU 런타임 컴파일러와 **torch-MLIR 경계에서 합류(join)**한다.

### 무엇을 빼고, 무엇을 남기나

- **뺀 것 = PagedAttention** (+ KV-cache / paging). 서버향 `PagedMHAttention` /
  `PagedGQAttention`을 온디바이스로 porting하며 제거.
- **남긴 것 = Attention · RMSNorm · MLP · Top-K.** Attention은 제거가 아니라 표준
  **SDPA(+GQA, causal)** 로 구현.

> ⚠ 흔한 오해: 뺀 건 **Paged**Attention(KV-cache/paging)이지 attention 자체가 아니다.

---

## amdsharktank와의 관계

발상은 같고(모델 zoo + PyTorch→MLIR→IREE), **방향은 반대**다 — amdsharktank는
데이터센터 GPU 서빙 스택, Zoo는 그것을 온디바이스 NPU용으로 뒤집은 프론트엔드.

| 항목 | amdsharktank | Zoo (torch-mlir-zoo) |
|---|---|---|
| 타깃 | AMD GPU (ROCm), multi-GPU | 온디바이스 NPU, single, CPU 호스팅 시뮬레이터 |
| 서버측 패턴 | **기반으로 삼음** (PagedAttention, KV-cache, ThetaLayer, sharding) | **의도적으로 제거** (`server_side_op_hits == 0`) |
| 실행 모델 | prefill/decode 분리, KV cache, sampling | forward-only, 매 position 재계산 |
| export 백엔드 | iree.turbine → **분해** (GPU linalg codegen) | 백엔드 2개; 합류 백엔드 = `torch_mlir.compile` → **fused `aten.linear` 보존** |
| 규모 | 70B~405B, sharding | 1B급, sharding 없음 |
| 역할 | 모델 라이브러리 + 서빙 스택(전체) | 런타임에 **합류**하는 프론트엔드 |

**핵심 3가지:**
1. **추상화 방향 반대** — sharktank는 서버측 추상화 위에, Zoo는 그걸 벗겨 표준 aten만
   남긴다. `ops/`는 `PagedMHAttention` / `RMSNormLayer(ThetaLayer)` / `FFN`의 온디바이스 대체물.
2. **fused 보존 vs 분해** — NPU 패스가 fused `aten.linear`를 매치 → 분해하는 turbine이
   아니라 `torch_mlir.compile` 백엔드로 합류.
3. **완제품 vs 프론트엔드** — sharktank는 모델+서빙 전체, Zoo는 런타임에 붙는 프론트엔드.

자세히: [docs/SHARK_AI_ANALYSIS.md](docs/SHARK_AI_ANALYSIS.md).

---

## 구성

| 패키지 / 경로 | 내용 |
|---|---|
| `npu_harness_framework` | 도메인 중립 코어 — `interfaces`(BaseStage) · `registry` · `pipeline` · `profiler` (~130 LOC) |
| `torch_mlir_zoo.ops` | 단위 op 4종 — `ScaledDotProductAttention` · `RMSNorm` · `SwiGLU` · `TopK` (순수 표준-aten `nn.Module`) |
| `torch_mlir_zoo.models` | `LlamaOnDevice` — forward-only Llama-3.2-1B (KV-cache / paging / sampling 없음) |
| `torch_mlir_zoo.exporters` | 두 export 백엔드, 동일 시그니처 — `torch_mlir_export`(합류) · `iree_turbine_export` |
| `torch_mlir_zoo.analysis` | `ir_summary` — op 히스토그램 + `server_side_op_hits` 카운터 |
| `configs/zoo/*.yaml` · `scripts/` | op·모델 × 백엔드 config(`type` 한 줄 swap) + export 드라이버 |

---

## 지원 모델

| 모델 | 구현 | export 백엔드 | `server_side_op_hits` |
|---|---|---|---|
| **Llama-3.2-1B** (on-device, forward-only) | `models/llama_on_device.py` | torch_mlir · turbine | `0` |
| **Whisper-tiny** (encoder-decoder, forward-only) | `scripts/run_whisper_export.py` | turbine | `0` |
| **단위 op ×4** (Attn · RMSNorm · SwiGLU · TopK) | `ops/` | torch_mlir · turbine | `0` |

> INT8 `block_scaled_q8` 양자화 커널은 **별도 repo에서 관리**한다 (이 저장소 밖).

---

## 빠른 시작

```bash
pip install -e .[shark]              # 전용 venv 권장

pytest                               # 단위 + export 테스트 (29 passed / 3 skipped)

python scripts/run_llama_export.py   # Llama-3.2-1B → torch-dialect MLIR + summary.json
python scripts/run_whisper_export.py # Whisper-tiny forward-only
```

성공 신호: export `summary.json`의 `server_side_op_hits == {}`. 자세히는
[RECIPES.md](docs/RECIPES.md).

---

## 아키텍처

Zoo는 스택의 **프론트엔드**(PyTorch → torch-dialect MLIR)를 담당하고, 하위 NPU 런타임
컴파일러와 **torch-MLIR 경계**에서 만난다:

```
PyTorch Model Zoo ──torch_mlir.compile(OutputType.TORCH)──▶ torch-dialect MLIR
  (표준 aten, fused aten.linear 보존)                            │  ◀── JOIN 경계
                                                                 ▼
                                     받은 NPU 런타임 컴파일러 + 시뮬레이터
```

**합류를 결정하는 단 하나의 선택** — 같은 모델, 두 백엔드:

| 백엔드 | `aten.linear` | 합류 |
|---|---|---|
| **`torch_mlir.compile(OutputType.TORCH)`** | **fused 보존** | ✅ NPU 패스가 fused `aten.linear`를 매치 |
| `iree.turbine.aot` | 분해 → `mm` / `bmm` | ✗ matmul-form 패턴 없음 |

전체 스택 다이어그램과 상세는 [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 특징

- **설계상 온디바이스** — 모든 모델이 `server_side_op_hits == 0`(paged-attention /
  KV-cache / vLLM op 없음), export마다 검증.
- **두 export 백엔드, 동일 시그니처** — config `type` 한 줄로 교체.
- **모델 swap 프레임워크** — 좁은 `BaseStage` + config-driven registry. 모델·백엔드
  추가가 프레임워크 코어를 건드리지 않는다(additive plugin).
- **재사용 단위 op** — 표준 op로 된 순수 `nn.Module` 4종.

---

## 문서

| 문서 | 내용 |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | system / software / lowering 3-view + 다이어그램 |
| [SHARK_AI_ANALYSIS.md](docs/SHARK_AI_ANALYSIS.md) | amdsharktank 서버향 → 온디바이스 포팅 분석 |
| [RECIPES.md](docs/RECIPES.md) | 재현 가능한 export / lowering 명령 |
| [GUIDELINES.md](docs/GUIDELINES.md) | 모델·op·백엔드 추가 규칙 + 엔지니어링 규율 |

---

## 기여

전체 규칙: [GUIDELINES.md](docs/GUIDELINES.md). 요약:

- **모델:** 표준 `torch.aten.*`만 — 커스텀 fused 커널·서버측 추상화 금지.
  lowering이 `server_side_op_hits == {}`를 보여야 함.
- **새 op / 모델 / 백엔드:** 모듈 + `@register` 한 줄 + YAML. 프레임워크 코어 fork 금지.
- **NPU 런타임용 export:** torch-mlir 백엔드(`aten.linear` 보존).

---

## 라이선스

[MIT](LICENSE).
