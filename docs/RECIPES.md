# 레시피

재현 가능한 명령 모음. 별도 표기 없으면 경로는 repo 기준 상대경로. 처음 쓴다면
[GETTING-STARTED.md](GETTING-STARTED.md)를 먼저 보는 편이 빠르다.

toolchain 의존성은 **전용 virtualenv**(`pip install -e .[shark]`)에 설치한다.
시스템 Python에 설치 금지.

---

## 1. 온디바이스 모델 export → torch-dialect MLIR

Config 기반 (파이프라인: tokenize → load_model → export → analyze):

```bash
# Llama-3.2-1B 온디바이스 (gated 체크포인트 — HF_TOKEN 필요)
export HF_TOKEN=hf_...
python scripts/run_llama_export.py       # → artifacts/llama-3.2-1b-on-device.mlir
                                         #   + .summary.json (op 수, server_side_op_hits)

# Whisper forward-only
python scripts/run_whisper_export.py

# 등록된 zoo 단위 op (attention / rmsnorm / mlp / topk)
python scripts/run_zoo_export.py --config configs/zoo/attention.yaml
```

성공 확인: `summary.json`의 `server_side_op_hits`가 반드시 `{}`.

## 2. 프로파일 — 융합을 결정하는 스위치

백엔드가 아니라 **프로파일**이 융합을 정한다. `run_decompositions()`를 돌리느냐
마느냐의 차이다.

```bash
alpaca export rmsnorm -o analysis.mlir                 # 기본: analysis
alpaca export rmsnorm --profile fused -o fused.mlir    # 융합 유지
alpaca audit fused.mlir --profile fused                # 감사도 프로파일별
```

```python
export_for_npu(model, args, profile="fused")
```

| 프로파일 | `linear` | `rms_norm` | SDPA | 한 decoder layer |
|---|---|---|---|---|
| `analysis` | `mm` + `t` | pow·mean·rsqrt | bmm + softmax | 98 ops / 25종 |
| `fused` | **유지** | **유지** | **유지** | 37 ops / 12종 |

확인:

```bash
python - <<'PY'
from torch_mlir_zoo import export_for_npu
from torch_mlir_zoo.ops import RMSNorm
import torch
for p in ("analysis", "fused"):
    r = export_for_npu(RMSNorm(512), (torch.randn(1, 32, 512),), profile=p)
    print(p, sorted(r.summary["op_counts"]))
PY
```

## 3. 임의 HF 체크포인트 스윕

```bash
python scripts/run_hf_models_export.py                       # 전체 · analysis
python scripts/run_hf_models_export.py --profile fused       # 전체 · fused
python scripts/run_hf_models_export.py --model qwen2.5-0.5b  # 하나만 (반복 가능)
```

프로파일별로 결과 파일이 갈린다 — `logs/hf-zoo/results.json` /
`results-fused.json`. 섞으면 커버리지 판단이 틀어진다.

각 행에 기록되는 것: 이긴 캡처 전략, 적용된 rewrite, aten 히스토그램, 감사 판정,
실패 시에는 traceback 대신 **층 진단**.

## 3-1. 내 커널 집합으로 어떤 모델이 되는지 알아내기

타깃 하드웨어를 갖고 있다면 가장 먼저 하고 싶은 질문이 이것이다: **"내가 구현한
커널로 어떤 모델이 끝까지 내려가나."** 스윕이 남긴 op 히스토그램이 그 답의 재료다.

```python
import json

# 내 타깃이 실행할 수 있는 aten op (커널이 있거나, 있는 커널로 표현되는 것)
HAVE = {"linear", "rms_norm", "scaled_dot_product_attention", "silu", "mul",
        "add", "softmax", "transpose", ...}

# 주소 계산이라 커맨드가 안 생기는 것 — DMA 오퍼랜드에 흡수된다
LAYOUT = {"view", "reshape", "expand", "permute", "unsqueeze", "squeeze", ...}

# 계산이 아니라 **데이터로 건네지는** 것 (아래 주의 참고)
STAGED = {"cos", "sin"}

for r in json.load(open("logs/hf-zoo/results-fused.json"))["results"]:
    if r["status"] != "ok":
        continue
    gap = sorted(set(r["op_counts"]) - HAVE - LAYOUT - STAGED)
    print(f"{r['name']:<18} {gap or '— 전부 덮임'}")
```

**반드시 `fused` 히스토그램으로 물을 것.** `analysis`로 물으면 갭이 과장된다 —
`mm`이 지원 안 되는 것처럼 보이지만 `fused`에서는 커널이 있는 `linear`로 남아
있었을 op이다.

> ⚠ **IR에 op이 있다고 커널이 필요한 건 아니다.** 실제로 겪은 사례: llama 계열
> export에 `cos`/`sin`이 나와서 6개 모델의 갭 목록에 올랐는데, 알고 보니 RoPE
> 테이블은 호스트가 미리 계산해 **데이터로 건네는** 것이었다. 계약 함수가 애초에
> `forward(hidden, cos, sin)`으로 받고 있었고, 디바이스 쪽엔 삼각함수 커널이 아예
> 없었다. 세 모델의 갭이 3 op에서 1 op으로 줄었다.
>
> 갭을 커널 요청으로 바꾸기 전에 물을 것: **이 op은 디바이스에서 계산되나, 아니면
> 결과가 이미 메모리에 올라가 있나?** 가중치·룩업 테이블·상수는 대개 후자다.

## 4. op allowlist 인증 — 문서가 아니라 컴파일이 기준

```bash
python scripts/certify_ops.py     # op별 최소 probe → export → CPU vmfb 컴파일
python scripts/build_op_coverage.py   # 실측에서 docs/generated/op-coverage.md 생성
```

컴파일되지 않은 op은 allowlist에 오르지 않는다. "지원한다"는 진술의 근거가
문서가 아니라 산출물이 되도록 하는 장치다.

**allowlist는 프로파일별이고, probe도 프로파일별이어야 한다.** `fused`에서만
살아남는 op(`dropout`·`layer_norm`·`conv1d`)은 오래 미인증 상태였다 —
`analysis`에서는 `run_decompositions`가 지워버려 감사에 아예 안 나타나기 때문이다.
게이트만 갈라놓고 probe를 안 가른 결과 HF 12개가 전부 빨간불이었다. 지금은 fused
probe로 인증돼 12/12다.

새 op을 넣을 때 같은 실수를 피하려면: **그 op이 어느 프로파일에서 보이는지 먼저
확인하고, 보이는 프로파일로 probe를 등록한다.**

## 5. export 실패 진단

```python
from torch_mlir_zoo import export_for_npu, ExportError
try:
    r = export_for_npu(model, args)
except ExportError as e:
    print(e.diagnosis.layer)   # L0-load / L1-capture / L2-lowering / unclassified
    print(e.diagnosis.cause)
    print(e.diagnosis.hint)
    raise e.original           # full traceback이 필요하면
```

`unclassified`가 나오면 시그니처를 `torch_mlir_zoo.capture._RULES`에 추가한다 —
그래야 다음 사람이 같은 벽에서 시간을 안 쓴다.

**자주 나오는 것들**

| 증상 | 층 | 처방 |
|---|---|---|
| `AttrProxy ... missing 1 required positional` | L1 | `forward` 안 `nn.ModuleList` 슬라이싱. strict 자동 재시도로 해결(이미 자동) |
| `Unhandled FakeTensor Device Propagation ... meta, cpu` | L0 | tied embedding 한쪽만 저장된 체크포인트. `tie_word_embeddings=False`로 로드 후 손으로 tie |
| `Higher-order operation 'wrap_with_autocast'` | L1 | `fused` 프로파일 + `with torch.autocast(...)`. `drop_context_markers()`가 자동 처리 — 그래도 나오면 `enabled=True` 블록이라 의도적으로 남긴 것이다 |
| SDPA가 불투명 composite 하나 | L2 | `rewrite=True`(기본)가 eager attention을 강제한다. `analysis` 프로파일에서만 필요 |

## 6. 컨텍스트 마커 직접 다루기

`export_for_npu`는 자동으로 적용한다. `torch.export`를 직접 부를 때만 필요하다:

```python
from torch_mlir_zoo.capture import drop_context_markers

with drop_context_markers():
    ep = torch.export.export(model, args, strict=True)
# ep.graph에 wrap_with_autocast / wrap_with_set_grad_enabled 없음
```

지우는 것: `_set_grad_enabled(*)`, `enabled=False`인 autocast 쌍.
남기는 것: `enabled=True`인 autocast — 지우면 내부 dtype이 조용히 바뀐다.

## 7. INT8 양자화 — lowering 후 마지막 단계

```python
r = export_for_npu(model, args, quantize="int8", block_size=32, verify=True)
print(r.accuracy["max_rel"], r.accuracy["cosine"], r.accuracy["argmax_match"])
```

```python
# 블록 크기와 headroom을 직접
from torch_mlir_zoo.kernels.quantized_linear import quantize_block_scaled_q8
qs, d = quantize_block_scaled_q8(w, block_size=64, headroom=127.0)
```

`headroom`은 블록 최대값을 int8 범위로 매핑하는 제수다. 127 = 전 범위(기본값).
더 작은 값은 위쪽을 비워 하위 누산기 overflow를 막는다 — **얼마나 비울지와 블록을
얼마로 잡을지는 그 누산기의 성질이지 이 함수의 성질이 아니다.** 값은 타깃
문서에서 가져올 것. 위 숫자는 호출 형태를 보여주는 예시일 뿐이다.

`block_size=None`은 per-channel(행당 스케일 하나).

> `profile="fused"`와 `quantize="int8"`은 함께 못 쓴다(`ValueError`). 양자화하면
> `aten.linear`가 전부 CustomOp 호출로 바뀌어 융합 매처가 아무것도 못 찾는다.

## 8. 단일 decoder layer 조립

하위 통합에서 흔한 단위. zoo op(RMSNorm + SDPA(GQA) + SwiGLU + RoPE)로 조립하고
`fused`로 export한다. cos/sin은 내부 buffer가 아니라 **forward 인자**로 뺀다.

조립 예시는 `tests/test_iree_cpu_numeric.py`의 `_DecoderBlock`. 상세한 델타 표는
[ARCHITECTURE.md §3.8](ARCHITECTURE.md#38-예시--단일-decoder-layer를-계약-shape으로).

## 9. 검증 한 바퀴

```bash
pytest                                    # 86개 중 3 skip
python scripts/certify_ops.py             # allowlist 인증
python scripts/run_hf_models_export.py --profile fused
alpaca audit artifacts/*.mlir --profile fused
```

## 10. 하위 런타임 통합 — 범위 밖

NPU 런타임 컴파일러 · 마이크로커널 · 디바이스 레이아웃 · 보드는 이 저장소에 없다.
이유와 경계는 [SCOPE.md](SCOPE.md). 이 저장소가 책임지는 계약은 torch-dialect MLIR
경계까지 — 위 §1~9다.
