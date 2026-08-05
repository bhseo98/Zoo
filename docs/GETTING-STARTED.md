# 시작하기

PyTorch 모델 하나를 torch-dialect MLIR로 내리고, 그 IR이 온디바이스에 적합한지
확인하기까지 — 10분.

- 배경과 구조: [ARCHITECTURE.md](ARCHITECTURE.md)
- 공개/비공개 경계: [SCOPE.md](SCOPE.md)
- 명령 모음: [RECIPES.md](RECIPES.md)
- 기여 규칙: [GUIDELINES.md](GUIDELINES.md)

---

## 1. 설치

`torch-mlir` / `iree-turbine`은 무겁고 torch 버전에 민감하다. **전용 virtualenv를
쓰고 시스템 Python에는 절대 설치하지 않는다.**

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install -e .          # 코어 + zoo (toolchain 없이 동작하는 부분)

# + iree-turbine — export를 실제로 돌리려면 필요.
# --pre 필수: iree-turbine 3.10은 pre-release로만 배포되고, 리졸버는 기본적으로
# 그걸 거부한다. 빼면 부분 설치가 아니라 "No solution found"로 아무것도 안 깔린다.
pip install -e .[shark] --pre \
  --find-links https://iree.dev/pip-release-links.html
```

`uv`를 쓰면 `--pre` 대신 `--prerelease=allow`다.

설치 확인:

```bash
pytest                    # 86개 중 3 skip (toolchain 없으면 skip이 늘어난다)
alpaca models             # export 가능한 zoo op 목록
```

> **버전 상한 두 개는 취향이 아니라 실측이다.**
> `torch<2.6` — 2.6의 `run_decompositions()`가 `aten.linear`를 더 이상 쪼개지
> 않아 `analysis`와 `fused`가 **같은 IR**을 낸다(SwiGLU 실측: 2.5에서 `mm=3`/
> `linear=3`으로 갈리던 것이 2.6에서 둘 다 `linear=3`). 이 SDK의 두-프로파일
> 계약이 무너지는데 **테스트는 못 잡는다.**
> `transformers<5` — 5.x에서 Qwen 수치가 HF 대비 ~1e-4에서 4.6e-2로 벌어진다.

> `torch_mlir` 백엔드는 시그니처만 남아 있고 **설치되지 않는다** — 배포 wheel
> 인덱스가 2024-01에 멈췄고 해석되지 않는 torch nightly를 pin한다. 실질 백엔드는
> `iree_turbine` 하나다. 융합 여부는 백엔드가 아니라 프로파일이 정하므로 문제되지
> 않는다([ARCHITECTURE.md §3.3](ARCHITECTURE.md#33-융합을-결정하는-것은-백엔드가-아니라-프로파일이다)).

## 2. 첫 export — CLI

```bash
alpaca export rmsnorm -o rmsnorm.mlir     # zoo 단위 op 하나
alpaca audit rmsnorm.mlir                 # allowlist 게이트
```

`audit`은 서버측 op이 새어나오거나 allowlist 밖 op이 있으면 **exit 2**로 실패한다.
CI에 그대로 걸 수 있다.

## 3. 첫 export — SDK

```python
from torch_mlir_zoo import export_for_npu, ExportError

try:
    r = export_for_npu(model, (example_input,))
except ExportError as e:
    print(e.diagnosis)        # [L0-load] / [L1-capture] / [L2-lowering] + 처방
else:
    r.mlir                    # torch-dialect MLIR 텍스트
    r.ok                      # 서버측 op 유출 없음 = 온디바이스 적합
    r.summary["op_counts"]    # aten 히스토그램
    r.capture                 # 이긴 캡처 전략 (nonstrict | strict)
    r.rewrites                # 적용된 rewrite 목록
    r.save("model.mlir")
```

`example_input`은 **트레이스 shape을 고정하는 역할**이다. 온디바이스 타깃은 어차피
정적 shape이므로 dynamic dim은 쓰지 않는다.

## 4. 프로파일 고르기 — 가장 중요한 선택

같은 모델이 두 가지로 나온다. **소비자가 누구냐**로 정한다.

```python
export_for_npu(model, args, profile="analysis")   # 기본 — 분해된 IR
export_for_npu(model, args, profile="fused")      # 융합 유지
```

| | `analysis` | `fused` |
|---|---|---|
| `aten.linear` | `mm` + `t`로 분해 | **유지** |
| `aten.rms_norm` | pow·mean·rsqrt로 분해 | **유지** |
| SDPA | bmm + softmax로 분해 | **유지** |
| 한 decoder layer | 98 ops / 25종 | 37 ops / 12종 |
| 쓸 때 | op 감사, IREE-CPU 실행 검증, "어떤 primitive가 나오나" | 융합 op 하나 = 커널 하나로 매치하는 하위 소비자 |

감사도 프로파일을 따라간다:

```bash
alpaca export rmsnorm --profile fused -o fused.mlir
alpaca audit fused.mlir --profile fused
```

`analysis`에서 정상인 `mm`이 `fused`에서는 **분해가 새어나왔다는 신호**다. 그래서
allowlist가 프로파일별로 다르다.

> ⚠ **지금 `fused` allowlist는 미완성이다.** HF 체크포인트 12개 전부 `dropout`으로,
> 7개는 추가로 `layer_norm`으로 걸린다(whisper는 `conv1d`도). 서버측 op 유출이
> 아니라 등재가 안 된 것 — `analysis`에서는 `run_decompositions`가 없애버려 감사에
> 나타나지 않던 op들이다. `--profile fused` 감사가 빨간불이면 먼저 이걸 의심할 것.
> 상세는 [ARCHITECTURE.md §3.5](ARCHITECTURE.md#35-온디바이스-lowering--두-개의-게이트).

> `profile="fused"`와 `quantize="int8"`은 **함께 쓸 수 없다.** 양자화하면
> `aten.linear`가 전부 `block_scaled_q8` 호출로 바뀌어 융합 매처가 아무것도 못
> 찾는다. `export_for_npu`가 `ValueError`로 막는다.

## 5. 임의의 HuggingFace 체크포인트 넣기

```python
from torch_mlir_zoo.models import TASKS, example_args, load_hf_model
from torch_mlir_zoo import export_for_npu

task = "causal_lm"
model = load_hf_model("Qwen/Qwen2.5-0.5B-Instruct", task)
args = example_args(model, task, seq_len=8)

r = export_for_npu(model, args, profile="fused", arg_names=TASKS[task][1])
```

`load_hf_model`은 **가중치를 전부 materialize**한다 — meta device에 남은 텐서가
있으면 export가 아니라 로드에서 실패하고, 진단이 그렇게 말해준다. tied embedding
한쪽만 저장한 체크포인트(예: `facebook/opt-125m`)도 로더가 tie를 복구한다.

`arg_names`는 모델의 앞쪽 positional이 실제 입력이 아닐 때 쓴다 — Whisper는
`(input_features, attention_mask, decoder_input_ids)`를 받으므로 mask를 건너뛰어야
한다.

스윕으로 한 번에:

```bash
python scripts/run_hf_models_export.py                       # 전체, analysis
python scripts/run_hf_models_export.py --profile fused       # 전체, fused
python scripts/run_hf_models_export.py --model qwen2.5-0.5b  # 하나만
```

결과는 프로파일별 파일로 분리 저장된다 (`logs/hf-zoo/results.json` /
`results-fused.json`) — 두 히스토그램을 섞으면 커버리지 판단이 틀어진다.

## 6. export가 실패했을 때

traceback을 읽지 말고 `Diagnosis`를 읽는다. 층이 처방을 정한다:

```
[L1-capture] nn.ModuleList sliced inside forward (e.g. self.layers[:n]) —
             the non-strict export tracer cannot rebuild the proxied container
  fix: retry with the strict capture strategy (export_for_npu does this
       automatically)
```

| 층 | 뜻 | 보통의 처방 |
|---|---|---|
| `L0-load` | 가중치가 materialize 안 됨 | 로더 문제. `tie_word_embeddings=False`로 다시 로드 후 손으로 tie |
| `L1-capture` | `torch.export`가 그래프를 못 만듦 | 캡처 전략(자동 재시도), 값 의존 슬라이싱 제거 |
| `L2-lowering` | 잡힌 op에 lowering이 없거나 composite로 뭉침 | eager attention 강제(`rewrite=True`가 자동) |
| `unclassified` | 처음 보는 시그니처 | export를 직접 돌려 full traceback을 보고 `capture._RULES`에 시그니처 추가 |

캡처 전략은 사다리다 — `nonstrict` 먼저, 진단이 "strict으로 재시도"라고 하면
자동으로 `strict`. `r.capture`에 이긴 쪽이 기록된다.

## 7. INT8 (선택) — 반드시 lowering 다음

```python
r = export_for_npu(model, args, quantize="int8", block_size=32, verify=True)
r.accuracy["max_rel"], r.accuracy["cosine"], r.accuracy["argmax_match"]
```

`verify=True`면 fp 모델과 양자화 모델을 같은 입력에 돌려 출력 rel-error / cosine /
argmax 일치 + Linear별 가중치 오차를 붙여준다.

**양자화는 lowering이 올바른 뒤에만 한다.** 메모리가 모자라서 양자화부터 하는 것은
이 저장소의 규율에 어긋난다 — 틀린 lowering을 작게 만들 뿐이다.

## 8. 성공 신호 정리

| 확인 | 어디서 |
|---|---|
| 서버측 op 유출 없음 | `r.ok is True` / `summary.json`의 `server_side_op_hits == {}` |
| allowlist 통과 | `alpaca audit` exit 0 |
| 융합 유지(fused) | `op_counts`에 `linear` / `rms_norm` / `scaled_dot_product_attention` 존재, `mm` 부재 |
| 메모리 budget | `logs/profile.jsonl` — `budget_mb` 초과 시 경고 |
| 수치 정합성(INT8) | `r.accuracy["argmax_match"] == 1.0` |

## 9. 다음

- 명령 모음 · 재현 절차 → [RECIPES.md](RECIPES.md)
- 모델/op/백엔드 추가 규칙 → [GUIDELINES.md](GUIDELINES.md)
- 구조와 설계 근거 → [ARCHITECTURE.md](ARCHITECTURE.md)
