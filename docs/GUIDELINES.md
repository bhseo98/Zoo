# 기여 가이드라인

Zoo에 모델·op·export 백엔드를 추가하거나 수정할 때 지키는 규칙.

---

## 온디바이스 규칙 (필수)

- **표준 `torch.aten.*` op만.** 커스텀 fused 커널·서버측 추상화(paged KV-cache,
  theta layer, device-affinity) 금지.
- 모든 export는 **`server_side_op_hits == {}`**. `paged_attention` / `kv_cache` /
  `flash_attention` / `vllm` / `tensor_parallel` / `device_affinity` 흔적 0.
- **forward-only.** prefill/decode 분리·KV cache·sampling 금지 — 컴파일러가 볼
  top-level torch graph를 하나로 유지한다.
- **정적 shape.** dynamic dim 회피. 온디바이스 타깃은 어차피 정적이다.

## 새 것 추가 (확장점)

- **새 단위 op**: `src/torch_mlir_zoo/ops/`에 순수 `nn.Module` + `stages.py`의
  `_OP_REGISTRY` 한 줄 + `configs/zoo/<op>.yaml`. 코어·다른 op 불변.
- **새 모델**: `models/`에 forward-only 모델 + `@register("model", ...)` loader.
- **새 export 백엔드**: `exporters/`에 `(module, args) -> str` 함수 +
  `@register("exporter", "<name>")` stage + sibling YAML. 기존 백엔드는 additive로 유지.
- **완전히 다른 파이프라인**: 코어(`npu_harness_framework`)만 쓰면 된다. MLIR도
  torch도 모르므로 export와 무관한 stage에도 그대로 붙는다 —
  [RECIPES §0](RECIPES.md#0-프레임워크-코어--stage--registry--pipeline--profiler).

### 코어는 건드리지 않는다

`interfaces` / `registry` / `pipeline` / `profiler` 네 파일은 **fork 금지**다.
zoo 전체가 코어에 diff 0인 additive plugin이고, 이게 유지되는 한:

- 새 op·모델·백엔드가 서로를 깨뜨리지 않는다
- 코어 197 LOC를 통째로 읽고 이해할 수 있다
- 다른 도메인이 코어만 가져다 쓸 수 있다

코어를 고쳐야 할 것 같으면 **대개 stage에 속할 로직이 코어로 새어 나온 것**이다.
확장점은 `@register`이지 코어 수정이 아니다. `BaseStage`가 `__call__` 하나뿐인
것도 같은 이유다 — batch·streaming·async를 넣자는 요구는 stage 안에서 해결한다.

## 프로파일과 백엔드

**융합을 결정하는 것은 백엔드가 아니라 프로파일이다.** 예전 문서는 반대로
적혀 있었다 — turbine이 `linear`를 분해하는 주체는 importer가 아니라 export *후*의
`run_decompositions()`이고, 그것만 건너뛰면 융합이 남는다.

- **융합 op을 매치하는 하위 소비자** = `profile="fused"`.
- **op 감사 / IREE-CPU 실행 검증** = `profile="analysis"`(기본).
- `torch_mlir` 백엔드는 시그니처만 남아 있고 **설치되지 않는다**. 새 코드가 그것에
  의존하면 안 된다.

감사도 프로파일을 따라간다. `analysis`에서 정상인 `mm`이 `fused`에서는 분해가
새어나왔다는 신호다.

## 컨텍스트 매니저

모델 `forward` 안의 `with torch.autocast(...)` / `with torch.no_grad()`는
`torch.export`가 higher-order op으로 감싸고, fx_importer가 거부한다.
`capture.drop_context_markers()`가 `export_for_npu` 경로에서 자동 처리한다.

- 새로 쓰는 zoo 모델은 **`forward` 안에 컨텍스트 매니저를 두지 않는다.** 가장 싼
  해법이다.
- 외부 체크포인트(HF 등)라 손댈 수 없으면 자동 처리에 맡긴다.
- `autocast(enabled=True)` 영역은 **의도적으로 남겨** 시끄럽게 실패시킨다. 지우면
  내부 dtype이 조용히 바뀐다. 그런 모델을 넣어야 하면 먼저 논의할 것.

## 양자화

- `block_scaled_q8`(INT8)는 **lowering 후 마지막 단계**. lowering을 올바르게 하는
  것의 대체(메모리 지름길)로 절대 쓰지 않는다.
- `profile="fused"` + `quantize="int8"`은 금지된 조합이다(`ValueError`). 양자화하면
  `aten.linear`가 전부 CustomOp 호출로 바뀌어 융합 매처가 아무것도 못 찾는다.
- `headroom`은 하위 누산기의 성질이다. 이 저장소의 기본값(127)은 int8 전 범위를
  쓴다.

## 실패는 진단한다

- export 실패를 traceback으로 던지지 않는다. `Diagnosis(layer, cause, hint)`로
  분류하고 **구체적 처방**을 붙인다.
- 처음 보는 실패는 `unclassified`로 두되, 시그니처를 `capture._RULES`에 추가한다.
- 층을 틀리면(로드 문제를 export 문제로 보는 등) 며칠을 잃는다. 층이 곧 처방이다.

## 갭을 보고할 때

타깃에 커널이 없다고 적기 전에, op을 네 부류로 갈라라. 셋은 요청이 아니다.

| 부류 | 판별 | 예 |
|---|---|---|
| **커널 있음** | 타깃이 실행한다 | `linear`, `rms_norm`, SDPA |
| **layout** | 주소 계산이라 커맨드가 안 생긴다 (DMA 오퍼랜드에 흡수) | `view`, `expand`, `permute` |
| **staged** | 계산이 아니라 **데이터로 건네진다** | RoPE `cos`/`sin` 테이블, 가중치 |
| **진짜 갭** | 위 셋 다 아니고 재작성도 안 된다 | ← 이것만 요청 목록에 올린다 |

셋째 부류가 가장 놓치기 쉽다. **IR에 op이 있다고 디바이스가 그걸 계산하는 건
아니다.** 실측 사례: `cos`/`sin`이 llama 6종의 갭 목록에 올라 있었는데, 참조 IR에
삼각함수 커널이 아예 없고 계약 함수가 `forward(hidden, cos, sin)`으로 테이블을
인자로 받고 있었다. 분류를 고치자 갭 총계가 34 → 22 op으로 줄었다.

판별 질문: **이 op은 디바이스에서 계산되나, 결과가 이미 메모리에 있나?**

증거 없이 재분류하지도 말 것. `cos`/`sin`은 셋(참조 IR·계약 시그니처·바이트 일치
검증)이 모여서 고쳤다. 그만한 근거가 없으면 갭으로 남기고 **왜 불확실한지**를 적는다.

## 검증 (추가/수정 시 통과 기준)

- op 수치 정확성: `pytest`(atol 1e-5).
- lowering 정합성: IREE-CPU 컴파일+실행이 PyTorch와 일치
  (`tests/test_iree_cpu_numeric.py`, max_err ~1e-6).
- on-device 적합성: `server_side_op_hits == {}`.
- allowlist: 새 op은 `scripts/certify_ops.py`가 CPU vmfb까지 컴파일해야 등재된다.
  문서에 적는 것으로는 등재되지 않는다.
- 2 GB budget: 프로파일러 경고 확인.

## 공개 경계

이 저장소는 프론트엔드만 다룬다([SCOPE.md](SCOPE.md)). 기여물에 다음이 들어가면
안 된다:

- 특정 하드웨어의 SKU명·마이크로커널명·ISA 인트린식
- 디바이스 바이트 레이아웃·메모리 맵·커맨드 스트림
- 사이클 수·특정 실리콘의 성능 수치

양자화 *스킴*은 여기 있어도 되지만, 특정 디바이스가 요구하는 바이트 배치는 그
디바이스의 것이다.

## 엔지니어링 규율

- 코딩 전 사고, 최소 코드, 외과적 변경(요청에 직접 추적되는 변경만), 검증 가능한
  목표(테스트로 성공 기준을 코드화).
- **검증할 수 없는 것을 문서가 주장하지 않는다.** 재현 명령이 없는 진술은 빼거나,
  왜 확인할 수 없는지를 함께 적는다.
