# 기여 가이드라인 (Guidelines)

Zoo에 모델·op·export 백엔드를 추가하거나 수정할 때 지키는 규칙.

## 온디바이스 규칙 (필수)

- **표준 `torch.aten.*` op만** 사용. 커스텀 fused 커널·서버측 추상화(paged
  KV-cache, theta layer, device-affinity) 금지.
- 모든 export는 **`server_side_op_hits == {}`** 여야 함(`analysis/ir_summary`가
  검사). paged_attention / kv_cache / flash_attention / vllm / tensor_parallel /
  device_affinity 흔적 0.
- **forward-only.** prefill/decode 분리·KV cache·sampling 금지 — 컴파일러 분석용
  top-level torch graph를 유지.
- **정적 shape.** dynamic dim 회피(NPU 타깃).

## 새 것 추가 (확장점)

- **새 단위 op**: `src/torch_mlir_zoo/ops/`에 순수 `nn.Module` + `stages.py`의
  `_OP_REGISTRY` 한 줄 + `configs/zoo/<op>.yaml`(백엔드별 ×2). 코어·다른 op 불변.
- **새 모델**: `models/`에 forward-only 모델 + `@register("model", ...)` loader.
- **새 export 백엔드**: `exporters/`에 `(module, args) -> str` 함수 +
  `@register("exporter", "<name>")` stage + sibling YAML. 기존 백엔드는 additive로
  유지.
- **프레임워크 코어**(`interfaces` / `registry` / `pipeline` / `profiler`)는 절대
  fork 금지 — plugin으로만 확장(코어 diff 0 불변식).

## export 백엔드 선택

- **NPU 런타임 겨냥** = torch-mlir 백엔드(`torch_mlir.compile(OutputType.TORCH)`)
  — fused `aten.linear` 보존.
- **분석 / IREE-native** = iree.turbine — core-aten 분해(linear → mm/bmm).

## 양자화

- `block_scaled_q8`(INT8)는 **lowering 후 마지막 단계**. lowering을 올바르게 하는
  것의 대체(메모리 지름길)로 절대 쓰지 않음.

## 검증 (추가/수정 시 통과 기준)

- op 수치 정확성: `pytest`(atol 1e-5).
- lowering 정합성: IREE-CPU 컴파일+실행이 PyTorch와 일치
  (`tests/test_iree_cpu_numeric.py`, max_err ~1e-6).
- on-device 적합성: `server_side_op_hits == {}`.
- 2 GB budget: 프로파일러 경고 확인.

## 엔지니어링 규율

- 코딩 전 사고, 최소 코드, 외과적 변경(요청에 직접 추적되는 변경만), 검증 가능한
  목표(테스트로 성공 기준 코드화).
