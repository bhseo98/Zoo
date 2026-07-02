# AMD-SHARK-AI (amdsharktank) 분석 — 서버향 → 온디바이스 포팅

> 설계 스펙(`torch_mlir_model_zoo.pdf`) 과정 3단계: *"AMD-SHARK-AI에서 PyTorch
> 기반 LLM 모델링 코드를 Torch-MLIR로 lowering 하였고 이를 오픈소스로 공개…
> 서버향 LLM 모델링을 On-Device로 바꿔 porting 하기 위해 어떤 변화가 필요할지
> 분석(e.g. PagedAttention 제거)."*
>
> 이 문서는 `ops/`·`models/`가 대체하는 **서버측 패턴**과, 각 패턴을 온디바이스
> 표준 op로 바꾼 **근거**를 정리한다. `ops/__init__.py`, `models/__init__.py`,
> `models/llama_on_device.py`, `analysis/ir_summary.py`, `tests/test_zoo_ops.py`가
> 이 문서를 참조한다.

---

## §1 amdsharktank이란

amdsharktank(AMD SHARK / sharktank)는 PyTorch 기반 LLM 모델링 코드를 torch-MLIR /
IREE 경로로 lowering 해 오픈소스로 공개한 **성공 사례**다. 다만 타깃은
**데이터센터 GPU 서빙**(AMD GPU, ROCm, multi-GPU sharding)이라, 처리량을 위한
서버측 추상화 위에 세워져 있다. 우리는 같은 발상(모델 zoo + PyTorch→MLIR→IREE)을
**방향만 반대로**(온디바이스 NPU, 2 GB, forward-only) 가져온다. 정성적 비교표는
[README](../README.md#-amdsharktank와의-차이) 참조.

핵심 관찰: amdsharktank의 서버측 primitive는 (a) 커스텀 fused/paged 커널이라
torch-MLIR이 **불투명 op**로 보거나, (b) 오토리그레시브 제어흐름이라
`torch.export` 추적을 깨뜨린다. 둘 다 온디바이스 NPU 컴파일러가 소비할
**깨끗한 top-level torch dialect**를 막는다. 따라서 포팅 = 이 패턴들을 표준
`torch.aten.*`로 **재작성**하는 것.

---

## §2 서버측 패턴과 온디바이스 대체물

### §2.1 Attention — `PagedMHAttention` / `PagedGQAttention` → `ScaledDotProductAttention`

| 서버측 (amdsharktank) | 온디바이스 (`ops/attention.py`) |
|---|---|
| paged KV-cache(block table)로 attention | KV-cache·paging 없음, forward-only 고정 seq_len |
| 손으로 쓴 fused/paged 커널 | 표준 `matmul → (causal mask) → softmax → matmul` |
| MHA/GQA 분리 구현 | `num_kv_heads`로 GQA 지원(K/V `repeat_interleave`) |

> ⚠ **흔한 오해**: "attention을 뺀다"가 아니다. 뺀 것은 **Paged**Attention
> (KV-cache/paging)이고, attention 자체는 zoo의 핵심 op다 — 표준 SDPA로 남긴다.

### §2.2 정규화 / FFN — `RMSNormLayer(ThetaLayer)` / `FFN(ThetaLayer)` → `RMSNorm` / `SwiGLU`

amdsharktank는 가중치를 `ThetaLayer`(샤딩·양자화 메타데이터 래퍼)로 감싼다.
온디바이스에서는 그 래핑을 벗겨 순수 `nn.Module` + `nn.Parameter`만 남긴다.

- `RMSNorm`(`ops/rmsnorm.py`): dtype-stable — rsqrt는 float32로 계산 후 원 dtype 복귀
  (amdsharktank semantics 유지).
- `SwiGLU`(`ops/mlp.py`): gated SiLU — `down(silu(gate(x)) * up(x))`, sharding hook 없음.

### §2.3 샘플링 — `ServicePagedLlmModelV1.prefill()`의 top-k 블록 → `TopK`

amdsharktank의 샘플링은 temperature / repeat penalty / 오토리그레시브 루프 등
**제어흐름**에 둘러싸여 `torch.export` 추적을 깨뜨린다. 온디바이스에서는 그
제어흐름을 걷어내고 `torch.topk` 한 번(`ops/topk.py`)만 primitive로 남긴다.
`lm_head`는 raw logits를 반환한다(샘플링은 그래프 밖).

### §2.4 전체 모델 — `PagedLlmModelV1` → `LlamaOnDevice`

`models/llama_on_device.py`가 대체하는 대상. `PagedLlmModelV1` 대비 **의도적으로
빠진 것**:

| 서버측 요소 | 온디바이스 처리 |
|---|---|
| KV cache + paging | forward가 매 position 재계산 |
| `prefill()` / `decode()` 분리 | 단일 `forward(input_ids) → logits` |
| 샘플링 / temperature / repeat penalty | `lm_head`가 raw logits 반환(§2.3) |
| `iree.turbine.aot.DeviceAffinity` / sharding | 단일-NPU, sharding hook 없음 |

나머지(RoPE, GQA, RMSNorm, SwiGLU)는 §2.1–2.2의 표준 op로 재조립한다.

---

## §3 왜 PyTorch-only 표준 aten인가 (스펙 2단계)

Torch-MLIR은 PyTorch 코드를 MLIR로 lowering 하지만, 범용 LLM 프레임워크
(HF Transformers)는 PyTorch만으로 모델링하지 않는다(커스텀 커널·서버측 런타임 의존).
온디바이스 NPU 컴파일러가 소비하려면 모델이 **순수 표준 `torch.aten.*`** 로만
export-time을 통과해야 한다. §2의 재작성은 전부 이 제약을 만족시키기 위한 것이다.

---

## §4 검증 — `server_side_op_hits == 0`

`analysis/ir_summary.summarize()`가 export된 MLIR 텍스트에서 서버측 흔적
(`paged_attention`, `kv_cache`, `flash_attention`, `vllm`, `tensor_parallel`,
`device_affinity`)을 카운트한다. 온디바이스 목표 = `{}`; 서버 reference ≥ 1.
`LlamaOnDevice`·`WhisperForwardOnly` 모두 `{}` 를 달성하며, 이 값이 §2 재작성이
서버측 패턴을 실제로 제거했음을 export마다 회귀 검사한다.

## §5 export 경로 참조

amdsharktank가 내부에서 쓰는 export 경로(`iree.turbine.aot.FxProgramsBuilder`
+ `aot.export`)를 `exporters/iree_turbine_export.py`가 그대로 따른다
(참조: `amdsharktank/models/t5/export.py:50-89`, `amdsharktank/utils/export.py:214-234`).
단, NPU 런타임 합류에는 fused `aten.linear`를 보존하는 `torch_mlir.compile`
백엔드를 쓴다(turbine은 `linear → mm` 분해). 백엔드 선택 근거는
[ARCHITECTURE.md §3.2](ARCHITECTURE.md) 참조.
