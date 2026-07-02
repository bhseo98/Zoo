# Model Zoo — 레시피

온디바이스 export/lowering 경로의 재현 가능한 명령. 별도 표기 없으면 모든 경로는
repo 기준 상대경로. shark 의존성은 전용 virtualenv 사용
(`pip install -e .[shark]`); 시스템 Python에 절대 설치 금지.

## 1. 온디바이스 모델 export → torch dialect MLIR

Config 기반 (파이프라인: tokenize → load_model → export → analyze):

```bash
# Llama-3.2-1B 온디바이스 (gated 체크포인트라 HF_TOKEN 필요)
export HF_TOKEN=hf_...
python scripts/run_llama_export.py            # → artifacts/llama-3.2-1b-on-device.mlir
                                              #   + .summary.json (op 수, server_side_op_hits)

# Whisper forward-only
python scripts/run_whisper_export.py

# 등록된 zoo 단위 op (attention / rmsnorm / mlp / topk) 를 config로
python scripts/run_zoo_export.py --config configs/zoo/attention.yaml
```

성공 확인: `summary.json`의 `server_side_op_hits`가 반드시 `{}` — 온디바이스
모델은 paged-attention / KV-cache / vLLM op를 내보내면 안 됨.

## 2. export 백엔드 선택 (런타임 합류를 결정)

같은 입력 `(module, args)`, 두 백엔드 (config `..._iree_turbine.yaml`이 turbine을,
기본 config가 torch-mlir을 선택):

| 백엔드 | config 접미사 | fused `aten.linear` | 용도 |
|---|---|---|---|
| `torch_mlir.compile(OutputType.TORCH)` | *(기본)* | **보존** | NPU 런타임 패스와 **합류(join)** |
| `iree.turbine.aot` | `_iree_turbine` | 분해 → `mm`/`bmm` | 분석 / IREE-native 경로 |

`torch-to-npu` 패스는 fused `aten.linear`를 매치한다(matmul-form 패턴 없음).
따라서 **NPU 런타임을 겨냥하면 torch-mlir 백엔드로 export**할 것.

## 3. 캡스톤: 런타임 계약 shape의 단일 decoder layer

계약 shape: `decoder_layer(hidden[1,32,2048]f16, cos[1,32,1,64]f16,
sin[1,32,1,64]f16) → [1,32,2048]f16`. zoo `SwiGLU` + RMSNorm + fused SDPA(GQA) +
RoPE(cos/sin as args)로 조립하고, NPU 런타임 겨냥 시 **torch-mlir 백엔드**로
export해 `aten.linear`를 보존한다(turbine은 `linear → mm` 분해로 합류 불가). zoo
op만으로 조립한 decoder block 예시는 `tests/test_iree_cpu_numeric.py`의
`_DecoderBlock`.

## 4. INT8 양자화 (lowering 후 마지막 단계)

`block_scaled_q8`(per-output-channel symmetric INT8)은 lowering이 올바른 *뒤에만*
적용 — 그 전의 메모리 지름길로 절대 쓰지 않음. 양자화 커널과 Rust parity·IREE
수치 검증은 별도로 관리한다.

## 5. 전달받은 NPU 런타임 — 소스에서 빌드 (요약)

> target NPU 런타임/VP 통합은 이 프론트엔드 repo 범위 밖(별도 관리). 참고용 요약:

1. 상류 IREE clone(`--recursive`), NPU dialect 플러그인 + host 샘플 overlay,
   CMake ≥ 3.26로 configure, `ninja iree-opt`(및 `iree-compile`).
   - `-DIREE_ENABLE_WERROR_FLAG=OFF` (LLVM `DenseMap.h`에서 GCC array-bounds
     false positive).
   - 툴에 플러그인 등록: `Tools/init_iree_{passes,dialects}.h` + CMake deps
     (이 배선은 shipped되지 않음).
   - host 샘플을 `samples/CMakeLists.txt`에 추가 (`add_subdirectory`).
2. NPU SystemC 모델: 독립 CMake, `cmake -B build && cmake --build build`.
3. 전체 VP: NPU 모델 디렉터리로 configure, `cmake --build`; 번들 qemu-configure는
   python `distlib` 필요.
4. RISC-V Linux 런타임 크로스빌드(`build-riscv-linux`): 최소 gcc `rv64gc/lp64d`
   toolchain 파일, `IREE_BUILD_COMPILER=OFF`, 샘플 on.

## 6. Layer-0 e2e (전달받은 harness)

`layer0_verify.sh`는 `iree-opt`로 재-lower하고, RISC-V 바이너리를 빌드하고, VP를
실행해 golden과 비교한다. 참조 데이터 정합화(DRAM load fragment, RoPE 테이블,
loader config)와 부트 체인은 별도 통합 쪽에서 관리. 현재 green 실행은 런타임팀의 `/dev/npu0` 커널 드라이버(shipped 안 됨)와
on-chip TCM allocator 블로커에 의존.
