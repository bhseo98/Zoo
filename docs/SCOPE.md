# Scope — 무엇을 공개하고, 무엇은 공개하지 않나

Torch-MLIR Model Zoo(Alpaca)는 온디바이스 NPU 스택의 **프론트엔드**만 다룹니다.
Qualcomm [`hexagon-mlir`](https://github.com/qualcomm/hexagon-mlir)과 같은 경계를
채택합니다 — **MLIR IR 레벨에서 열고, 하드웨어 백엔드는 닫는다.** hexagon-mlir이
MLIR 스택·프론트엔드·문서를 열고 SKU·사이클 수·마이크로커널 구현·ISA 인트린식을
닫는 것과 같은 선입니다.

경계선은 **torch-dialect MLIR** 한 곳입니다. 그 위는 이 저장소, 그 아래는 하위 런타임.

```
┌──────────── 공개 (이 저장소) — 프론트엔드 ────────────┐
│  PyTorch Model Zoo   표준 aten 온디바이스 모델         │
│  capture 계층        L0/L1/L2 진단 + 자동 rewrite      │
│  Torch-MLIR export   torch-dialect MLIR (2 백엔드)     │
│                      × 2 프로파일 (analysis / fused)   │
│  계약 검증           server_side_op_hits = 0 + allowlist│
│  SDK / CLI / GUI     export_for_npu · IR 익스플로러     │
└──────────────── 경계 = torch-dialect MLIR ────────────┘
                    │  ◀── 여기까지만 공개 ──▶
┌──────────── 비공개 (하위 런타임) — 범위 밖 ───────────┐
│  NPU 런타임 컴파일러 (코드 생성)                       │
│  하드웨어 마이크로커널 · 인트린식                      │
│  가중치 바이트 레이아웃 · 디바이스 메모리 맵           │
│  온디바이스 런타임 · 시뮬레이터 · 보드                 │
└───────────────────────────────────────────────────────┘
```

## 공개 (이 저장소)

| 계층 | 내용 |
|---|---|
| **PyTorch Model Zoo** | 표준 `aten`만으로 된 온디바이스 모델 (PagedAttention/KV-cache 제거, forward-only) — 단위 op 4종(Attention·RMSNorm·SwiGLU·TopK) + `LlamaOnDevice` + Whisper-tiny |
| **capture 계층** | 트레이스 실패를 L0-load / L1-capture / L2-lowering으로 진단하고, 측정으로 확인된 rewrite만 자동 적용 |
| **Torch-MLIR export** | 두 백엔드 — `torch_mlir.compile` / `iree.turbine` — × 두 프로파일(`analysis` / `fused`) |
| **계약 검증** | `server_side_op_hits == 0`, op allowlist(컴파일로 인증), 프로파일별 게이트 |
| **INT8 커널** | `block_scaled_q8` · `fused_rmsnorm` CustomOp + 이식 가능한 linalg 마이크로커널 |
| **SDK / CLI / GUI** | `export_for_npu(model, args) → torch-dialect MLIR`, IR 익스플로러 |

## 비공개 (하위 런타임 — 범위 밖)

경계 아래는 하드웨어 소유자가 관리하며 이 저장소에 포함되지 않습니다.

- **NPU 런타임 컴파일러** — torch-dialect보다 낮은 IR, 커맨드 스트림 생성, 코드 생성
- **하드웨어 마이크로커널 · 인트린식**
- **가중치 바이트 레이아웃 · 디바이스 메모리 맵** — 양자화 *스킴*은 여기 있지만,
  특정 디바이스가 요구하는 바이트 배치는 그 디바이스의 것입니다
- **온디바이스 런타임 · 시뮬레이터 · 보드**

## 이 경계가 보장하는 것과, 보장하지 않는 것

**보장하는 것.** 이 저장소는 표준 `torch.aten.*`만으로 된 torch-dialect MLIR을
생산하고, 그 사실을 매 export마다 기계로 검사합니다 — `server_side_op_hits == {}`,
allowlist 통과, 프로파일별 op 히스토그램. 하위 스택이 무엇이든 **입력의 성질은
검증된 값**입니다.

**보장하지 않는 것.** 특정 하위 컴파일러가 이 IR을 *그대로* 소비한다는 것은 이
저장소가 검증할 수 없습니다. 실제 스택에서 코드 생성이 어느 IR 레벨에서 일어나는지는
그 스택의 설계 결정이고, torch-dialect보다 낮은 IR에서 시작하는 구성도 있습니다.
따라서 이 저장소가 주장하는 것은 **"우리가 내보내는 IR이 무엇인지"**까지이며,
**"하위가 그것을 어떻게 받는지"**는 각 통합의 계약 문제입니다.

이 구분은 중요합니다. 전자는 여기서 재현 가능하고(`pytest`, `scripts/`), 후자는
하드웨어와 하위 툴체인 없이는 확인할 수 없습니다. 확인할 수 없는 것을 문서가
주장하지 않는 편이 낫습니다.

## 참고 — 왜 이렇게 나누나

Qualcomm `hexagon-mlir`은 프론트엔드(Triton/PyTorch → MLIR dialect·pass, IR 검사)만
오픈소스로 공개하고, 하드웨어 코드 생성·마이크로커널·런타임은 proprietary로 둡니다.
공개 문서에 SKU명·사이클 수·내부 커널명이 등장하지 않습니다. Alpaca도 **정확히 같은
경계**를 따릅니다: 프론트엔드는 열고, 하드웨어에 밀착된 하위 스택과 그 수치는 각
하드웨어 소유자가 관리합니다.
