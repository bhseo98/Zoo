# Scope — 무엇을 공개하고, 무엇은 공개하지 않나

Torch-MLIR Model Zoo(Han-Neuro)는 온디바이스 NPU 스택의 **프론트엔드**만 다룹니다.
Qualcomm [`hexagon-mlir`](https://github.com/qualcomm/hexagon-mlir)과 같은 경계를
채택합니다 — **MLIR IR 레벨에서 열고, 하드웨어 백엔드는 닫는다.**

경계선은 **torch-dialect MLIR** 한 곳입니다. 그 위는 이 저장소, 그 아래는 하위 런타임.

```
┌──────────── 공개 (이 저장소) — 프론트엔드 ────────────┐
│  PyTorch Model Zoo   표준 aten 온디바이스 모델         │
│  Torch-MLIR export   torch-dialect MLIR (2 백엔드)     │
│  계약 검증           server_side_op_hits = 0           │
│  SDK / CLI / GUI     export_for_npu · IR 익스플로러    │
└──────────────── 경계 = torch-dialect MLIR ────────────┘
                    │  ◀── 여기까지만 공개 ──▶
┌──────────── 비공개 (하위 런타임) — 범위 밖 ───────────┐
│  NPU 런타임 컴파일러 (코드 생성)                       │
│  하드웨어 마이크로커널 · 인트린식                      │
│  온디바이스 런타임 · 시뮬레이터                        │
└───────────────────────────────────────────────────────┘
```

## 공개 (이 저장소)

경계 = **torch-dialect MLIR**. 여기까지가 프론트엔드입니다.

| 계층 | 내용 |
|---|---|
| **PyTorch Model Zoo** | 표준 `aten`만으로 된 온디바이스 모델 (PagedAttention/KV-cache 제거, forward-only) — 단위 op 4종(Attention·RMSNorm·SwiGLU·TopK) + `LlamaOnDevice` + Whisper-tiny |
| **Torch-MLIR export** | 두 백엔드 — `torch_mlir.compile`(합류·fused `aten.linear` 보존) / `iree.turbine`(core-aten) |
| **계약 검증** | `server_side_op_hits == 0`, fused `aten.linear` 보존 |
| **SDK / CLI** | `export_for_npu(model, args) → torch-dialect MLIR` |
| **IR 익스플로러 (GUI)** | 결과 IR을 노드 그래프로 시각화 |

## 비공개 (하위 런타임 — 범위 밖)

경계 아래는 별도로 관리되는 하위 스택이며, 이 저장소에 포함되지 않습니다.

- **NPU 런타임 컴파일러** (torch-dialect MLIR → 하드웨어 코드 생성)
- **하드웨어 마이크로커널 · 인트린식**
- **온디바이스 런타임 · 시뮬레이터**

우리 프론트엔드는 위 하위 스택이 **그대로 입력으로 받을 수 있는 표준 torch-dialect
MLIR**만 생산합니다. 두 절반은 이 경계에서 합류합니다.

## 참고 — 왜 이렇게 나누나

Qualcomm `hexagon-mlir`은 프론트엔드(Triton/PyTorch → MLIR dialect·pass, IR 검사)만
오픈소스로 공개하고, 하드웨어 코드 생성·마이크로커널·런타임은 proprietary로 둡니다.
Han-Neuro도 **정확히 같은 경계**를 따릅니다: 프론트엔드는 열고, 하드웨어에 밀착된
하위 스택은 각 하드웨어 소유자가 관리합니다.
