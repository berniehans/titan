# Design: Phase 14 — Unified Engine Server & CLI Orchestration (Resident, Streaming, Speculative & MoE Modes)

## 1. Architectural Architecture

```
                                  ┌────────────────────────┐
                                  │      CLI / HTTP        │
                                  │ (titan serve / chat)   │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │       UnifiedModel       │
                                 │   (EngineMode::Auto)     │
                                 └────────────┬─────────────┘
                                              │
                    ┌─────────────────────────┼────────────────────────┐
                    ▼                         ▼                        ▼
         ┌─────────────────────┐   ┌──────────────────────┐  ┌──────────────────┐
         │    ForwardDriver    │   │StreamingForwardDriver│  │HybridMoEExecutor │
         │ (Resident GPU +     │   │ (PCIe DMA Double-    │  │ (Dynamic Slot    │
         │  CUDA Graph Dec.)   │   │  Buffer <200MB VRAM) │  │  Expert Cache)   │
         └──────────┬──────────┘   └──────────┬───────────┘  └────────┬─────────┘
                    │                         │                       │
                    └─────────────────────────┼───────────────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │   SpeculativeVerifier    │
                                 │  (NgramDraftProposer)    │
                                 └──────────────────────────┘
```

---

## 2. Component Design

### 2.1 Unified Model & Driver Instance (`engine-server/src/runtime.rs`)
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
pub enum EngineMode {
    Auto,
    Resident,
    Streaming,
    Moe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
pub enum SpeculativeMode {
    Auto,
    Ngram,
    None,
}

pub enum DriverInstance<'a> {
    Resident(ForwardDriver<'a>),
    Streaming(StreamingForwardDriver<'a>),
}
```

### 2.2 Heuristic for Engine Mode `Auto`
1. Read model config from GGUF.
2. Sum all tensor bytes (`total_weight_bytes`).
3. If `total_weight_bytes > 5.2 GB` (or if GPU VRAM $< 6\text{ GB}$), select `EngineMode::Streaming`.
4. Else, select `EngineMode::Resident`.

### 2.3 CLI Command Arguments (`engine-server/src/main.rs`)
```rust
#[derive(clap::Parser)]
pub struct ServeArgs {
    #[arg(long, default_value = "auto")]
    pub engine: EngineMode,

    #[arg(long, default_value = "auto")]
    pub speculative: SpeculativeMode,

    #[arg(long, default_value = "128")]
    pub kv_capacity: usize,

    #[arg(long, default_value = "8080")]
    pub port: u16,
}
```

---

## 3. Telemetry & Verification Strategy

- **HTTP Headers:** Emits `x-titan-engine-mode`, `x-titan-speculative-mode`, and `x-titan-vram-mb`.
- **E2E Gate:** `e2e_unified_modes_gate.rs` spins up the local Axum server with `EngineMode::Resident` and `EngineMode::Streaming`, querying `/v1/chat/completions` with streaming SSE and verifying exact completions and response headers.
