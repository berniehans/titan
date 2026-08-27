# Proposal: Phase 14 — Unified Engine Server & CLI Orchestration (Resident, Streaming, Speculative & MoE Modes)

## 1. Summary

Unify all specialized execution engines developed across prior phases (`ForwardDriver` with CUDA Graph acceleration, `StreamingForwardDriver` with PCIe DMA double-buffering, `HybridMoEExecutor` for sparse MoE, and `SpeculativeVerifier` / `NgramDraftProposer` for speculative decoding) into a unified engine runtime abstraction (`DriverInstance` / `EngineMode`) in `engine-server`. Expose seamless CLI controls (`titan serve`, `titan chat`) supporting automatic engine selection (`--engine auto|resident|streaming|moe`) and speculative acceleration (`--speculative auto|ngram|none`).

---

## 2. Motivation

- **Seamless Model Execution Across Scales:** Users should be able to run any GGUF model—whether a compact 0.6B/3B model that fits fully in VRAM, a dense 14B/32B/70B model requiring PCIe layer streaming, or a sparse MoE model—using the exact same CLI and HTTP OpenAI-compatible endpoints without manual code changes.
- **Intelligent Engine Auto-Selection:** The server and CLI should inspect model metadata, tensor sizes, and GPU VRAM budget to automatically select the optimal execution mode (`Resident` for $\le 5.2\text{ GB}$, `Streaming` for $> 5.2\text{ GB}$, `MoE` for sparse models).
- **End-to-End Speculative Acceleration:** Enable speculative decoding across both resident and streaming modes directly in production HTTP SSE streams and interactive terminal chat sessions.

---

## 3. Scope & Sub-Changes

1. **Sub-change 14.1 — Unified Driver Runtime Abstraction (`engine-core` / `engine-server`):**
   - Implement `DriverInstance` enum / abstraction unifying `ForwardDriver` and `StreamingForwardDriver`.
   - Implement `EngineMode` (`Auto`, `Resident`, `Streaming`, `MoE`) with automatic hardware-aware mode resolution.
   - Implement `UnifiedModel` handling prefill, decode, and speculative decode across all underlying drivers.

2. **Sub-change 14.2 — Server HTTP Endpoints & SSE Streaming Integration (`engine-server`):**
   - Wire `UnifiedModel` into Axum server state for `/v1/chat/completions` and `/v1/models`.
   - Support speculative streaming tokens over Server-Sent Events (SSE) with standard OpenAI JSON chunks.
   - Add telemetry metadata in HTTP response headers (`x-titan-engine-mode`, `x-titan-vram-mb`).

3. **Sub-change 14.3 — CLI Flag Unification & Interactive Chat Enhancements (`engine-server`):**
   - Update `titan serve` and `titan chat` CLI commands with `--engine`, `--speculative`, `--kv-capacity`, `--temperature`, `--top-p`, and `--system-prompt`.
   - Display a clean startup diagnostic banner detailing hardware detection, VRAM working set, and selected engine mode.

4. **Sub-change 14.4 — End-to-End Integration Gates, Benchmarks & Phase 14 Seal (`engine-server/tests` / `docs`):**
   - Implement `e2e_unified_modes_gate.rs` testing live HTTP completions and CLI chat across resident, streaming, and speculative configurations.
   - Run full workspace test suite `cargo test --workspace` with 0 failures.
   - Update `docs/BENCHMARKS.md`, sync delta spec to `spec.md`, and archive change.
