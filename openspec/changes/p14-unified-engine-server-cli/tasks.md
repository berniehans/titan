# Implementation Tasks: Phase 14 — Unified Engine Server & CLI Orchestration (Resident, Streaming, Speculative & MoE Modes)

## 1. Unified Driver Runtime Abstraction (sub-change 14.1)
- [ ] 1.1 Implement `EngineMode` and `SpeculativeMode` enums in `engine-server/src/runtime.rs`.
- [ ] 1.2 Implement `DriverInstance` and `UnifiedModel` unifying `ForwardDriver` and `StreamingForwardDriver`.
- [ ] 1.3 Implement hardware-aware automatic engine selection heuristic.
- Gate: `cargo test -p engine-server --lib` PASS.

## 2. Server HTTP Endpoints & SSE Streaming Integration (sub-change 14.2)
- [ ] 2.1 Update `engine-server` Axum router and state to hold `Arc<Mutex<UnifiedModel>>`.
- [ ] 2.2 Route `/v1/chat/completions` (SSE streaming & JSON) through `UnifiedModel`.
- [ ] 2.3 Add engine telemetry metadata headers (`x-titan-engine-mode`, `x-titan-vram-mb`).
- Gate: `cargo test -p engine-server --test e2e_chat_completions` PASS.

## 3. CLI Flag Unification & Startup Diagnostics (sub-change 14.3)
- [ ] 3.1 Add `--engine` and `--speculative` CLI flags to `titan serve` and `titan chat` in `engine-server/src/main.rs`.
- [ ] 3.2 Display startup diagnostic banner with GPU specs, VRAM footprint, and selected engine mode.
- [ ] 3.3 Verify interactive CLI chat with streaming engine on GPU fixture.
- Gate: `titan chat --help` and `titan serve --help` display all options cleanly.

## 4. End-to-End Multi-Mode Verification & Phase 14 Seal (sub-change 14.4)
- [ ] 4.1 Create `engine-server/tests/e2e_unified_modes_gate.rs` testing live HTTP completions across resident, streaming, and speculative modes.
- [ ] 4.2 Run full workspace test suite `cargo test --workspace` with 0 failures.
- [ ] 4.3 Update `docs/BENCHMARKS.md` with unified multi-engine serving metrics.
- [ ] 4.4 Sync delta spec to main spec and archive change.
- Gate: All engine modes operational, tests green, Phase 14 sealed.
