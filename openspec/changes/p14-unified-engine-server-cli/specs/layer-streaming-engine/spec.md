# Delta Specification: Phase 14 — Unified Engine Server & CLI Orchestration (Resident, Streaming, Speculative & MoE Modes)

## ADDED Requirements

### Requirement: Unified Multi-Engine Driver Abstraction
The server runtime SHALL provide a unified model interface (`UnifiedModel` / `DriverInstance`) capable of dispatching forward and speculative passes to `ForwardDriver` (resident GPU), `StreamingForwardDriver` (PCIe layer streaming), or `HybridMoEExecutor` (sparse MoE) through a common API.

#### Scenario: Automatic Engine Selection
- **WHEN** loading a model with `--engine auto`
- **THEN** if total model weight memory exceeds usable VRAM budget ($\le 5.2\text{ GB}$), the runtime SHALL automatically initialize `StreamingForwardDriver`
- **AND** if total model weight fits within VRAM budget, the runtime SHALL initialize `ForwardDriver` with CUDA Graph acceleration
- **AND** the active engine mode SHALL be logged during startup

### Requirement: Unified CLI Execution Flags
The CLI commands (`titan serve`, `titan chat`) SHALL support explicit and automatic engine mode selection (`--engine auto|resident|streaming|moe`) and speculative acceleration flags (`--speculative auto|ngram|none`).

#### Scenario: CLI chat with layer streaming
- **WHEN** running `titan chat --model <path> --engine streaming`
- **THEN** the CLI SHALL stream layer weights over PCIe double-buffers
- **AND** generate streaming responses interactively in the terminal with VRAM usage $< 2.0\text{ GB}$

### Requirement: Server Engine Telemetry Metadata
The HTTP server SHALL report the active engine mode and execution metrics via response headers on `/v1/chat/completions`.

#### Scenario: HTTP Response Headers
- **WHEN** a client sends a completion request to `POST /v1/chat/completions`
- **THEN** response headers SHALL include `x-titan-engine-mode` indicating the active backend (`resident`, `streaming`, or `moe`)
