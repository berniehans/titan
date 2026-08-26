# Delta Specification: Phase 10 — OpenAI Chat Completions Server, Streaming SSE & Interactive CLI

## ADDED Requirements

### Requirement: OpenAI Chat Completions Wire Protocol
The server SHALL expose standard endpoints conforming to the OpenAI REST API specification:
- `POST /v1/chat/completions`: accepts messages array (`system`, `user`, `assistant`), optional sampling controls, and streaming flag.
- `GET /v1/models`: returns list of currently loaded model identifiers.

#### Scenario: Non-streaming Chat Completion
- **WHEN** client sends a `POST /v1/chat/completions` request with `stream: false`
- **THEN** server SHALL return JSON object with `choices[0].message.content` and `usage` token accounting

#### Scenario: Streaming Server-Sent Events (SSE)
- **WHEN** client sends a `POST /v1/chat/completions` request with `stream: true`
- **THEN** server SHALL emit chunks of type `text/event-stream` formatted as `data: {"choices": [{"delta": {"content": "..."}}]}\n\n`
- **AND** terminate the stream with `data: [DONE]\n\n` upon reaching stop sequence or max tokens

### Requirement: Advanced Sampling and Stop Sequence Control
The inference pipeline SHALL support configurable sampling parameters to control generation randomness and termination:
- Temperature scaling (with greedy argmax when $\le 10^{-4}$)
- Top-$K$ and Top-$P$ (nucleus) probability filtering
- Repetition penalty
- Stop tokens / custom stop word sequences

#### Scenario: Stop sequence trimming
- **WHEN** model generates `<|im_end|>` or any configured stop word sequence
- **THEN** generation SHALL terminate immediately with finish reason `stop`
- **AND** the stop token itself SHALL NOT be appended to user-visible content

### Requirement: Interactive Terminal CLI
The engine binary SHALL provide an interactive command-line interface (`titan chat`) for direct terminal conversation with live token-by-token streaming.
