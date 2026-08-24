# Tasks: 6.1-config-tokenizer-goldens

> Execute via bot coder. Strict TDD. One commit per task group.

## 1. ModelConfig (engine-io)
- [ ] 1.1 Failing test: parse GGUF metadata from the real `Qwen3-0.6B-Q4_K_M.gguf` fixture into a typed ModelConfig, asserting n_layer, n_embd, n_head, n_head_kv, n_ff, rope freq/scale, rms_norm_eps
- [ ] 1.2 Implement config reader mapping GGUF metadata keys → typed fields
- [ ] 1.3 Failing test: absent/optional metadata falls back to sane defaults — no silent garbage

## 2. BPE tokenizer (engine-core)
- [ ] 2.1 Failing test: encode the fixed prompt set (`tests/fixtures/prompts.txt`) → ids match llama.cpp token stream
- [ ] 2.2 Implement `tokenizer.rs` (~300 lines, stdlib-only BPE, Qwen3 family) following llama.cpp BPE merge/split semantics
- [ ] 2.3 Round-trip decode(encode(t)) == t across ≥10 prompts

## 3. Golden dump harness (pinned llama.cpp)
- [ ] 3.1 Add `llama-cli -m <fixture> -p "<prompt>" --temp 0 --seed 42` runner under `tools/` (dump mode writes once)
- [ ] 3.2 Generate artifacts ONCE into `tests/fixtures/golden/`: metadata.json + per-layer activations + teacher-forced logits (≥10 prompts)
- [ ] 3.3 Reproducible: re-dump is byte-identical unless fixture changed (commit goldens as immutable)

## 4. Gate
- [ ] 4.1 Round-trip == llama token stream on 20 prompts
- [ ] 4.2 CI consumes `tests/fixtures/golden/` WITHOUT llama.cpp installed
- [ ] 4.3 Gate 6.1: round-trip green + goldens committed + full suite green