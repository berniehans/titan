# Change: Config + BPE tokenizer + llama.cpp golden harness (Phase 6.1)

## Why
Phase 6 makes inference real, but real inference needs three foundations first: typed hyperparameters (so downstream kernel math is stable), a working tokenizer (the stub used none), and golden artifacts from the pinned llama.cpp reference (so every later parity gate has a trusted, reproducible anchor). This change builds all three and commits the goldens as fixtures.

## What Changes
- Typed `ModelConfig` derived from GGUF metadata: n_layer, n_embd, n_head, n_head_kv (GQA), n_ff, rope_freq_base / rope_freq_scale, rms_norm_eps, vocab_size, per-group quant format, rope dims. Reuses the existing GGUF reader.
- Real BPE tokenizer in `engine-core` translated from llama.cpp BPE (~300 lines, stdlib-only, no heavy deps). Qwen3 BPE family (the fixture is Qwen3-0.6B-Q4_K_M).
- Golden dump harness (small tool): runs pinned `llama-cli` (temp=0, greedy, seed 42) on the fixture and exports artifacts ONCE into `tests/fixtures/golden/`:
  - `metadata.json` — config ground truth from GGUF,
  - per-layer activations (layer 0 / 1 / N-1),
  - teacher-forced logits for ≥10 fixed prompts.
- Round-trip tokenizer == llama.cpp token stream; goldens committed and reproducible; CI consumes goldens WITHOUT llama.cpp installed.
- If `llama-eval-callback.exe` cannot dump layer activations, add a tiny dump patch on a SEPARATE branch of the pinned clone — never on upstream master.

## Non-goals
- No kernel work in this change (CPU-only).
- No inference compute beyond tokenizer + golden dump.
- No tokenizer dependencies (plain BPE is allowed).

## Impact
- **Affected code:** `engine-io` GGUF metadata → config, `engine-core/src/tokenizer.rs`, golden dump tool
- **Gate:** round-trip tokenizer == llama.cpp token stream on 20 prompts; goldens committed and reproducible; CI consumes goldens without llama.cpp installed

## Tasks (summary — details in tasks.md)
1. GGUF metadata → typed ModelConfig + tests
2. BPE tokenizer (llama.cpp-pattern, Qwen3) + tests
3. Golden dump harness (pinned llama.cpp) → fixtures
4. Round-trip gate + CI consuming goldens

## Environment notes
- NVRTC via `%LOCALAPPDATA%/Temp` PATH trick; cargo GPU tests `#[ignore]` (not used this change).
- llama.cpp reference pinned `cb1adf8`, binaries at `%LOCALAPPDATA%/llama.cpp/build/bin/Release/`.