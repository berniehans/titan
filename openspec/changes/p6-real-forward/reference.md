# Phase 6 reference: llama.cpp pinned build

## Pinned reference
- **Repo:** https://github.com/ggml-org/llama.cpp
- **Commit:** `cb1adf8` ("server : handle failures to restore host cache (#17078)")
- **Local clone:** `%LOCALAPPDATA%/llama.cpp` (shallow clone)
- **Build:** CPU-only, Release, `-DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF`
- **Binaries:** `%LOCALAPPDATA%/llama.cpp/build/bin/Release/` (llama-cli, llama-bench, ...)

## Fixture verification (2026-08-23)
- Model: `testdata/Qwen3-0.6B-Q4_K_M.gguf` loads and generates correctly.
- Command template for golden runs:
  ```
  llama-cli.exe -m <fixture> -p "<prompt>" -n <n> --temp 0 --seed 42 --no-warmup
  ```
- Sanity run "Hello" → coherent Qwen3 chat output (`<think>` trace + response). ✔

## Golden dump plan (change 6.1)
- Teacher-forced logits for ≥10 fixed prompts via a small tool or `--logits` output mode.
- Per-layer activations (layer 0 / 1 / N-1) via eval-callback (`llama-eval-callback.exe`) if sufficient; else add a tiny dump patch in a SEPARATE branch of the pinned clone — never upstream master.
- Goldens committed to `tests/fixtures/golden/` (compressed .npz-style or raw f32 with JSON manifest).

## Porting sources per kernel (traceability)
| Kernel change | Upstream source |
|---|---|
| gemv_q4k.cu (6.3) | `ggml/src/ggml-cuda/vecdotq.cuh::vec_dot_q4_K_q8_K` @ cb1adf8 |
| norm_rope.cu RMSNorm/RoPE/SwiGLU (6.4) | `ggml/src/ggml.c::ggml_compute_forward_rms_norm`, rope refs; Qwen3 uses NeoX-style partial rotary — verify against `ggml/src/ggml-cuda/rope.cu` |
| paged_attention.cu (6.5) | vLLM `csrc/pos_encoding_kernels.cu` + paged attention v1/v2 decode kernel (Apache-2.0) |
