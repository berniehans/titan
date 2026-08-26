# Change: Phase 6 — Real forward pass (master plan)

## Why
Phases 0-5 built verified infrastructure (loader, streaming pipeline, GPU dequant, paged KV cache, SSE server with batching) but the forward pass is a deterministic placeholder (`stub_next_token`). Phase 6 makes inference real. This proposal is the master plan; each sub-change below gets its own `openspec/changes/<id>/` with detailed tasks.

## Porting policy (do not reinvent the wheel)
The arithmetic of transformer inference is commodity code, proven in production. Titan's identity is the layer-streaming architecture, NOT kernel arithmetic. Therefore:

- **Kernels are ports, not inventions.** Each CUDA kernel translates proven C/CUDA from:
  - **llama.cpp** (MIT): `vec_dot_q4_K_q8_K`, `dequantize_q4_K` scale/min constants and accumulation order (`ggml/src/ggml-cuda/`), RMSNorm epsilon placement, RoPE convention for Qwen (NeoX-style partial rotary), SwiGLU semantics (`ggml.c`).
  - **vLLM** (Apache-2.0): PagedAttention decode kernel with online softmax over block table (`csrc/pos_encoding_kernels.cu`, paged attention v1/v2).
- **Traceability gate**: every kernel change declares in its proposal which upstream file/function it ports (with repo + commit hash) and adds a comment `// Port of llama.cpp <func> (<file> @ <commit>)` at the top of the .cu source.
- **Reference pinning (6.1)**: clone llama.cpp at a fixed tag matching the fixture era; commit its short hash to `openspec/changes/p6-*/reference.md`. Golden dumps are generated once from that pinned version and committed as fixtures under `tests/fixtures/golden/`.
- **CPU references are written from paper formulas / ggml.c semantics, never by transliterating our own CUDA back** — avoids shared-bug blindness between twin implementations.
- Python (HF/PyTorch) is used only as readable semantic reference for model config/tokenizer; kernels port from C/CUDA sources where the accumulation order is explicit.

## Sub-changes

### 6.1 — ModelConfig + tokenizer + llama.cpp golden harness
Typed hyperparameters from GGUF metadata; real BPE tokenizer translated from llama.cpp (~300 lines, no heavy deps); harness that runs llama.cpp (pinned tag, temp=0 greedy, fixed seed) on the fixture and exports golden artifacts ONCE into `tests/fixtures/golden/`: metadata JSON, per-layer activations (layer 0, 1, N-1), teacher-forced logits for ≥10 fixed prompts.
**Gate:** round-trip tokenizer == llama.cpp token stream on 20 prompts; goldens committed and reproducible; CI consumes goldens without llama.cpp installed.

### 6.2 — CPU reference forward bank on a real-GGUF synthetic
FP32 CPU forward (RMSNorm, quantized matmul via dequant→dot, RoPE, attention, residuals, SwiGLU MLP, logits) over a minimal REAL .gguf file (same tensor naming convention, same quant formats incl. non-Q4 embedding/head). Single CPU authority, cross-validated against llama.cpp goldens at layer 0.
**Gate:** synthetic logits bit-exact vs CPU bank; bank vs llama.cpp golden L0 cos-sim ≥ 0.9999.

### 6.3 — Multi-format GEMV kernel (`gemv_q4k.cu`)
Q4_K_M warp-dequant-and-dot port of llama.cpp `vec_dot_q4_K_q8_K`, PLUS Q8_0/F16 paths for embedding/output head weights (they are rarely Q4_K_M in GGUF).
**Gate:** teacher-forced parity: rel-L2 < 1e-3 vs CPU ref AND vs llama.cpp L0 activations on real fixture tensors.

### 6.4 — Fused norm/rope/swiglu kernel (`norm_rope.cu`)
RMSNorm + residual add fused; in-place RoPE (Qwen convention); SwiGLU gating.
**Gate:** parity vs CPU twins cos-sim ≥ 0.9999; VRAM worst-case declared in proposal and guarded.

### 6.5 — PagedAttention decode kernel (`paged_attention.cu`)
Online-softmax single-pass decode over block table (port of vLLM paged attention), GQA head mapping, zero intermediate allocations, causal handling for prefill path.
**Gate:** parity vs CPU SDPA reference across scattered multi-block sequences (1..2048 tokens) cos-sim ≥ 0.9999; no runtime cudaMalloc.

### 6.6 — Single-layer parity gate
Wire one complete transformer block (norm → QKV GEMV+bias → RoPE → paged append+attention → out GEMV → residual → norm → SwiGLU → down GEMV → residual) against llama.cpp layer-0 golden activation. Debugging compound drift here is cheap; later is not.
**Gate:** cos-sim > 0.999, rel-L2 < 1e-3 vs golden L0.

### 6.7 — Full forward driver (additive beside stub)
Prefill and single-token decode as separate entry points; runs the full stack over the streamed pipeline without touching `stub_next_token`; hard per-kernel VRAM guards; cumulative drift checkpoint over ≥10 teacher-forced tokens.
**Gate:** all existing suites green (stub untouched); cumulative logits drift within tolerance vs goldens; VRAM ≤ budget every step.

### 6.8 — Swap the stub (3 sub-gates)
(1) byte-identical driver check vs goldens on fixed prompt; (2) autoregressive generation of coherent text via SSE; (3) throughput vs pre-defined baseline (stub-path ids/s measured BEFORE swap as the baseline artifact).
**Gate:** teacher-forced logit cos-sim > 0.999 vs llama.cpp goldens (NOT raw top-k match — one borderline flip collapses naive autoregressive comparison); SSE E2E green; throughput within declared target.

### 6.9 — VRAM audit + benchmarks seal
Post-integration accounting per stage (ping-pong slots, KV pool growth/token, activations, logits transfer); fill docs/BENCHMARKS.md Phase 4 deferred row and Phase 6 rows with real numbers.
**Gate:** total ≤ 5.2 GB asserted by test; BENCHMARKS updated.

## Dependency order
6.1 → 6.2 → {6.3, 6.4} → 6.5 → 6.6 → 6.7 → 6.8 → 6.9 (6.4 parallelizable with 6.3 after 6.2).

## Top risks & mitigations
1. **Cumulative numerical drift** — mitigated by per-layer goldens (6.1), single-layer gate (6.6), cumulative checkpoint (6.7) before any swap.
2. **VRAM overrun** — static allocation map, per-kernel declared worst case, guard tests in 6.4/6.5/6.7.
3. **NVRTC-only constraint** — kernels are ports of simple C loops (no cuBLAS dependency), consolidated into few translation units; compile-once-per-process pattern documented.
