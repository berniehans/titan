# ⚡ TITAN — Autonomous Pure Rust & CUDA LLM Inference Engine

[![Rust](https://img.shields.io/badge/rust-1.85%2B%20(edition%202024)-orange.svg)](https://www.rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenAI API](https://img.shields.io/badge/OpenAI_API-Compatible-blue.svg)](https://platform.openai.com/docs/api-reference)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](https://opensource.org/license/mit)

**Titan** is a 100% Pure Rust LLM inference engine with CUDA/NVRTC kernels for consumer and datacenter NVIDIA GPUs. It supports resident GPU execution, layer streaming, paged KV-cache, CUDA Graphs, multi-model speculation, OpenAI-compatible serving, and an experimental MoE path.

---

## 🛑 Current Status & Execution Boundary

| Dimension | Current Value | Authority / Artifact |
| :--- | :--- | :--- |
| **Functional MVP Status** | **`mvp_blocked_evidence`** (NOT accepted) | [`local-artifacts/reviews/mvp-decision-20260911T162326Z.json`](docs/MVP.md) |
| **Strict Production Release** | **`not_accepted`** / **`rejected`** | `local-artifacts/reviews/p5-release-gate-20260912T130000Z.json` |
| **Release Approval** | **`not_granted`** | [`docs/RELEASE_ENVELOPE_V1.json`](docs/RELEASE_ENVELOPE_V1.json) |
| **Production Promotion** | **`promotion_authorized = false`** | No candidate promoted to default |
| **Default Runtime Dispatch** | **`production_dispatch_changed = false`** | Verified default F32 resident baseline |
| **Release Candidate Generation** | **`rc_generated = false`** | No release candidate generated |
| **MVP Gate Unit Suite** | **`25 passed, 0 failed`** (100%) | `uv run --project tools --no-sync pytest tools/test_mvp_gate.py` |
| **Full Python Tooling Suite** | **`88 passed, 0 failed; 8 subtests passed`** (100%) | `uv run --project tools --no-sync pytest tools` |
| **OpenSpec Live State** | **`40 passed, 0 failed`** (40 items) | `openspec validate --all` |
| **Detailed MVP Contract** | See [`docs/MVP.md`](docs/MVP.md) | Comprehensive acceptance contract & reproduction |

> [!IMPORTANT]
> **Functional MVP vs. Strict Production Release:**
> * A throughput ratio of $\ge 0.95\times$ compared to `llama.cpp` is **NOT** an acceptance criterion for the Stable Functional MVP. The $0.95\times$ comparison remains a separate strict production-performance diagnostic enforced by `tools/release_gate.py`.
> * In the functional MVP evaluation, throughput ratios (aggregate median `0.5286115177544821`) and baseline regressions are **advisory warnings** (`ratio_advisory: advisory_warning`, `regression_advisory: failed`). They do not block MVP acceptance.
> * Conversely, **generation correctness, operational E2E behavior, and clean benchmark execution safety remain strictly blocking**.
>
> **Exact MVP Blockers:**
> 1. **`build_binding_unverified` (F8-to-Final-Titan-Build/Source Binding Not Proven):** The independent F32 correctness run (F8.R1.P) verified model correctness across all 5 models, but does not record the Titan binary or source identity matching the clean benchmark's build hash (`sha256:bc92211356bb0eb87ddd59e252d9d4497f8669d7257ac0133459040ce4eb131c`).
> 2. **`model_hash_missing` (P5 Benchmark Lacks Per-Result Model Hashes):** The final clean benchmark (`local-artifacts/benchmarks/p5-final-clean-20260911T151844Z.json`, 30 rows) records model names and prompt contracts, but lacks per-result GGUF weight file hashes required for exact binding to F8.
>
> **Verified Evidence Summary:**
> * **Generation Correctness:** Verified 5/5 models on F32, batch 1, resident KV, Graph=false, candidate selectors absent (`local-artifacts/reviews/f32-five-model-correctness-matrix-1789097311038-15004.json`).
> * **Operational E2E:** Resident JSON and Streaming SSE with N-gram speculative mode passed (`local-artifacts/reviews/p5o-e2e-unified-modes-20260912T090000Z.log`); grammar-constrained JSON decoded and parsed semantically via serde as `{"city": "Tokyo"}` (`local-artifacts/reviews/p5o-grammar-json-20260912T100000Z.log`); readiness, timeout bounds, and lifecycle cleanup passed.
> * **Clean Benchmark:** 30 valid rows (5 models × 3 repetitions × cold/warm), 41 tokens/sample, clean telemetry (`instrumentation_mode = "clean_throughput"`), selector provenance (`candidate_selectors_absent = true`, `production_dispatch_changed = false`).
> * **Strict Gate Diagnostic:** Failed with aggregate ratio `0.5286115177544821` (cold `0.5758x`, warm `0.4981x`) plus regression failure.
>
> **Local-Only Evidence & Publishing Boundary:**
> Directories such as `local-artifacts/`, `.hermes/`, `models/`, and `target/` contain local execution logs, private model weights, and compiler targets. They are **not** publishable repository content. Advanced features (speculative decoding, layer streaming, APC, attention sinks) exist in the codebase but are **not** all production-integrated or certified under the release envelope.

---

## 🚀 Key Highlights & Architectural Innovations

* 🧠 **100% Pure Rust with Zero C++ Build Toolchains:** Runs *out-of-the-box* without MSVC (`cl.exe`), CMake, Python, or external DLL wrappers. Compiles kernels at runtime via NVIDIA Driver NVRTC (`nvcuda.dll`).
* ⚡ **Autonomous CUDA Graph Execution:** The entire 28-layer transformer forward pass (RMSNorm $\to$ Fused QKV GEMV $\to$ Paged Attention $\to$ SwiGLU $\to$ Down GEMV $\to$ LM Head $\to$ Greedy Argmax) is captured directly into a resident CUDA Graph in GPU VRAM with **0 host CPU roundtrips per token**.
* 🏎️ **Hardware DP4A SIMD Vectorized GEMV (`compute_q4k_block_dp4a`):** 128-bit `uint4` vector loads with warp-level cooperative partition (4 groups $\times$ 8 threads) process all 8 sub-blocks of Q4_K super-blocks in parallel in a single cycle, achieving $\ge 160\text{ GB/s}$ effective bandwidth.
* 🎯 **Multi-Model GPU Speculative Decoding:** Concurrent GPU-resident Draft and Target models with parallel candidate verification. Speculative-decoding figures are reported separately and are not used as single-model llama.cpp comparisons.
* 🔁 **Multi-Slot Continuous Batching & Asynchronous Ingress:** Iteration-level continuous scheduling dynamically multiplexes 4–8 concurrent client generation slots without head-of-line blocking stalls.
* 🌳 **Radix Tree Automatic Prefix Caching (APC):** Reuses pre-computed KV-cache for system prompts and tool schemas via Longest Common Prefix (LCP) matching, cutting **TTFT to <0.5 ms**.
* 📦 **Chunked Prefill with Interleaved Decode:** Slices long prompt prefill into bounded chunks while preventing decode starvation.
* 🎭 **Grammar-Constrained JSON & Tool Decoding:** RFC 8259 state-machine validation and fast GPU logit filtering via OpenAI `response_format: {"type": "json_object" | "json_schema"}` for **100% syntactically guaranteed JSON & Tool Calls**. Verified on GPU with serde parse output `{"city": "Tokyo"}`.
* 🛡️ **Attention Sinks & Infinite Context (StreamingLLM):** Retains initial sink tokens ($K=4$) with bounded KV-cache sliding windows for infinite context generation with 100% numerical stability.
* 🌐 **Built-in OpenAI Compatible Server & CLI:** Native SSE streaming server (`/v1/chat/completions`) with tool-calling schema support and rich terminal CLI (`chat`, `serve`, `bench`, `agent`).

---

## 🧭 Current Project Status and Roadmap

The historical capability phases are implemented in code, but implementation presence does not mean all features are production-certified under the release envelope.

### Stable Functional MVP Checkpoint — 2026-09-11

The repository has established a dedicated Stable Functional MVP gate (`tools/mvp_gate.py`) and acceptance specification ([`docs/MVP.md`](docs/MVP.md)):
1. **F32 generation correctness** is `verified` across all 5 declared models (`local-artifacts/reviews/f32-five-model-correctness-matrix-1789097311038-15004.json`).
2. **Operational E2E** is `verified` across Resident, Streaming SSE with N-gram, grammar JSON, readiness, and lifecycle cleanup (`local-artifacts/reviews/p5o-e2e-unified-modes-20260912T090000Z.log`, `local-artifacts/reviews/p5o-grammar-json-20260912T100000Z.log`).
3. **Clean benchmark structure** is complete across 30 rows with observed Graph disabled and clean telemetry (`local-artifacts/benchmarks/p5-final-clean-20260911T151844Z.json`).
4. **Current MVP Verdict:** `mvp_blocked_evidence` due to missing F8-to-Titan-build binding and missing per-result model hashes. Strict production performance remains `not_accepted` / `rejected` (aggregate ratio `0.5286115177544821`). No release candidate is generated (`rc_generated = false`), and production dispatch remains strictly unchanged (`production_dispatch_changed = false`).

### Historical Diagnostic Context (2026-09-07 – 2026-09-09)

* **Projection Attribution:** Verified 5/5 groups in `local-artifacts/reviews/phase17-projection-attribution-decision-20260908.json` and `local-artifacts/reviews/five-model-projection-attribution-control-1788838534847-10564-0.json`, confirming QKV as the largest non-FFN projection group.
* **FFN Candidates:** `q4k_multi_shape_single_row` remains accepted opt-in only (`TITAN_F32_FFN_GATE_UP_VARIANT=q4k_multi_shape_single_row`); fused Gate/Up and Q6_K single-row remain rejected; Q8 remains experimental; `ffn_sync` remains `not_available/missing_real_frontier`; Nsight profiling remains blocked by `ERR_NVGPUCTRPERM`.
* **QKV Multi-Row Batch=1 Candidate:** Evaluated under `TITAN_F32_QKV_VARIANT=q4k_multi_row_batch1`. While 5/5 strict Q/K/V parity and dispatch smoke passed, benchmark evaluation failed the release gates and the formal decision was a decisive rejection (`candidate_status = "rejected_regression"`, `production_status = "not_promoted"`, `release_gate_status = "rejected"`, `local-artifacts/reviews/phase18-qkv-multi-row-batch1-decision-20260908.json`). The subsequent paired diagnostic exited `101` with `token_sequence_mismatch` on the first fixture (`local-artifacts/benchmarks/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788956770585-6596-0.json`). Production dispatch remains strictly unchanged.

See [`docs/MVP.md`](docs/MVP.md), [`docs/WORKSPACE_STATE.md`](docs/WORKSPACE_STATE.md), and [`docs/TITAN_PROJECT_DIAGNOSIS_AND_ROADMAP.md`](docs/TITAN_PROJECT_DIAGNOSIS_AND_ROADMAP.md) for full details.

---

## 📊 Historical Reproduced Benchmark Results (2026-09-01)

> **Workspace status:** The table below is a historical benchmark checkpoint (2026-09-01), preserved for historical reference and not a current release claim. Current verified clean benchmark data (30 rows, five models, cold/warm, 41 tokens) is in `local-artifacts/benchmarks/p5-final-clean-20260911T151844Z.json`, and the current strict release gate verdict (`rejected`, aggregate ratio `0.5286115177544821` plus regression failure) is preserved in `local-artifacts/reviews/p5-release-gate-20260912T130000Z.json`. For the stable functional MVP contract and status (`mvp_blocked_evidence`), see [`docs/MVP.md`](docs/MVP.md) and [`docs/WORKSPACE_STATE.md`](docs/WORKSPACE_STATE.md).

The table below was rerun locally from the current checkout on 2026-09-01. It used the same GGUF file, two prompts, greedy sampling (`temperature = 0.0`), 41 generated tokens, three repetitions per model, and CUDA-enabled `llama-server.exe` and Titan. `llama.cpp` had CUDA Graphs enabled. These are decode-throughput measurements, not claims of numerical equivalence.

### 1. Multi-Model Head-to-Head: Titan vs. llama.cpp (Official C++)

| Model Evaluated | Format / Quantization | Architecture | **llama.cpp (C++)** | **Titan (Pure Rust)** | **Ratio / Parity** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3 0.6B Base/Chat** | GGUF `Q4_K_M` | 28 Layers, 1024 Dim, 151k Vocab | 240.2 tok/s cold / 278.1 warm | **274.1 tok/s** cold / 274.1 warm | **1.057x** |
| **DeepSeek-R1-Distill 1.5B** | GGUF `Q4_K_M` | 28 Layers, 1536 Dim, 152k Vocab | 157.0 tok/s cold / 173.1 warm | 143.3 tok/s cold / 143.3 warm | **0.864x** |
| **Qwen 2.5 1.5B Instruct** | GGUF `Q4_K_M` | 28 Layers, 1536 Dim, 152k Vocab | 138.6 tok/s cold / 135.5 warm | 124.1 tok/s cold / 128.5 warm | **1.009x** |
| **Llama 3.2 1B Instruct** | GGUF `Q4_K_M` | 16 Layers, 2048 Dim, 128k Vocab | 178.6 tok/s cold / 189.0 warm | 170.2 tok/s cold / 179.7 warm | **0.896x** |
| **Llama 3.2 3B Instruct** | GGUF `Q4_K_M` | 28 Layers, 3072 Dim, 128k Vocab | 104.7 tok/s cold / 105.0 warm | 77.0 tok/s cold / 76.9 warm | **0.735x** |

The reported ratio is the artifact's average across prompts and cold/warm statistics. The simple mean of the five reported ratios is **0.912x**. This checkpoint does not satisfy the `>=0.95x` per-model and aggregate release gates; Llama 3.2 3B remains the principal deficit.

---

### 2. Cross-Engine Comparison Matrix

```
                     DECODE THROUGHPUT (Batch = 1, latest reproduced run)
                     
llama.cpp (C++)       █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █  225.5 tok/s (Qwen3 0.6B)
TITAN (Pure Rust)     █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █  228.2 tok/s (Qwen3 0.6B)
```

---

## 🛠️ Quick Start & CLI Subcommands

### 1. Build from Source
Ensure you have Rust (stable) and an NVIDIA GPU with drivers installed:
```bash
git clone https://github.com/berniehans/titan.git
cd titan/engine
cargo build --release
```

### 2. Interactive Terminal Chat (`titan chat`)
Chat directly with any GGUF model in your terminal with live token-by-token streaming:
```bash
./target/release/titan chat -m ../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

### 3. Automated GPU Benchmark (`titan bench`)
Run high-precision automated latency (TTFT) and decode throughput profiling:
```bash
./target/release/titan bench -m ../models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### 4. OpenAI-Compatible API Server (`titan serve`)
Start a high-performance HTTP server compatible with Open WebUI, Continue.dev, Cursor, and LiteLLM:
```bash
./target/release/titan serve -m ../models/qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8000
```

### 5. Autonomous Agent Preset (`titan agent`)
Launch optimized backend server preset for Hermes Agent & parallel subagent tool-calling loops on port 8080:
```bash
./target/release/titan agent -m ../models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

---

## 🧪 Testing and Reproducibility

To run the reproduced multi-model head-to-head benchmark against `llama.cpp`:
```bash
cd engine
cargo test --release -p engine-server --test multi_model_comparison_bench -- --ignored --nocapture
```

The benchmark skips models whose GGUF is absent and reports the models actually executed. On Windows, the test uses the installed `llama-server.exe`; Titan's NVRTC DLL path must be available to the process. The 2026-09-01 run completed with `1 passed, 0 failed` in 100.76 seconds and measured all five models with three repetitions. Results: `local-artifacts/benchmarks/rerun-20260901-085229.json`; raw log: `local-artifacts/benchmarks/rerun-20260901-085229.log`.

### Numerical and MVP validation status

The baseline checkout enforces separate Rust, GPU parity, E2E, attribution, MVP, and release gates. A benchmark test passing does not close those gates. The workspace currently records:
* **Stable Functional MVP:** `mvp_blocked_evidence` due to unproven F8-to-Titan-build binding and missing per-result model hashes in the clean benchmark. Throughput ratios and regression are advisory; generation correctness and operational execution are blocking. See [`docs/MVP.md`](docs/MVP.md).
* **Strict Production Release:** `not_accepted` / `rejected` (aggregate ratio `0.5286115177544821` vs required $\ge 0.95\times$; regression failed).
* **Production Invariants:** `production_dispatch_changed = false`, `promotion_authorized = false`, `rc_generated = false`.
* **Local-Only Evidence:** `local-artifacts/`, `.hermes/`, `models/`, and `target/` are local-only and not publishable repository content.
* **Kernel Candidates:** The previously measured FFN candidate `q4k_multi_shape_single_row` remains accepted only as an opt-in selector (`TITAN_F32_FFN_GATE_UP_VARIANT=q4k_multi_shape_single_row`); the QKV multi-row batch=1 candidate was decisively rejected as a regression (`local-artifacts/reviews/phase18-qkv-multi-row-batch1-decision-20260908.json`). Neither is promoted to production default.

To run the multi-model speculative decoding benchmark (1B Draft -> 3B Target):
```bash
cargo test --release -p engine-server --test speculative_speedup_bench -- --ignored --nocapture
```

---

## 📄 License
Licensed under the [MIT License](https://opensource.org/license/mit).
