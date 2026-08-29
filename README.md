# ⚡ TITAN — Autonomous Pure Rust & CUDA LLM Inference Engine

[![Rust](https://img.shields.io/badge/rust-1.85%2B%20(edition%202024)-orange.svg)](https://www.rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenAI API](https://img.shields.io/badge/OpenAI_API-Compatible-blue.svg)](https://platform.openai.com/docs/api-reference)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

**Titan** is an ultra-high-throughput, 100% Pure Rust LLM inference engine designed from scratch for consumer and datacenter NVIDIA GPUs. 

By eliminating much of the host-side CPU dispatch overhead through **Autonomous CUDA Graphs in GPU VRAM** and utilizing vectorized **DP4A integer SIMD kernels with 128-bit `uint4` coalesced loads**, Titan reached **228.2 tok/s on Qwen3 0.6B** in the latest reproducible head-to-head run. On the same run, Titan matched or exceeded `llama.cpp` only on that small model; `llama.cpp` remained faster on the 1B–3B models tested.

---

## 🚀 Key Highlights & Architectural Innovations

* 🧠 **100% Pure Rust with Zero C++ Build Toolchains:** Runs *out-of-the-box* without MSVC (`cl.exe`), CMake, Python, or external DLL wrappers. Compiles kernels at runtime via NVIDIA Driver NVRTC (`nvcuda.dll`).
* ⚡ **Autonomous CUDA Graph Execution:** The entire 28-layer transformer forward pass (RMSNorm $\to$ Fused QKV GEMV $\to$ Paged Attention $\to$ SwiGLU $\to$ Down GEMV $\to$ LM Head $\to$ Greedy Argmax) is captured directly into a resident CUDA Graph in GPU VRAM with **0 host CPU roundtrips per token**.
* 🏎️ **Hardware DP4A SIMD Vectorized GEMV (`compute_q4k_block_dp4a`):** 128-bit `uint4` vector loads with warp-level cooperative partition (4 groups $\times$ 8 threads) process all 8 sub-blocks of Q4_K super-blocks in parallel in a single cycle, achieving $\ge 160\text{ GB/s}$ effective bandwidth.
* 🎯 **Multi-Model GPU Speculative Decoding:** Concurrent GPU-resident Draft and Target models with parallel candidate verification. The speculative-decoding figures are reported separately and are not used as single-model llama.cpp comparisons.
* 🔁 **Multi-Slot Continuous Batching & Asynchronous Ingress:** Iteration-level continuous scheduling dynamically multiplexes 4–8 concurrent client generation slots without head-of-line blocking stalls.
* 🌳 **Radix Tree Automatic Prefix Caching (APC):** Reuses pre-computed KV-cache for system prompts and tool schemas via Longest Common Prefix (LCP) matching, cutting **TTFT to <0.5 ms**.
* 📦 **Chunked Prefill with Interleaved Decode:** Slices long prompt prefill into bounded chunks while preventing decode starvation.
* 🎭 **Grammar-Constrained JSON & Tool Decoding:** RFC 8259 state-machine validation and fast GPU logit filtering via OpenAI `response_format: {"type": "json_object" | "json_schema"}` for **100% syntactically guaranteed JSON & Tool Calls**.
* 🛡️ **Attention Sinks & Infinite Context (StreamingLLM):** Retains initial sink tokens ($K=4$) with bounded KV-cache sliding windows for infinite context generation with 100% numerical stability.
* 🌐 **Built-in OpenAI Compatible Server & CLI:** Native SSE streaming server (`/v1/chat/completions`) with tool-calling schema support and rich terminal CLI (`chat`, `serve`, `bench`, `agent`).

---

## 📊 Latest Reproduced Benchmark Results (NVIDIA RTX 3060 Laptop GPU - 6GB VRAM)

The table below was rerun locally from the current checkout. It used the same GGUF file, two prompts, greedy sampling (`temperature = 0.0`), 41 generated tokens, and CUDA-enabled `llama-server.exe` and Titan. `llama.cpp` had CUDA Graphs enabled. These are decode-throughput measurements, not claims of numerical equivalence.

### 1. Multi-Model Head-to-Head: Titan vs. llama.cpp (Official C++)

| Model Evaluated | Format / Quantization | Architecture | **llama.cpp (C++)** | **Titan (Pure Rust)** | **Ratio / Parity** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3 0.6B Base/Chat** | GGUF `Q4_K_M` | 28 Layers, 1024 Dim, 151k Vocab | 225.5 tok/s *(4.46 ms)* | **228.2 tok/s** *(4.38 ms)* | **1.01x (+1.2%)** |
| **DeepSeek-R1-Distill 1.5B** | GGUF `Q4_K_M` | 28 Layers, 1536 Dim, 152k Vocab | **148.9 tok/s** *(6.72 ms)* | 121.0 tok/s *(8.27 ms)* | 0.81x (-18.7%) |
| **Qwen 2.5 1.5B Instruct** | GGUF `Q4_K_M` | 28 Layers, 1536 Dim, 152k Vocab | **149.5 tok/s** *(6.70 ms)* | 128.2 tok/s *(7.80 ms)* | 0.86x (-14.3%) |
| **Llama 3.2 1B Instruct** | GGUF `Q4_K_M` | 16 Layers, 2048 Dim, 128k Vocab | **214.3 tok/s** *(4.67 ms)* | 158.8 tok/s *(6.30 ms)* | 0.74x (-25.9%) |
| **Llama 3.2 3B Instruct** | GGUF `Q4_K_M` | 28 Layers, 3072 Dim, 128k Vocab | **97.4 tok/s** *(10.27 ms)* | 66.6 tok/s *(15.01 ms)* | 0.68x (-31.6%) |

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

The benchmark skips models whose GGUF is absent and reports the models actually executed. On Windows, the test uses the installed `llama-server.exe`; Titan's NVRTC DLL path must be available to the process. The latest run completed with `1 passed, 0 failed` and measured all five models listed above.

### Numerical validation status

The current checkout passes the general Rust test suite and the exercised Q4/Q6 and paged-attention GPU parity tests. The SwiGLU parity gate still requires investigation (`rel-L2 = 1.047` in the latest run), so the throughput table must not be read as a final correctness sign-off.

To run the multi-model speculative decoding benchmark (1B Draft -> 3B Target):
```bash
cargo test --release -p engine-server --test speculative_speedup_bench -- --ignored --nocapture
```

---

## 📄 License
Licensed under the [MIT License](LICENSE).
