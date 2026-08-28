# Proposal — Agent-Centric Runtime Optimizations for Constrained VRAM (6 GB)

## 1. Motivation & Background

Autonomous AI agents (such as **Hermes Agent**, AutoGen, and tool-use loop orchestrators) exhibit radically distinct inference traffic patterns compared to single-turn conversational chatbots:
1. **High Prefix Redundancy:** In multi-turn tool calling, 80%–95% of the prompt context (system instructions, tool definitions, environment state, and prior message history) remains static across turns. Standard inference re-prefills this entire prefix on every interaction, causing severe **Time-To-First-Token (TTFT)** penalties.
2. **Branching & Speculative Exploration:** Modern reasoning frameworks (e.g. Tree-of-Thoughts, subagent delegation, beam rollouts) fork conversation histories into divergent paths. Allocating duplicate KV-cache buffers quickly leads to catastrophic Out-Of-Memory (OOM) failures on 6 GB VRAM devices.
3. **Structured Output & Tool Reliability:** Unconstrained generation frequently produces invalid JSON schemas or malformed function calls, causing agent loop crashes and retry loops. Traditional CPU-side grammar filtering introduces severe host-device synchronization latency.
4. **Context Saturation & Degradation:** Long agent sessions easily exceed 8k–16k tokens, causing VRAM exhaustion on laptop GPUs and degradation of initial prompt attention.

This proposal introduces a comprehensive, zero-copy, grammar-constrained agent runtime for the **Titan** inference engine on **NVIDIA RTX 3060 Laptop (6 GB VRAM)**.

---

## 2. Proposed Architectural Enhancements

### Pillar 1: Automatic Prefix Caching (APC) via RadixTree (`engine-kvcache`)
* Replaces flat sequence allocation with a thread-safe `RadixTree` (Trie) over token sequences where nodes point to physical block chains (`Vec<PhysicalBlockId>`).
* Computes **Longest Common Prefix (LCP)** upon incoming requests, skipping prefill computation for matched blocks and cutting TTFT from hundreds of milliseconds to under **0.5 ms**.
* Implements tree-level LRU node eviction with strict **pinned protection** (`is_pinned = true`) for immutable System Prompts and Tool Declarations to avoid eviction thrashing and system prompt leakage.

### Pillar 2: Zero-Copy State Forking with Copy-on-Write (`engine-kvcache`)
* Implements atomic sequence branching using `Arc<PhysicalBlock>` and reference-counted `BlockTable` entries.
* Multiple concurrent reasoning branches share underlying physical memory pages. Physical pages are only cloned (*Copy-on-Write*) when a branch mutates or appends new tokens to a shared block.

### Pillar 3: Overlapped Logit Bitmasking / Grammar Decoding (`engine-cuda` & `engine-core`)
* Implements an asynchronous dual-pipeline for structured decoding (JSON Schema, Tool Call regexes, EBNF grammars):
  - **Host Stage:** The CPU grammar parser computes the valid token bitmask (packed `u32` bitfields) for step $t+1$ concurrently while the GPU executes the forward pass for step $t$.
  - **Device Stage:** A high-throughput CUDA kernel (`apply_logit_mask_kernel`) applies the bitmask directly to the logits tensor in GPU VRAM (setting invalid token logits to $-\infty$) before sampling, with **zero full-vocabulary host synchronization**.

### Pillar 4: Attention Sinks & Protected Paged Sliding Window (`engine-cuda` & `engine-kvcache`)
* Updates `paged_attention_decode_kernel` and the KV scheduler with **StreamingLLM / Attention Sinks** support:
  - Preserves initial $K=4$ attention sink tokens and all pinned system blocks.
  - Applies a rolling sliding window over intermediate tool execution outputs, evicting or offloading aged KV blocks to `PinnedHostMemory` once VRAM consumption crosses the safety guard threshold (`vram_guard`).

---

## 3. Success Metrics & Verification Gates

| Target Metric | Baseline (Titan v0.5.0) | Target with Agent Runtime | Verification Criteria |
| :--- | :--- | :--- | :--- |
| **TTFT on Prefix Match (2k tokens)** | ~180 ms | **< 1.0 ms (Near Zero)** | Automated Radix benchmark test with 95% prefix overlap. |
| **Tool Calling JSON Validity** | ~92% (dependent on model) | **100.0% Bit-Exact Valid** | 100 consecutive tool invocations tested with JSON schema bitmasking. |
| **Grammar Masking Overhead** | N/A (unconstrained) | **< 1.0% decode latency** | Overlapped CPU bitmask generation vs bare GPU decode. |
| **Max Concurrent Agent Branches (3B)**| 1–2 (OOM limit) | **$\ge 8$ concurrent forks** | Zero-copy CoW tree test with divergent subagent branches. |
| **VRAM Stability over Long Runs (16k)**| OOM at >6k tokens | **Bounded $< 4.8$ GB VRAM** | Paged sliding window + attention sinks long-context stress test. |

---

## 4. Compatibility & Architecture Invariants

* **Layer-Streaming & Resident Engine:** Fully compatible with both resident models (e.g. Qwen 2.5 1.5B, Llama 3.2 1B/3B) and PCIe layer-streaming out-of-core pipelines.
* **CUDA Graphs:** Bitmasking kernel executes either inside the captured CUDA graph or immediately preceding the argmax/sampling reduction node.
* **Constitutional Conformance:** All GPU allocations remain governed by `DeviceBuffer` RAII; zero heap allocations in the critical generation loop; 100% test-driven development (TDD).

---

## 5. Explicit Non-Goals

1. **In-GPU Dynamic Grammar Parsing:** Compiling full CFG grammars inside GPU kernels (parsing runs on host CPU in parallel with GPU GEMV).
2. **Multi-Node Distributed KV Caching:** Distributed network caching across machines (restricted to local workstation / mobile RTX 3060).
3. **Weight Fine-Tuning or LoRA Training:** Engine remains focused strictly on low-latency, high-throughput forward inference.
