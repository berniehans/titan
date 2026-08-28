# Proposal: Speculative Decoding Multi-Draft Acceleration (Llama 3.2 1B -> 3B)

## Problem & Motivation
Llama 3.2 3B achieves 68.9 tok/s in standalone resident decode on the RTX 3060 Laptop GPU due to memory bus bandwidth saturation (1.81 GB weights loaded per token). In contrast, the smaller draft model Llama 3.2 1B achieves 170.4 tok/s (0.75 GB weights per token). 

By executing speculative decoding with Llama 3.2 1B generating =4$ proposed tokens in GPU memory and Llama 3.2 3B evaluating all $ candidates in a single parallel verification forward pass, Titan can emit an average of 2.8 - 3.4 tokens per target pass, boosting effective generation speed on Llama 3.2 3B to **120 - 140 tok/s** (surpassing llama.cpp's 93.6 tok/s).

## Proposed Changes
1. **GPU-Resident Multi-Model Loader:** Allocate both Draft (1B) and Target (3B) models concurrently in GPU VRAM (total ~2.83 GB / 6.0 GB).
2. **Batched Parallel Verification Pass:** Extend ForwardDriver::verify_speculative_batch to execute all $ candidate tokens simultaneously across all 28 layers.
3. **Tree / Fast Rollback KV-Cache Mechanism:** Update virtual BlockTable and pos_dev pointers upon acceptance/rejection without data copies.
4. **Speculative Orchestrator:** Connect draft generation loop and target verification stream via CUDA events for zero host synchronization.

## Verification
- Automated benchmark test speculative_speedup_bench.rs verifying output equality with pure target generation and measuring throughput $\ge 120\text{ tok/s}$.
