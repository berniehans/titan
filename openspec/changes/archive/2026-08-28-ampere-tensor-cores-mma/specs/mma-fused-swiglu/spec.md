# mma-fused-swiglu Delta Specification

## Purpose

Fuses Gate and Up projections with in-register SiLU elementwise multiplication using Tensor Cores to minimize DRAM read/write traffic.

## Requirements

### Requirement: Fused Dual-Projection Tensor Core Execution
The CUDA kernel subsystem SHALL provide `gemm_q4k_fused_gate_up_swiglu_mma_kernel` which simultaneously accumulates Gate and Up projection vectors in Tensor Core registers.

#### Scenario: Feed-forward network (FFN) forward step
- **WHEN** computing the intermediate activation of the SwiGLU MLP layer
- **THEN** the kernel writes only the final $\text{silu}(W_{\text{gate}} x) \cdot (W_{\text{up}} x)$ output directly to global memory without storing intermediate Gate or Up tensors.
