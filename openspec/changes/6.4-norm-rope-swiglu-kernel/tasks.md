# Tasks: 6.4-norm-rope-swiglu-kernel

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + `%LOCALAPPDATA%/Temp` PATH trick for NVRTC.

## 1. CPU twins + RED (engine-core reference from 6.2)
- [ ] 1.1 Failing test: CPU RMSNorm+residual twin vs hand-computed vector (cos-sim ≥ 0.9999)
- [ ] 1.2 Failing test: CPU in-place RoPE twin (Qwen3 NeoX partial rotary) vs hand-computed rotation
- [ ] 1.3 Failing test: CPU SwiGLU twin vs known-gate output

## 2. Kernel — GREEN (engine-cuda)
- [ ] 2.1 Implement fused `norm_rope.cu` (RMSNorm+residual, in-place RoPE, SwiGLU, single launch) with top-of-source port comment:
  `// Port of llama.cpp ggml_compute_forward_rms_norm + rope (ggml/src/ggml.c, ggml/src/ggml-cuda/rope.cu @ cb1adf8)`
- [ ] 2.2 RAII launcher mirroring the engine-cuda wrapper style; streams through the ping-pong slot
- [ ] 2.3 Failing GPU test (`#[ignore]`): fused output vs CPU fused twin cos-sim ≥ 0.9999
- [ ] 2.4 Verify PASS on local GPU

## 3. Parity + VRAM guard
- [ ] 3.1 Failing test (`#[ignore]`): per-op parity (norm / rope / swiglu) each ≥ 0.9999
- [ ] 3.2 Failing test: declared VRAM worst-case asserted (alloc_norm_rope_total ≤ declared; declared + resident_kv + pingpong ≤ 5.2 GB)
- [ ] 3.3 Record numbers; verify PASS

## 4. Gate
- [ ] 4.1 Full suite green (CPU + GPU `--ignored`, clippy)
- [ ] 4.2 Gate sealed: parity ≥ 0.9999, VRAM bound asserted