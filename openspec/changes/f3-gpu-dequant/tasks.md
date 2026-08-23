# Tasks: f3-gpu-dequant

> Execute via bot coder (`hermes -p coder`). Strict TDD. One commit per task.

## 1. Layout + CPU reference (engine-core)
- [ ] 1.1 Failing test: parse a hand-built block_q4_K buffer (known values) through CPU reference dequant → exact expected floats
- [ ] 1.2 Implement `dequant.rs`: Q4_K_M super-block layout (8x32, 6-bit scales/mins, fp16 d/dmin), `dequant_q4k_cpu(src: &[u8], n_elements) -> Vec<f32>`
- [ ] 1.3 Cross-check reference against fixture tensors from engine-io (metadata gives rope freqs etc.; verify element count matches tensor dims)

## 2. GPU kernel (engine-cuda)
- [x] 2.1 Failing test (#[ignore]): launch dequant kernel on synthetic Q4_K_M device buffer, read back floats
- [x] 2.2 Implement kernel (CUDA C compiled at build time via nvcc/nvrtc or precompiled PTX) + RAII-safe launch wrapper
- [x] 2.3 Verify PASS on local GPU

## 3. Parity gate
- [x] 3.1 Failing test (#[ignore]): random Q4_K_M data (deterministic seed) → GPU vs CPU reference block-by-block, assert max abs error < 0.01
- [x] 3.2 Verify PASS on local GPU

## 4. Pipeline integration
- [x] 4.1 Failing test (#[ignore]): Pipeline::run with dequant enabled produces dequantized outputs for each layer
- [x] 4.2 Wire kernel into compute stage behind copy_done event
- [x] 4.3 Full suite green locally

## 5. Gate
- [x] 5.1 Re-run pipeline bench with real compute: log overlap improvement
- [x] 5.2 Gate F3: parity < 0.01/elem, all tests green, clippy clean
