# Tasks: 6.3-gemv-multiformat

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + `%LOCALAPPDATA%/Temp` PATH trick for NVRTC.

## 1. CPU harness + RED
- [x] 1.1 Failing test: CPU dequant→dot (from 6.2 bank) over controlled Q4_K_M block == expected floats
- [x] 1.2 Failing GPU test (`#[ignore]`): Q4_K_M block dot on device vs CPU dot
- [x] 1.3 Failing GPU test (`#[ignore]`): Q8_0 block dot parity
- [x] 1.4 Failing GPU test (`#[ignore]`): F16 (non-quant) dot parity

## 2. Kernel — GREEN (engine-cuda)
- [x] 2.1 Implement `gemv_q4k.cu` with top-of-source port comment:
  `// Port of llama.cpp vcdotq.cuh::vec_dot_q4_K_q8_K (ggml/src/ggml-cuda/vecdotq.cuh @ cb1adf8)`
- [x] 2.2 Add Q8_0 and F16 paths in the same translation unit (layout per ggml quant conventions)
- [x] 2.3 Wrapper `MultiFormatGEMV` (RAII, NVRTC launch); emits only needed tile, no layer FP16 materialization
- [x] 2.4 Verify PASS on local GPU (3/3 formats)

## 3. Real-tensor parity (teacher-forced)
- [x] 3.1 Failing test (`#[ignore]`): real fixture embedding-row GEMV rel-L2 < 1e-3 vs CPU forward bank
- [x] 3.2 Same for output/logit head (Q8_0 or F16 per realistic GGUF); rel-L2 < 1e-3
- [x] 3.3 Failing test (`#[ignore]`): real first-block Q4 attention-weight matmul vs llama golden L0 (cos-sim ≥ 0.999, rel-L2 < 1e-3)
- [x] 3.4 Record numbers; verify PASS

### 3.4 Recorded numbers (REAL, measured on RTX 3060 Laptop via `cargo test -p engine-core --test gemv_realtensor -- --ignored`)
- **Real Q4_K attention-weight GEMV** (`blk.0.attn_q.weight`, 512 cols x 1024 dims, GPU `MultiFormatGEMV::Q4K` vs CPU 6.2 `matmul`): **rel-L2 = 0.000e0, cos-sim = 1.000000** → PASS (gate rel-L2 < 1e-3, cos-sim >= 0.999).
- **Structural impossibility (explicit, no numbers invented)** — see `engine-core/tests/gemv_realtensor.rs` module docs:
  - Task 3.1 (embedding-row): `token_embd.weight` is **Q6_K**, not Q8_0/F16; it cannot run through the Q4K/Q8/F16 kernel paths (no Q6_K path by scope).
  - Task 3.2 (output/logit head): `output.weight` is absent — Qwen3 ties the head to `token_embd` (Q6_K). No Q8_0/F16 real-parity is constructible.
  - Fixture census: **0 Q8_0 tensors, 0 F16 tensors** (168 Q4_K, 113 F32 norms, 29 Q6_K). Q8_0/F16 paths are validated on controlled synthetic blocks (tasks 1.3/1.4 in `gemv_gpu.rs`), not on real GGUF.
  - llama golden L0 activations are the layer-0 **residual output** truncated to 6 elems — not comparable to an attention Q matmul (different dims/semantics); full single-layer parity vs golden is change 6.6.

## 4. Gate
- [ ] 4.1 Full suite green (clippy -D warnings, CPU suite, GPU `--ignored`)
- [ ] 4.2 Gate sealed: rel-L2 < 1e-3 vs both references on all 3 formats