# Tasks: f4-paged-kvcache

> Execute via bot coder. Strict TDD. One commit per task group.

## 1. CPU reference (engine-kvcache)
- [x] 1.1 Failing tests: create pool (n_blocks, block_size=16 tokens, head_dim), append token KV, read back exact floats
- [x] 1.2 Implement CPU paged cache: fixed-size blocks in a flat buffer, block table Vec<Vec<u32>>, O(1) free-list allocator, typed errors (pool exhausted)
- [x] 1.3 Test multi-append spanning >1 block (logical contiguity over scattered physical blocks)

## 2. GPU kernels (engine-cuda, NVRTC pattern like dequant_q4k.cu)
- [x] 2.1 Failing GPU test (#[ignore]): append_kv kernel writes K/V rows into device-side blocks given a block table uploaded to device
- [x] 2.2 Implement append_kv + paged-read (gather) kernels; RAII wrapper mirroring Q4KDequantizer style
- [x] 2.3 Verify PASS on local GPU (PATH trick: %LOCALAPPDATA%/Temp has nvrtc64_120_0.dll)

## 3. Parity gate
- [x] 3.1 Failing test (#[ignore]): deterministic pseudo-random K/V data (seeded xorshift), CPU vs GPU block-by-block, max abs error < 0.01/elem
- [x] 3.2 Verify PASS locally — measured max abs err = 0.0/elem (bit-exact), 22 tokens across 4 phys blocks

## 4. Budget enforcement
- [x] 4.1 Tests: allocate blocks until pool exhausts → typed error; free returns block to pool and realloc succeeds
- [x] 4.2 Verify green

## 5. Gate
- [x] 5.1 Full suite green (fmt, clippy -D warnings, CPU tests, GPU --ignored)
- [x] 5.2 Record measurable numbers in docs/BENCHMARKS.md — parity 0.0/elem under Phase 4 row; throughput N/A (correctness-only phase)