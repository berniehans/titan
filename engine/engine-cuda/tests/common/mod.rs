//! Shared test helpers for engine-cuda integration and parity tests.

#![allow(dead_code)]

/// Initializes and preflights the CUDA/NVRTC runtime before cudarc is touched.
pub fn initialize_cuda() {
    let path = engine_cuda::initialize_cuda_runtime()
        .unwrap_or_else(|error| panic!("CUDA test initialization failed: {error}"));
    if !path.as_os_str().is_empty() {
        println!("CUDA test runtime: {}", path.display());
    }
}

/// Deterministic xorshift32 PRNG for test vector generation.
pub struct Xorshift(pub u32);

impl Xorshift {
    pub fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0x6D2B79F5 } else { seed })
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform float in [-1.0, 1.0).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64 + 1.0) * 2.0 - 1.0) as f32
    }
}

/// Serializes a slice of `f32` to little-endian bytes.
pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Serializes a slice of `u32` to little-endian bytes.
pub fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|u| u.to_le_bytes()).collect()
}

/// Deserializes a byte slice to a `Vec<f32>` (little-endian).
pub fn bytes_f32(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Computes cosine similarity between two float vectors in fp64.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine input lengths must match");
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x64 = x as f64;
        let y64 = y as f64;
        dot += x64 * y64;
        norm_a += x64 * x64;
        norm_b += y64 * y64;
    }
    assert!(norm_a > 0.0 && norm_b > 0.0, "zero norm vector in cosine");
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Relative L2 error `||a - b||_2 / ||b||_2` in fp64.
pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "rel_l2 input lengths must match");
    let mut diff_norm = 0.0f64;
    let mut ref_norm = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = (x - y) as f64;
        let y64 = y as f64;
        diff_norm += diff * diff;
        ref_norm += y64 * y64;
    }
    diff_norm.sqrt() / ref_norm.sqrt()
}

/// Number of floats in a single KV row: `n_head_kv * head_dim`.
pub fn row_len_of(n_head_kv: usize, head_dim: usize) -> usize {
    n_head_kv * head_dim
}

/// Number of floats in a physical block: `block_tokens * 2 * row_len`.
pub fn floats_per_block_of(block_tokens: usize, n_head_kv: usize, head_dim: usize) -> usize {
    block_tokens * 2 * row_len_of(n_head_kv, head_dim)
}

/// Populate a paged pool deterministically based on logical token order.
pub fn fill_pool_paged(
    pool: &mut [f32],
    block_table: &[u32],
    block_tokens: usize,
    n_head_kv: usize,
    head_dim: usize,
    seed: u32,
    n_tokens: usize,
) {
    let row_len = row_len_of(n_head_kv, head_dim);
    let floats_per_token = 2 * row_len;
    let floats_per_block = block_tokens * floats_per_token;
    let mut rng = Xorshift::new(seed);

    for t in 0..n_tokens {
        let block_idx = t / block_tokens;
        let slot = t % block_tokens;
        let phys = block_table[block_idx] as usize;
        let base = phys * floats_per_block + slot * floats_per_token;

        // Key row: hk in 0..n_head_kv
        for hk in 0..n_head_kv {
            let k_offset = base + hk * head_dim;
            for d in 0..head_dim {
                pool[k_offset + d] = rng.next_f32();
            }
        }
        // Value row: hk in 0..n_head_kv
        for hk in 0..n_head_kv {
            let v_offset = base + row_len + hk * head_dim;
            for d in 0..head_dim {
                pool[v_offset + d] = rng.next_f32();
            }
        }
    }
}
