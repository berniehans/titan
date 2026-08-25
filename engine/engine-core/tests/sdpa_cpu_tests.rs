//! Phase 6.5 group-1 CPU SDPA reference tests.
//!
//! Tests for the CPU scaled dot-product attention (SDPA) reference implementation
//! over a paged KV cache pool layout.

use engine_core::forward_cpu::sdpa_decode;

/// Deterministic xorshift32 PRNG for test vector generation.
struct Xorshift(u32);

impl Xorshift {
    fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0x6D2B79F5 } else { seed })
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        // Uniform float in [-1.0, 1.0)
        (self.next_u32() as f64 / (u32::MAX as f64 + 1.0) * 2.0 - 1.0) as f32
    }
}

/// Compute cosine similarity between two float vectors in fp64.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
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

/// Populate a paged pool deterministically based on logical token order.
fn fill_pool(
    pool: &mut [f32],
    block_table: &[u32],
    block_tokens: usize,
    n_head_kv: usize,
    head_dim: usize,
    seed: u32,
    n_tokens: usize,
) {
    let row_len = n_head_kv * head_dim;
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

// ---------------------------------------------------------------------------
// TEST 1 (1.1): sdpa_sequential_blocks_match_hand_computed
// ---------------------------------------------------------------------------
#[test]
fn sdpa_sequential_blocks_match_hand_computed() {
    let block_tokens = 2;
    let n_head = 1;
    let n_head_kv = 1;
    let head_dim = 2;
    let seq_tokens = 3;
    let block_table = [0u32, 1u32];
    let num_blocks = 2;
    let row_len = n_head_kv * head_dim;
    let floats_per_token = 2 * row_len;
    let floats_per_block = block_tokens * floats_per_token;
    let mut pool = vec![0.0f32; num_blocks * floats_per_block];

    let seed = 0x55AA;
    fill_pool(
        &mut pool,
        &block_table,
        block_tokens,
        n_head_kv,
        head_dim,
        seed,
        seq_tokens,
    );

    let query = [0.8f32, -0.4f32];

    // Compute expected output independently in f64
    let scale = 1.0f64 / (head_dim as f64).sqrt();
    let mut scores = Vec::with_capacity(seq_tokens);
    let mut values = Vec::with_capacity(seq_tokens);

    for t in 0..seq_tokens {
        let block_idx = t / block_tokens;
        let slot = t % block_tokens;
        let phys = block_table[block_idx] as usize;
        let base = phys * floats_per_block + slot * floats_per_token;

        let key = &pool[base..base + head_dim];
        let val = &pool[base + row_len..base + row_len + head_dim];

        let dot = (query[0] as f64) * (key[0] as f64) + (query[1] as f64) * (key[1] as f64);
        let score = dot * scale;
        scores.push(score);
        values.push([val[0] as f64, val[1] as f64]);
    }

    // Softmax: max-subtract, exp, normalize
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp: f64 = exps.iter().sum();
    let weights: Vec<f64> = exps.iter().map(|&e| e / sum_exp).collect();

    // Output: out[j] = sum_t w(t) * value(t)[j]
    let mut expected = vec![0.0f32; head_dim];
    for d in 0..head_dim {
        let mut sum = 0.0f64;
        for t in 0..seq_tokens {
            sum += weights[t] * values[t][d];
        }
        expected[d] = sum as f32;
    }

    let actual = sdpa_decode(
        &pool,
        &block_table,
        block_tokens,
        seq_tokens,
        &query,
        n_head,
        n_head_kv,
        head_dim,
        false,
        seq_tokens,
    );

    assert_eq!(actual.len(), expected.len());
    let cos_sim = cosine(&actual, &expected);
    assert!(cos_sim >= 0.9999, "cosine similarity {cos_sim} < 0.9999");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < 1e-5,
            "element {i} diff {diff} exceeds 1e-5 (actual={a}, expected={e})"
        );
    }
}

// ---------------------------------------------------------------------------
// TEST 2 (1.2): sdpa_scattered_blocks_match_contiguous
// ---------------------------------------------------------------------------
#[test]
fn sdpa_scattered_blocks_match_contiguous() {
    let block_tokens = 2;
    let n_head = 2;
    let n_head_kv = 1;
    let head_dim = 3;
    let seq_tokens = 7;
    let num_blocks = 4;
    let row_len = n_head_kv * head_dim;
    let floats_per_token = 2 * row_len;
    let floats_per_block = block_tokens * floats_per_token;
    let pool_len = num_blocks * floats_per_block;

    let contig_table = [0u32, 1u32, 2u32, 3u32];
    let scat_table = [3u32, 0u32, 2u32, 1u32];

    let mut contig_pool = vec![0.0f32; pool_len];
    let mut scat_pool = vec![0.0f32; pool_len];

    let seed = 0x1234_5678;
    fill_pool(
        &mut contig_pool,
        &contig_table,
        block_tokens,
        n_head_kv,
        head_dim,
        seed,
        seq_tokens,
    );
    fill_pool(
        &mut scat_pool,
        &scat_table,
        block_tokens,
        n_head_kv,
        head_dim,
        seed,
        seq_tokens,
    );

    let mut query_rng = Xorshift::new(0xABCD);
    let query: Vec<f32> = (0..n_head * head_dim)
        .map(|_| query_rng.next_f32())
        .collect();

    let r_contig = sdpa_decode(
        &contig_pool,
        &contig_table,
        block_tokens,
        seq_tokens,
        &query,
        n_head,
        n_head_kv,
        head_dim,
        false,
        seq_tokens,
    );

    let r_scat = sdpa_decode(
        &scat_pool,
        &scat_table,
        block_tokens,
        seq_tokens,
        &query,
        n_head,
        n_head_kv,
        head_dim,
        false,
        seq_tokens,
    );

    assert_eq!(r_contig.len(), n_head * head_dim);
    assert_eq!(r_scat.len(), n_head * head_dim);

    let cos_sim = cosine(&r_contig, &r_scat);
    assert!(
        cos_sim >= 0.9999,
        "cosine similarity {cos_sim} < 0.9999 between contiguous and scattered"
    );

    let norm_contig: f64 = r_contig
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    let norm_scat: f64 = r_scat
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    assert!(
        r_contig.iter().all(|x| x.is_finite()) && norm_contig > 0.0,
        "r_contig has non-finite values or zero norm"
    );
    assert!(
        r_scat.iter().all(|x| x.is_finite()) && norm_scat > 0.0,
        "r_scat has non-finite values or zero norm"
    );
}

// ---------------------------------------------------------------------------
// TEST 3 (1.3): sdpa_causal_masks_upper_triangle
// ---------------------------------------------------------------------------
#[test]
fn sdpa_causal_masks_upper_triangle() {
    let block_tokens = 2;
    let n_head = 1;
    let n_head_kv = 1;
    let head_dim = 2;
    let block_table = [0u32, 1u32, 2u32, 3u32];
    let num_blocks = 4;
    let row_len = n_head_kv * head_dim;
    let floats_per_token = 2 * row_len;
    let floats_per_block = block_tokens * floats_per_token;
    let mut pool = vec![0.0f32; num_blocks * floats_per_block];

    let seed = 0xCAFE;
    fill_pool(
        &mut pool,
        &block_table,
        block_tokens,
        n_head_kv,
        head_dim,
        seed,
        8,
    );

    let query = [0.6f32, 0.9f32];

    // Causal decode attending up to query_pos = 3 (tokens 0..=3)
    let r_c = sdpa_decode(
        &pool,
        &block_table,
        block_tokens,
        8,
        &query,
        n_head,
        n_head_kv,
        head_dim,
        true,
        3,
    );

    // Non-causal decode restricted to seq_tokens = 4 (tokens 0..3)
    let r_r = sdpa_decode(
        &pool,
        &block_table,
        block_tokens,
        4,
        &query,
        n_head,
        n_head_kv,
        head_dim,
        false,
        4,
    );

    assert_eq!(r_c.len(), head_dim);
    assert_eq!(r_r.len(), head_dim);

    let cos_sim = cosine(&r_c, &r_r);
    assert!(
        cos_sim >= 0.9999,
        "causal vs restricted cosine similarity {cos_sim} < 0.9999"
    );

    for (i, (&c, &r)) in r_c.iter().zip(r_r.iter()).enumerate() {
        let diff = (c - r).abs();
        assert!(
            diff < 1e-5,
            "element {i} diff {diff} exceeds 1e-5 (causal={c}, restricted={r})"
        );
    }
}
