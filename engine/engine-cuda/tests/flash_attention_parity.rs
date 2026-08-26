//! FlashAttention-2 GPU vs CPU Parity Test (Phase 11, Sub-change 11.2).
//!
//! Asserts that:
//! 1. `FlashAttention2::launch` computes causal multi-head attention directly over resident paged KV blocks.
//! 2. Numerics match exact CPU reference causal attention with cosine similarity >= 0.9999 across sequence lengths S in {1, 4, 16, 64, 128}.

use cudarc::driver::CudaDevice;
use engine_cuda::{CudaStream, DeviceBuffer, FlashAttention2, PagedKvGpu, PagedKvLayout};

type DynError = Box<dyn std::error::Error + Send + Sync>;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

#[test]
#[ignore]
fn test_flash_attention_2_parity() -> Result<(), DynError> {
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(device.clone())?;
    let flash_attn = FlashAttention2::new(device.clone())?;
    let pkv = PagedKvGpu::new(device.clone())?;

    let nh = 16;
    let nkv = 2;
    let hd = 128;
    let block_tokens = 64;
    let gqa_group = nh / nkv;

    let layout = PagedKvLayout {
        n_blocks: 4,
        block_tokens,
        row_len: nkv * hd,
    };

    let test_seq_lens = [1, 4, 16, 64, 128];

    for &seq_tokens in &test_seq_lens {
        let pool_floats = layout.floats_total();
        let pool_dev = DeviceBuffer::alloc(device.clone(), pool_floats * 4)?;
        let block_table_bytes = vec![0u8; 4 * 4]; // block 0..3 maps to phys 0..3
        let mut bt_u32 = Vec::new();
        for i in 0..4u32 {
            bt_u32.extend_from_slice(&i.to_le_bytes());
        }
        let bt_dev = DeviceBuffer::alloc(device.clone(), block_table_bytes.len())?;
        bt_dev.copy_from_host(&stream, &bt_u32)?;

        // Generate synthetic keys and values for all positions 0..seq_tokens
        let mut keys_host = Vec::with_capacity(seq_tokens * nkv * hd);
        let mut vals_host = Vec::with_capacity(seq_tokens * nkv * hd);

        for t in 0..seq_tokens {
            for kvh in 0..nkv {
                for d in 0..hd {
                    let k_val = (((t * 100 + kvh * 10 + d) as f32 * 0.013).sin()) * 0.5;
                    let v_val = (((t * 100 + kvh * 10 + d) as f32 * 0.017).cos()) * 0.5;
                    keys_host.push(k_val);
                    vals_host.push(v_val);
                }
            }
        }

        let k_dev = DeviceBuffer::alloc(device.clone(), keys_host.len() * 4)?;
        let v_dev = DeviceBuffer::alloc(device.clone(), vals_host.len() * 4)?;
        k_dev.copy_from_host(&stream, &f32_bytes(&keys_host))?;
        v_dev.copy_from_host(&stream, &f32_bytes(&vals_host))?;

        // Append to resident paged KV pool
        pkv.append_kv(
            &stream,
            &layout,
            &pool_dev,
            &k_dev,
            &v_dev,
            &bt_dev,
            0,
            seq_tokens,
        )?;

        // Generate synthetic Q vectors for all positions 0..seq_tokens
        let mut q_host = Vec::with_capacity(seq_tokens * nh * hd);
        for t in 0..seq_tokens {
            for h in 0..nh {
                for d in 0..hd {
                    let q_val = (((t * 50 + h * 5 + d) as f32 * 0.019).sin()) * 0.5;
                    q_host.push(q_val);
                }
            }
        }

        let q_dev = DeviceBuffer::alloc(device.clone(), q_host.len() * 4)?;
        let out_dev = DeviceBuffer::alloc(device.clone(), seq_tokens * nh * hd * 4)?;
        q_dev.copy_from_host(&stream, &f32_bytes(&q_host))?;

        // Launch FlashAttention-2
        flash_attn.launch(
            &stream,
            &q_dev,
            &pool_dev,
            &bt_dev,
            &out_dev,
            nh,
            nkv,
            hd,
            block_tokens,
            seq_tokens,
        )?;

        let mut out_bytes = vec![0u8; seq_tokens * nh * hd * 4];
        out_dev.copy_to_host(&stream, &mut out_bytes)?;

        let mut out_gpu = vec![0.0f32; seq_tokens * nh * hd];
        for i in 0..out_gpu.len() {
            out_gpu[i] = f32::from_le_bytes(out_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        }

        // CPU reference causal multi-head attention:
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out_cpu = vec![0.0f32; seq_tokens * nh * hd];

        for q_pos in 0..seq_tokens {
            for qh in 0..nh {
                let kh = qh / gqa_group;
                let q_vec = &q_host[(q_pos * nh * hd + qh * hd)..(q_pos * nh * hd + (qh + 1) * hd)];

                // Compute scores against all k_pos in [0, q_pos]
                let mut scores = Vec::with_capacity(q_pos + 1);
                for k_pos in 0..=q_pos {
                    let k_vec = &keys_host[(k_pos * nkv * hd + kh * hd)..(k_pos * nkv * hd + (kh + 1) * hd)];
                    let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                    scores.push(dot * scale);
                }

                // Softmax
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut weights = Vec::with_capacity(scores.len());
                for &s in &scores {
                    let e = (s - max_s).exp();
                    weights.push(e);
                    sum_exp += e;
                }
                for w in &mut weights {
                    *w /= sum_exp;
                }

                // Weighted sum of V
                let mut out_head = vec![0.0f32; hd];
                for (k_pos, &w) in weights.iter().enumerate() {
                    let v_vec = &vals_host[(k_pos * nkv * hd + kh * hd)..(k_pos * nkv * hd + (kh + 1) * hd)];
                    for d in 0..hd {
                        out_head[d] += w * v_vec[d];
                    }
                }

                let cpu_slice_start = q_pos * nh * hd + qh * hd;
                out_cpu[cpu_slice_start..cpu_slice_start + hd].copy_from_slice(&out_head);
            }
        }

        // Compare GPU vs CPU for all (seq_tokens, nh) vectors
        for q_pos in 0..seq_tokens {
            for qh in 0..nh {
                let offset = q_pos * nh * hd + qh * hd;
                let cpu_head = &out_cpu[offset..offset + hd];
                let gpu_head = &out_gpu[offset..offset + hd];
                let cs = cosine_similarity(cpu_head, gpu_head);
                assert!(
                    cs >= 0.9999,
                    "Cosine similarity {cs} below threshold at seq_tokens = {seq_tokens}, q_pos = {q_pos}, head = {qh}"
                );
            }
        }

        println!("FlashAttention-2 PASS for seq_tokens = {seq_tokens}");
    }

    Ok(())
}
