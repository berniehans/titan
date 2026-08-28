//! Golden gate integration test for single-token decode and teacher-forced drift curve (Phase 6.7, Group 2).
//!
//! Validates:
//! 1. Single-token decode reusing resident KV pool emits logits matching full prefill and CPU reference.
//! 2. Teacher-forced decode across >=10 checkpoints maintains cumulative drift bounds vs independent CPU fp32 reference.

use engine_core::forward_cpu::{
    Tensor, TensorType, embed_lookup, logits_from_hidden, matmul, rms_norm, rope_neox_partial,
    sdpa_decode, silu,
};
use engine_core::forward_driver::{ForwardDriver, run_prefill};
use engine_core::tokenizer::BpeTokenizer;
use engine_io::{GgmlType, GgufReader, LoadedPinned, ModelConfig, load_to_pinned};
use std::io::Read;
use std::path::PathBuf;

fn get_fixture_path() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    let md = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for c in [
        md.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        md.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
    ] {
        if c.exists() {
            return Some(c.canonicalize().unwrap_or(c));
        }
    }
    None
}

fn get_prompts_path() -> Option<PathBuf> {
    let md = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for c in [
        md.join("../../tests/fixtures/prompts.txt"),
        md.join("../tests/fixtures/prompts.txt"),
        PathBuf::from("tests/fixtures/prompts.txt"),
        PathBuf::from("../tests/fixtures/prompts.txt"),
    ] {
        if c.exists() {
            return Some(c.canonicalize().unwrap_or(c));
        }
    }
    None
}

fn get_golden_logits_path(idx: usize) -> Option<PathBuf> {
    let md = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for c in [
        md.join(format!(
            "../../tests/fixtures/golden/logits/logits_{idx:02}.bin"
        )),
        md.join(format!(
            "../tests/fixtures/golden/logits/logits_{idx:02}.bin"
        )),
        PathBuf::from(format!("tests/fixtures/golden/logits/logits_{idx:02}.bin")),
        PathBuf::from(format!(
            "../tests/fixtures/golden/logits/logits_{idx:02}.bin"
        )),
    ] {
        if c.exists() {
            return Some(c.canonicalize().unwrap_or(c));
        }
    }
    None
}

fn ggml_to_bank(t: GgmlType) -> Option<TensorType> {
    match t {
        GgmlType::F32 => Some(TensorType::F32),
        GgmlType::Q4_K => Some(TensorType::Q4K),
        GgmlType::Q6_K => Some(TensorType::Q6K),
        _ => None,
    }
}

fn bank_tensor<'a>(read: &GgufReader, pinned: &'a LoadedPinned, name: &str) -> Tensor<'a> {
    let info = read
        .get_tensor(name)
        .unwrap_or_else(|| panic!("missing {name}"));
    assert_eq!(info.dims.len(), 2, "{name} not 2-D");
    let ty = ggml_to_bank(info.ggml_type).unwrap_or_else(|| panic!("unsupported quant {name}"));
    Tensor {
        ty,
        data: pinned
            .tensor(name)
            .unwrap_or_else(|| panic!("{name} bytes")),
        ne0: info.dims[0] as usize,
        ne1: info.dims[1] as usize,
        n_rot: 0,
    }
}

fn f32_norm(pinned: &LoadedPinned, name: &str) -> Vec<f32> {
    let b = pinned
        .tensor(name)
        .unwrap_or_else(|| panic!("{name} bytes"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_f32(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_golden_logits(idx: usize) -> Option<Vec<f32>> {
    let path = get_golden_logits_path(idx)?;
    let raw = std::fs::read(&path).ok()?;
    let mut decompressed = Vec::new();
    let mut d_zlib = flate2::read::ZlibDecoder::new(&raw[..]);
    if d_zlib.read_to_end(&mut decompressed).is_err() || decompressed.is_empty() {
        decompressed.clear();
        let mut d_gz = flate2::read::GzDecoder::new(&raw[..]);
        let _ = d_gz.read_to_end(&mut decompressed);
    }
    if decompressed.is_empty() || decompressed.len() % 4 != 0 {
        return None;
    }
    Some(bytes_f32(&decompressed))
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        num += d * d;
        den += *y as f64 * *y as f64;
    }
    (num / den).sqrt()
}

fn cosim(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    dot / (na * nb).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn run_cpu_reference(
    reader: &GgufReader,
    pinned: &LoadedPinned,
    cfg: &ModelConfig,
    tokens: &[u32],
) -> Vec<f32> {
    let h = cfg.hidden_size as usize;
    let hd = cfg.head_dim as usize;
    let nh = cfg.n_head as usize;
    let nkv = cfg.n_head_kv as usize;
    let ff = cfg.intermediate_size as usize;
    let eps = cfg.rms_norm_eps;
    let base = cfg.rope_freq_base;
    let n_rot = hd; // full head-dim rotation (Qwen3; head_dim/2 falsified by golden)
    let qdim = nh * hd;
    let kvd = nkv * hd;
    let seq_len = tokens.len();
    let n_layer = cfg.n_layer as usize;

    let emb = bank_tensor(reader, pinned, "token_embd.weight");
    let head_norm = f32_norm(pinned, "output_norm.weight");

    struct CpuLayer<'a> {
        wq: Tensor<'a>,
        wk: Tensor<'a>,
        wv: Tensor<'a>,
        wo: Tensor<'a>,
        wgate: Tensor<'a>,
        wup: Tensor<'a>,
        wdown: Tensor<'a>,
        an: Vec<f32>,
        qn: Vec<f32>,
        kn: Vec<f32>,
        fnw: Vec<f32>,
    }

    let mut layers = Vec::with_capacity(n_layer);
    for l in 0..n_layer {
        layers.push(CpuLayer {
            wq: bank_tensor(reader, pinned, &format!("blk.{l}.attn_q.weight")),
            wk: bank_tensor(reader, pinned, &format!("blk.{l}.attn_k.weight")),
            wv: bank_tensor(reader, pinned, &format!("blk.{l}.attn_v.weight")),
            wo: bank_tensor(reader, pinned, &format!("blk.{l}.attn_output.weight")),
            wgate: bank_tensor(reader, pinned, &format!("blk.{l}.ffn_gate.weight")),
            wup: bank_tensor(reader, pinned, &format!("blk.{l}.ffn_up.weight")),
            wdown: bank_tensor(reader, pinned, &format!("blk.{l}.ffn_down.weight")),
            an: f32_norm(pinned, &format!("blk.{l}.attn_norm.weight")),
            qn: f32_norm(pinned, &format!("blk.{l}.attn_q_norm.weight")),
            kn: f32_norm(pinned, &format!("blk.{l}.attn_k_norm.weight")),
            fnw: f32_norm(pinned, &format!("blk.{l}.ffn_norm.weight")),
        });
    }

    let mut cpu_pools = vec![vec![0.0f32; seq_len * 2 * kvd]; n_layer];
    let mut x = vec![0.0f32; h];

    for (p, &token_id) in tokens.iter().enumerate() {
        x = embed_lookup(&emb, token_id as usize);

        for (l, layer) in layers.iter().enumerate() {
            let normed = rms_norm(&x, &layer.an, eps);
            let mut q = vec![0.0f32; qdim];
            let mut k = vec![0.0f32; kvd];
            let mut v = vec![0.0f32; kvd];
            matmul(&mut q, &layer.wq, &normed);
            matmul(&mut k, &layer.wk, &normed);
            matmul(&mut v, &layer.wv, &normed);

            for hh in 0..nh {
                let s = hh * hd;
                let row = q[s..s + hd].to_vec();
                let q_normed = rms_norm(&row, &layer.qn, eps);
                let q_rot = rope_neox_partial(&q_normed, p as u32, n_rot, base);
                q[s..s + hd].copy_from_slice(&q_rot);
            }

            for hh in 0..nkv {
                let s = hh * hd;
                let row = k[s..s + hd].to_vec();
                let k_normed = rms_norm(&row, &layer.kn, eps);
                let k_rot = rope_neox_partial(&k_normed, p as u32, n_rot, base);
                k[s..s + hd].copy_from_slice(&k_rot);
            }

            let slot_base = p * (2 * kvd);
            cpu_pools[l][slot_base..slot_base + kvd].copy_from_slice(&k);
            cpu_pools[l][slot_base + kvd..slot_base + 2 * kvd].copy_from_slice(&v);

            let attn = sdpa_decode(
                &cpu_pools[l],
                &[0u32],
                seq_len,
                p + 1,
                &q,
                nh,
                nkv,
                hd,
                true,
                p,
            );

            let mut op = vec![0.0f32; h];
            matmul(&mut op, &layer.wo, &attn);
            let mut h1 = vec![0.0f32; h];
            for i in 0..h {
                h1[i] = x[i] + op[i];
            }

            let ffin = rms_norm(&h1, &layer.fnw, eps);
            let mut gate = vec![0.0f32; ff];
            let mut up = vec![0.0f32; ff];
            matmul(&mut gate, &layer.wgate, &ffin);
            matmul(&mut up, &layer.wup, &ffin);
            let g = silu(&gate);
            let mut proj = vec![0.0f32; ff];
            for i in 0..ff {
                proj[i] = g[i] * up[i];
            }

            let mut down = vec![0.0f32; h];
            matmul(&mut down, &layer.wdown, &proj);
            let mut h2 = vec![0.0f32; h];
            for i in 0..h {
                h2[i] = h1[i] + down[i];
            }
            x = h2;
        }
    }

    logits_from_hidden(&emb, &head_norm, &x, eps)
}

#[test]
#[ignore] // GPU test
fn decode_reuses_resident_kv_and_emits_logits() {
    engine_cuda::ensure_cuda_dll_paths();
    let fixture_path = match get_fixture_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: fixture not present");
            return;
        }
    };
    let prompts_path = match get_prompts_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: prompts fixture not present");
            return;
        }
    };

    let reader = GgufReader::open(&fixture_path).expect("open gguf");
    let cfg = ModelConfig::from_reader(&reader).expect("read config");
    let pinned = load_to_pinned(&reader, &fixture_path).expect("load pinned");
    let tokenizer = BpeTokenizer::from_reader(&reader).expect("tokenizer");

    let prompts: Vec<String> = std::fs::read_to_string(&prompts_path)
        .expect("read prompts")
        .lines()
        .map(|l| l.trim_end_matches('\r').to_string())
        .filter(|l| !l.trim().is_empty())
        .take(12)
        .collect();

    assert_eq!(
        prompts.len(),
        12,
        "expected 12 prompts in prompts.txt, found {}",
        prompts.len()
    );

    let mut min_cosim_cpu = 1.0f64;
    let mut max_rel_l2_cpu = 0.0f64;
    let mut min_cosim_prefill = 1.0f64;
    let mut max_rel_l2_prefill = 0.0f64;

    for (i, prompt) in prompts.iter().enumerate() {
        let token_ids = tokenizer.encode(prompt).expect("encode prompt");
        assert!(!token_ids.is_empty(), "prompt {i:02} produced 0 tokens");

        let len = token_ids.len();
        let mut drv =
            ForwardDriver::new(&reader, &pinned, &cfg, len + 1).expect("ForwardDriver::new failed");

        let mut dec = Vec::new();
        for &tok in &token_ids {
            dec = drv.decode(tok).expect("decode failed");
        }

        let cpu_ref = run_cpu_reference(&reader, &pinned, &cfg, &token_ids);
        let full_pre = run_prefill(&reader, &pinned, &cfg, &token_ids)
            .expect("run_prefill full prompt failed");

        println!("    [T_DBG] dec: {:?}\n    [T_DBG] cpu: {:?}\n    [T_DBG] pre: {:?}", &dec[..5], &cpu_ref[..5], &full_pre.logits[..5]);
        let cs_cpu = cosim(&dec, &cpu_ref);
        let rl_cpu = rel_l2(&dec, &cpu_ref);
        let cs_pre = cosim(&dec, &full_pre.logits);
        let rl_pre = rel_l2(&dec, &full_pre.logits);

        println!(
            "Prompt {i:02} (len {len}): dec-vs-cpu cos={cs_cpu:.6} rl={rl_cpu:.3e} (gate>0.99, <0.15); dec-vs-prefill cos={cs_pre:.6} rl={rl_pre:.3e} (gate>0.99)"
        );

        assert!(
            cs_cpu > 0.99,
            "Prompt {i:02} dec-vs-cpu cos_sim {cs_cpu:.6} <= 0.99"
        );
        assert!(
            rl_cpu < 0.15,
            "Prompt {i:02} dec-vs-cpu rel_l2 {rl_cpu:.3e} >= 0.15"
        );
        assert!(
            cs_pre > 0.99,
            "Prompt {i:02} dec-vs-prefill cos_sim {cs_pre:.6} <= 0.99"
        );

        min_cosim_cpu = min_cosim_cpu.min(cs_cpu);
        max_rel_l2_cpu = max_rel_l2_cpu.max(rl_cpu);
        min_cosim_prefill = min_cosim_prefill.min(cs_pre);
        max_rel_l2_prefill = max_rel_l2_prefill.max(rl_pre);
    }

    println!(
        "\n=== Phase 6.7 Group 2 Single-Token Decode Summary Across 12 Prompts ===\n\
         Min cos_sim vs CPU ref   : {min_cosim_cpu:.6} (gate > 0.99)\n\
         Max rel_l2 vs CPU ref   : {max_rel_l2_cpu:.3e} (gate < 0.15)\n\
         Min cos_sim vs Prefill  : {min_cosim_prefill:.6} (gate > 0.99)\n\
         Max rel_l2 vs Prefill   : {max_rel_l2_prefill:.3e}\n"
    );

    assert!(
        min_cosim_cpu > 0.99,
        "Overall min cos_sim vs CPU ref {min_cosim_cpu:.6} <= 0.99"
    );
    assert!(
        max_rel_l2_cpu < 0.15,
        "Overall max rel_l2 vs CPU ref {max_rel_l2_cpu:.3e} >= 0.15"
    );
    assert!(
        min_cosim_prefill > 0.99,
        "Overall min cos_sim vs full prefill {min_cosim_prefill:.6} <= 0.99"
    );
}

/// Integration test for teacher-forced decode drift curve (Phase 6.7, Group 2).
///
/// Pinned llama.cpp goldens cover only the final position of each prompt;
/// intermediate per-step drift is validated against the model's own independent CPU fp32 reference.
#[test]
#[ignore] // GPU test
fn teacher_forced_drift_curve_10_checkpoints() {
    let fixture_path = match get_fixture_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: fixture not present");
            return;
        }
    };
    let prompts_path = match get_prompts_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: prompts fixture not present");
            return;
        }
    };

    let reader = GgufReader::open(&fixture_path).expect("open gguf");
    let cfg = ModelConfig::from_reader(&reader).expect("read config");
    let pinned = load_to_pinned(&reader, &fixture_path).expect("load pinned");
    let tokenizer = BpeTokenizer::from_reader(&reader).expect("tokenizer");

    let prompts: Vec<String> = std::fs::read_to_string(&prompts_path)
        .expect("read prompts")
        .lines()
        .map(|l| l.trim_end_matches('\r').to_string())
        .filter(|l| !l.trim().is_empty())
        .take(12)
        .collect();

    assert_eq!(
        prompts.len(),
        12,
        "expected 12 prompts in prompts.txt, found {}",
        prompts.len()
    );

    let mut total_checkpoints = 0;
    let mut min_cosim = 1.0f64;
    let mut max_rel_l2 = 0.0f64;

    for (i, prompt) in prompts.iter().enumerate() {
        let token_ids = tokenizer.encode(prompt).expect("encode prompt");
        if token_ids.len() < 2 {
            continue;
        }

        let mut drv = ForwardDriver::new(&reader, &pinned, &cfg, token_ids.len())
            .expect("ForwardDriver::new failed");

        let _pre = drv
            .prefill(&token_ids[..1])
            .expect("prefill token 0 failed");

        let mut last_logits = Vec::new();
        for p in 1..token_ids.len() {
            let logits = drv.decode(token_ids[p]).expect("decode failed");
            let cpu = run_cpu_reference(&reader, &pinned, &cfg, &token_ids[..=p]);
            let cos = cosim(&logits, &cpu);
            let rel = rel_l2(&logits, &cpu);

            println!(
                "Prompt {i:02} step p={p:02} (token {}): cos={cos:.6} rel={rel:.3e} (gate cos>0.99, rel<0.15)",
                token_ids[p]
            );

            assert!(
                cos > 0.99,
                "Prompt {i:02} step p={p} cos_sim {cos:.6} <= 0.99"
            );
            assert!(
                rel < 0.15,
                "Prompt {i:02} step p={p} rel_l2 {rel:.3e} >= 0.15"
            );

            min_cosim = min_cosim.min(cos);
            max_rel_l2 = max_rel_l2.max(rel);
            total_checkpoints += 1;
            last_logits = logits;
        }

        // Bonus: check final position against llama.cpp golden logits if available
        if let Some(golden_logits) = load_golden_logits(i) {
            let cs_golden = cosim(&last_logits, &golden_logits);
            println!("Prompt {i:02} final decode vs llama.cpp golden: cos={cs_golden:.6}");
            assert!(
                cs_golden > 0.99,
                "Prompt {i:02} final decode vs golden cos_sim {cs_golden:.6} <= 0.99"
            );
        }
    }

    println!(
        "\n=== Phase 6.7 Group 2 Teacher-Forced Drift Curve Summary ===\n\
         Total checkpoints evaluated : {total_checkpoints} (gate >= 10)\n\
         Min cos_sim vs CPU ref     : {min_cosim:.6} (gate > 0.99)\n\
         Max rel_l2 vs CPU ref     : {max_rel_l2:.3e} (gate < 0.15)\n"
    );

    assert!(
        total_checkpoints >= 10,
        "Total checkpoints {total_checkpoints} < 10 required"
    );
    assert!(
        min_cosim > 0.99,
        "Overall min cos_sim {min_cosim:.6} <= 0.99"
    );
    assert!(
        max_rel_l2 < 0.15,
        "Overall max rel_l2 {max_rel_l2:.3e} >= 0.15"
    );
}
