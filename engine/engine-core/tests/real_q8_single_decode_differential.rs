//! Diagnostic-only one-step Q8-vs-F32 ForwardDriver differential.

use engine_core::forward_driver::ForwardDriver;
use engine_core::tokenizer::BpeTokenizer;
use engine_io::{GgufReader, ModelConfig, load_to_pinned};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

const DECODE_PATH: &str = "TITAN_FORWARD_DECODE_PATH";
const PROMPT: &str = "Explain gravity in one short sentence.";

fn fixture() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        std::env::var_os("ENGINE_TESTDATA").map(PathBuf::from),
        Some(manifest.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf")),
        Some(manifest.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf")),
        Some(PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf")),
    ]
    .into_iter()
    .flatten()
    .find(|path| path.exists())
}

fn metrics(a: &[f32], b: &[f32]) -> (f64, f64, bool, f64) {
    let finite = a.len() == b.len() && a.iter().chain(b).all(|value| value.is_finite());
    if !finite {
        return (f64::NAN, f64::NAN, false, f64::NAN);
    }
    let (mut squared_delta, mut squared_b, mut dot, mut squared_a, mut max_abs) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        squared_delta += (x - y).powi(2);
        squared_b += y.powi(2);
        dot += x * y;
        squared_a += x.powi(2);
        max_abs = max_abs.max((x - y).abs());
    }
    let relative_l2 = if squared_b > 0.0 {
        (squared_delta / squared_b).sqrt()
    } else {
        0.0
    };
    let cosine = if squared_a > 0.0 && squared_b > 0.0 {
        dot / (squared_a * squared_b).sqrt()
    } else {
        1.0
    };
    (relative_l2, cosine, true, max_abs)
}

fn run_with_path(
    path_name: &str,
    reader: &GgufReader,
    pinned: &engine_io::LoadedPinned,
    config: &ModelConfig,
    tokens: &[u32],
    next_token: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    // The capacity includes the prefetched prompt and the one additional decode token.
    // The lock in the test protects this process-local diagnostic mutation.
    unsafe { std::env::set_var(DECODE_PATH, path_name) };
    let mut driver = ForwardDriver::new(reader, pinned, config, tokens.len() + 1)?;
    driver.prefill(tokens)?;
    Ok(driver.decode(next_token)?)
}

#[test]
#[ignore]
fn real_q8_single_decode_differential() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    engine_cuda::ensure_cuda_dll_paths();
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _lock = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
    let previous = std::env::var_os(DECODE_PATH);
    let restore = || match &previous {
        Some(value) => unsafe { std::env::set_var(DECODE_PATH, value) },
        None => unsafe { std::env::remove_var(DECODE_PATH) },
    };

    let result = (|| {
        let fixture = fixture().ok_or("Qwen3 fixture not present")?;
        let reader = GgufReader::open(&fixture)?;
        let pinned = load_to_pinned(&reader, &fixture)?;
        let config = ModelConfig::from_reader(&reader)?;
        let tokenizer = BpeTokenizer::from_reader(&reader)?;
        let tokens = tokenizer.encode(PROMPT)?;
        if tokens.is_empty() {
            return Err("prompt encoded to zero tokens".into());
        }

        let f32_prefill = run_with_path("f32", &reader, &pinned, &config, &tokens, tokens[0])?;
        let next_token = f32_prefill
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index as u32)
            .ok_or("F32 prefill returned empty logits")?;
        let f32_logits = run_with_path("f32", &reader, &pinned, &config, &tokens, next_token)?;
        let q8_logits = run_with_path("q8", &reader, &pinned, &config, &tokens, next_token)?;
        let (relative_l2, cosine, finite, max_abs_error) = metrics(&q8_logits, &f32_logits);
        let artifact = json!({
            "schema_version": 1,
            "model_path": fixture,
            "prompt": PROMPT,
            "prompt_token_length": tokens.len(),
            "decode_token": next_token,
            "logit_length": f32_logits.len(),
            "paths": {"f32": "f32", "q8": "q8"},
            "metrics": {
                "relative_l2": relative_l2,
                "cosine": cosine,
                "finite": finite,
                "max_absolute_error": max_abs_error
            },
            "status": "diagnostic_only",
            "conclusion": "Measurement only; no pass threshold is asserted."
        });
        let output = std::env::var_os("TITAN_Q8_SINGLE_JSON")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../local-artifacts/reviews/real-q8-single-decode-differential.json")
            });
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&artifact)?)?;
        println!("Q8-vs-F32 single decode: {artifact}");
        println!("diagnostic artifact: {}", output.display());
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })();
    restore();
    result
}
