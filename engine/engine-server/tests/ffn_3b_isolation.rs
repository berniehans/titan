//! Focused Titan-only Llama 3.2 3B FFN isolation benchmark.
//!
//! This is an ignored, local GPU benchmark. It deliberately measures only Titan's
//! real ForwardDriver path; it does not invoke or compare another runtime.

use engine_core::{BpeTokenizer, ForwardDriver, Sampler, SamplerParams};
use engine_io::{GgufReader, ModelConfig, load_to_pinned};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const MODEL_RELATIVE_PATH: &str = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf";
const PROMPT: &str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a concise assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nExplain why measuring FFN stages matters in a transformer decode benchmark.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
const REPETITIONS: usize = 5;
const GENERATED_TOKENS: usize = 41;

#[derive(Debug, Serialize)]
struct IsolationReport {
    schema_version: u32,
    model_path: String,
    repetitions: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    warmup: Warmup,
    runs: Vec<MeasuredRun>,
    command: String,
    configuration: Configuration,
}

#[derive(Debug, Serialize)]
struct Warmup {
    generated_tokens: usize,
    wall_clock_decode_ms: f64,
}

#[derive(Debug, Serialize)]
struct MeasuredRun {
    repetition: usize,
    generated_tokens: usize,
    wall_clock_decode_ms: f64,
    decode_telemetry: Option<engine_core::DecodeTelemetry>,
}

#[derive(Debug, Serialize)]
struct Configuration {
    prompt: &'static str,
    native_nvrtc_preflight: bool,
    dispatch_telemetry: bool,
    max_sequence: usize,
    sampler: &'static str,
    comparison_runtime: Option<&'static str>,
}

fn model_path() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        manifest_dir.join("../../").join(MODEL_RELATIVE_PATH),
        manifest_dir.join("../").join(MODEL_RELATIVE_PATH),
        manifest_dir.join(MODEL_RELATIVE_PATH),
        PathBuf::from(MODEL_RELATIVE_PATH),
    ]
    .into_iter()
    .find(|path| path.exists())
    .map(|path| path.canonicalize().unwrap_or(path))
}

fn artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../local-artifacts/benchmarks/ffn-3b-isolation.json")
}

fn decode_once(
    driver: &mut ForwardDriver<'_>,
    tokenizer: &BpeTokenizer,
    sampler: &mut Sampler,
    prompt_tokens: &[u32],
) -> Result<(usize, f64, Option<engine_core::DecodeTelemetry>), Box<dyn std::error::Error>> {
    driver.reset_pos();
    let logits = driver.prefill(prompt_tokens)?;
    let first = sampler.sample(
        &logits,
        prompt_tokens,
        &SamplerParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 1.0,
            seed: None,
        },
    );
    driver.decode(first)?;
    let start = Instant::now();
    let generated = driver.generate_autonomous_gpu(first, GENERATED_TOKENS)?;
    let wall_clock_decode_ms = start.elapsed().as_secs_f64() * 1000.0;
    let telemetry = driver.take_decode_telemetry();
    let _ = tokenizer.decode(&generated);
    Ok((generated.len(), wall_clock_decode_ms, telemetry))
}

#[test]
#[ignore]
fn llama_3b_ffn_isolation_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    engine_cuda::ensure_cuda_dll_paths();
    // This ignored test is single-threaded and owns the process-wide benchmark flag.
    unsafe { std::env::set_var("TITAN_BENCHMARK_DISPATCH_TELEMETRY", "1") };

    let model = model_path().ok_or_else(|| format!("model not found: {MODEL_RELATIVE_PATH}"))?;
    let reader = GgufReader::open(&model)?;
    let config = ModelConfig::from_reader(&reader)?;
    let pinned = load_to_pinned(&reader, &model)?;
    let tokenizer = BpeTokenizer::from_reader(&reader)?;
    let prompt_tokens = tokenizer.encode(PROMPT)?;
    let mut driver = ForwardDriver::new(&reader, &pinned, &config, 512)?;
    driver.enable_dispatch_telemetry();
    let _ = driver.capture_autonomous_decode_graph();
    let mut sampler = Sampler::new(0);

    let (warmup_tokens, warmup_ms, _) =
        decode_once(&mut driver, &tokenizer, &mut sampler, &prompt_tokens)?;
    let mut runs = Vec::with_capacity(REPETITIONS);
    for repetition in 1..=REPETITIONS {
        let (generated_tokens, wall_clock_decode_ms, decode_telemetry) =
            decode_once(&mut driver, &tokenizer, &mut sampler, &prompt_tokens)?;
        let stages = decode_telemetry
            .as_ref()
            .map(|telemetry| &telemetry.stage_timings);
        println!(
            "rep={repetition} ffn_ms={:.3} gemv_ms={:.3} attention_ms={:.3} lm_head_ms={:.3} wall_clock_ms={wall_clock_decode_ms:.3}",
            stages.and_then(|s| s.ffn.elapsed_ms).unwrap_or_default(),
            stages
                .and_then(|s| s.gemv_gemm.elapsed_ms)
                .unwrap_or_default(),
            stages
                .and_then(|s| s.attention.elapsed_ms)
                .unwrap_or_default(),
            stages
                .and_then(|s| s.lm_head.elapsed_ms)
                .unwrap_or_default(),
        );
        runs.push(MeasuredRun {
            repetition,
            generated_tokens,
            wall_clock_decode_ms,
            decode_telemetry,
        });
    }

    let report = IsolationReport {
        schema_version: 1,
        model_path: model.display().to_string(),
        repetitions: REPETITIONS,
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: GENERATED_TOKENS,
        warmup: Warmup {
            generated_tokens: warmup_tokens,
            wall_clock_decode_ms: warmup_ms,
        },
        runs,
        command: "cargo test -p engine-server --test ffn_3b_isolation -- --ignored --nocapture"
            .to_string(),
        configuration: Configuration {
            prompt: PROMPT,
            native_nvrtc_preflight: true,
            dispatch_telemetry: true,
            max_sequence: 512,
            sampler: "greedy (temperature=0, top_p=1, top_k=1, repetition_penalty=1)",
            comparison_runtime: None,
        },
    };
    let output = artifact_path();
    fs::create_dir_all(output.parent().expect("artifact parent"))?;
    fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("wrote {}", output.display());
    Ok(())
}
