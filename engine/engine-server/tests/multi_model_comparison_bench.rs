use engine_core::{BpeTokenizer, DecodeTelemetry, ForwardDriver};
use engine_io::{GgufReader, ModelConfig, load_to_pinned};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

#[cfg(test)]
mod observability_tests {
    use super::{
        BenchResult, BenchmarkReport, BenchmarkStatistics, CacheCondition, ConditionStatistics,
        ConfigurationMetadata, EngineMeasurement, PromptResult, aggregate_repeated_runs,
        benchmark_model_matches_filter, benchmark_skip_llama, metric_statistics,
    };
    use engine_core::{DecodeStageTimings, DecodeTelemetry};

    #[test]
    fn serializes_stable_machine_readable_shape() {
        let result = PromptResult {
            prompt_index: 1,
            cache_condition: "cold".to_string(),
            prompt_tokens: 12,
            llama: Some(EngineMeasurement::fixture(10.0, 20.0)),
            titan: EngineMeasurement::fixture(12.0, 25.0),
            decode_ratio: Some(1.25),
        };
        let report = BenchmarkReport {
            schema_version: 1,
            configuration: ConfigurationMetadata {
                command: "cargo test --test multi_model_comparison_bench -- --ignored".to_string(),
                generated_tokens: 41,
                temperature: 0.0,
                llama_port: 8098,
                repetitions: 1,
            },
            results: vec![super::BenchmarkModelResult {
                model: "model".to_string(),
                model_path: "model.gguf".to_string(),
                prompts: vec![result],
                llama_decode_tok_s: Some(50.0),
                llama_decode_ms: Some(20.0),
                titan_decode_tok_s: 40.0,
                titan_decode_ms: 25.0,
                decode_ratio: Some(0.8),
                repetitions: 1,
                statistics: BenchmarkStatistics {
                    cold: ConditionStatistics {
                        llama_decode_tok_s: Some(metric_statistics(&[10.0])),
                        titan_decode_tok_s: metric_statistics(&[20.0]),
                        decode_ratio: Some(metric_statistics(&[1.25])),
                    },
                    warm: ConditionStatistics {
                        llama_decode_tok_s: Some(metric_statistics(&[10.0])),
                        titan_decode_tok_s: metric_statistics(&[20.0]),
                        decode_ratio: Some(metric_statistics(&[1.25])),
                    },
                },
                runs: Vec::new(),
            }],
        };
        let json = serde_json::to_value(report).expect("report must serialize");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["configuration"]["generated_tokens"], 41);
        assert_eq!(json["results"][0]["model"], "model");
        assert_eq!(json["results"][0]["prompts"][0]["cache_condition"], "cold");
        assert_eq!(
            json["results"][0]["prompts"][0]["llama"]["prefill_tok_s"],
            10.0
        );
        assert_eq!(json["results"][0]["prompts"][0]["titan"]["decode_ms"], 25.0);
        assert_eq!(json["results"][0]["prompts"][0]["decode_ratio"], 1.25);
    }

    #[test]
    fn serializes_decode_telemetry_with_measured_stages() {
        let telemetry = DecodeTelemetry::from_measured_stages(
            12.5,
            DecodeStageTimings::measured(2.0, 3.0, 4.0, 1.0, 1.5, 0.5),
            0.5,
            0.25,
            41,
        );
        let json = serde_json::to_value(telemetry).expect("telemetry must serialize");

        assert_eq!(json["decode_ms"], 12.5);
        assert_eq!(json["attribution"], "cuda_events_plus_host_boundaries");
        assert_eq!(json["stage_timings"]["gemv_gemm"]["elapsed_ms"], 2.0);
        assert_eq!(json["stage_timings"]["attention"]["elapsed_ms"], 3.0);
        assert_eq!(json["stage_timings"]["ffn"]["elapsed_ms"], 4.0);
        assert_eq!(json["stage_timings"]["lm_head"]["elapsed_ms"], 1.0);
        assert_eq!(json["stage_timings"]["copies"]["elapsed_ms"], 1.5);
        assert_eq!(json["stage_timings"]["waits"]["elapsed_ms"], 0.5);
        assert_eq!(json["overhead_ms"], 0.5);
        assert_eq!(json["reconciliation_tolerance_ms"], 0.25);
        assert_eq!(json["counters"]["graph_launches"], 41);
    }

    #[test]
    fn measured_telemetry_reconciles_total_with_named_overhead() {
        let telemetry = DecodeTelemetry::from_measured_stages(
            12.5,
            DecodeStageTimings::measured(2.0, 3.0, 4.0, 1.0, 1.5, 0.5),
            0.5,
            0.25,
            41,
        );

        assert!(telemetry.reconciles(0.001));
        assert_eq!(telemetry.stage_sum_ms(), 12.0);
    }

    #[test]
    fn overlapping_cuda_events_are_reported_without_negative_overhead() {
        let telemetry = DecodeTelemetry::from_measured_stages_with_accounting(
            10.0,
            DecodeStageTimings::measured(2.0, 3.0, 4.0, 2.0, 1.0, 5.0),
            41,
        );

        assert_eq!(telemetry.overhead_ms, 0.0);
        assert_eq!(telemetry.overlap_ms, 7.0);
        assert!(telemetry.reconciles(0.001));

        let json = serde_json::to_value(telemetry).expect("telemetry must serialize");
        assert_eq!(json["overhead_ms"], 0.0);
        assert_eq!(json["overlap_ms"], 7.0);
    }

    #[test]
    fn copy_bucket_excludes_host_queue_drain_from_async_copy_api() {
        let telemetry = DecodeTelemetry::from_measured_stages_with_accounting(
            629.0,
            DecodeStageTimings::measured(120.0, 180.0, 160.0, 100.0, 0.4, 0.1),
            41,
        );

        assert_eq!(telemetry.stage_timings.copies.elapsed_ms, Some(0.4));
        assert_eq!(telemetry.stage_timings.waits.elapsed_ms, Some(0.1));
        assert_eq!(telemetry.stage_timings.graph_replay.elapsed_ms, Some(0.0));
        assert!(telemetry.stage_timings.copies.elapsed_ms.unwrap() < 629.0 * 0.5);
        assert!(telemetry.overhead_ms >= 0.0);
        assert!(telemetry.reconciles(0.001));
    }

    #[test]
    fn graph_replay_boundary_is_explicitly_accounted() {
        let timings =
            DecodeStageTimings::measured_with_graph_replay(100.0, 120.0, 80.0, 20.0, 0.2, 0.1, 6.0);
        let telemetry = DecodeTelemetry::from_measured_stages_with_accounting(326.3, timings, 41);

        assert_eq!(telemetry.stage_timings.graph_replay.elapsed_ms, Some(6.0));
        assert!(telemetry.stage_timings.graph_replay.elapsed_ms.unwrap() >= 0.0);
        assert!(telemetry.reconciles(0.001));
        let json = serde_json::to_value(telemetry).expect("telemetry must serialize");
        assert_eq!(json["stage_timings"]["graph_replay"]["elapsed_ms"], 6.0);
    }

    #[test]
    fn default_decode_telemetry_preserves_explicit_stage_shape() {
        let timings = DecodeStageTimings::default();
        let json = serde_json::to_value(timings).expect("stage timings must serialize");

        assert_eq!(json["gemv_gemm"]["elapsed_ms"], serde_json::Value::Null);
        assert_eq!(json["gemv_gemm"]["status"], "not_applicable");
    }

    #[test]
    fn metric_statistics_reports_median_population_variance_and_stddev() {
        let stats = metric_statistics(&[1.0, 3.0, 5.0, 7.0]);

        assert_eq!(stats.samples, 4);
        assert_eq!(stats.median, 4.0);
        assert_eq!(stats.variance, 5.0);
        assert_eq!(stats.stddev, 5.0_f64.sqrt());
    }

    #[test]
    fn repeated_run_aggregation_keeps_cold_and_warm_conditions_separate() {
        let runs = vec![
            BenchResult::fixture(10.0, 20.0, 0.5, 0.8),
            BenchResult::fixture(14.0, 22.0, 0.7, 0.9),
            BenchResult::fixture(18.0, 24.0, 0.9, 1.0),
        ];

        let aggregate = aggregate_repeated_runs("model", "model.gguf", runs)
            .expect("three complete repetitions should aggregate");

        assert_eq!(aggregate.repetitions, 3);
        assert_eq!(aggregate.statistics.cold.titan_decode_tok_s.median, 22.0);
        assert_eq!(aggregate.statistics.warm.titan_decode_tok_s.median, 24.0);
        assert!(
            (aggregate
                .statistics
                .cold
                .decode_ratio
                .as_ref()
                .unwrap()
                .variance
                - 0.026666666666666672)
                .abs()
                < 1e-12
        );
        assert!(
            (aggregate
                .statistics
                .warm
                .decode_ratio
                .as_ref()
                .unwrap()
                .variance
                - 0.006666666666666668)
                .abs()
                < 1e-12
        );
        assert_eq!(aggregate.llama_decode_tok_s, Some(14.0));
    }

    #[test]
    fn cache_condition_serializes_explicitly() {
        let json = serde_json::to_value(CacheCondition::Cold).expect("condition serializes");
        assert_eq!(json, "cold");
        let json = serde_json::to_value(CacheCondition::Warm).expect("condition serializes");
        assert_eq!(json, "warm");
    }

    #[test]
    fn model_filter_matches_substrings_and_defaults_to_all_models() {
        assert!(benchmark_model_matches_filter(
            "Llama 3.2 3B Instruct",
            None
        ));
        assert!(benchmark_model_matches_filter(
            "Llama 3.2 3B Instruct",
            Some("Llama 3.2 3B")
        ));
        assert!(!benchmark_model_matches_filter(
            "Llama 3.2 1B Instruct",
            Some("Llama 3.2 3B")
        ));
    }

    #[test]
    fn skip_llama_requires_explicit_one_value() {
        assert!(benchmark_skip_llama(Some("1")));
        assert!(!benchmark_skip_llama(None));
        assert!(!benchmark_skip_llama(Some("true")));
        assert!(!benchmark_skip_llama(Some("0")));
    }

    #[test]
    fn skip_llama_with_filter_runs_only_matching_model_and_omits_llama_metrics() {
        let selected = ["Qwen 2.5 1.5B Instruct", "Llama 3.2 3B Instruct"]
            .into_iter()
            .filter(|name| benchmark_model_matches_filter(name, Some("Qwen")))
            .collect::<Vec<_>>();
        assert_eq!(selected, ["Qwen 2.5 1.5B Instruct"]);
        assert!(benchmark_skip_llama(Some("1")));

        let prompt = |index: usize, condition: &str| PromptResult {
            prompt_index: index,
            cache_condition: condition.to_string(),
            prompt_tokens: 4,
            llama: None,
            titan: EngineMeasurement::fixture(10.0, 20.0),
            decode_ratio: None,
        };
        let aggregate = aggregate_repeated_runs(
            "Qwen 2.5 1.5B Instruct",
            "qwen.gguf",
            vec![BenchResult {
                prompts: vec![prompt(1, "cold"), prompt(2, "warm")],
                llama_decode_toks: None,
                llama_decode_lat: None,
                titan_decode_toks: 50.0,
                titan_decode_lat: 20.0,
                ratio: None,
            }],
        )
        .expect("Titan-only run should aggregate");

        assert!(aggregate.llama_decode_tok_s.is_none());
        assert!(aggregate.decode_ratio.is_none());
        assert!(aggregate.statistics.cold.llama_decode_tok_s.is_none());
        assert!(aggregate.statistics.cold.decode_ratio.is_none());
        let json = serde_json::to_value(aggregate).expect("Titan-only report must serialize");
        assert!(json["prompts"][0]["llama"].is_null());
        assert!(json["prompts"][0].get("decode_ratio").is_none());
        assert!(json.get("llama_decode_tok_s").is_none());
        assert!(json.get("decode_ratio").is_none());
    }
}

fn resolve_model_path(rel: &str) -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../").join(rel),
        manifest_dir.join("../").join(rel),
        manifest_dir.join(rel),
        PathBuf::from(rel),
        PathBuf::from("../").join(rel),
    ];
    for c in &candidates {
        if c.exists() {
            return Some(c.canonicalize().unwrap_or_else(|_| c.clone()));
        }
    }
    None
}

#[derive(Serialize)]
struct LlamaReq<'a> {
    prompt: &'a str,
    n_predict: usize,
    temperature: f32,
}

#[derive(Deserialize, Debug)]
struct LlamaTimings {
    #[allow(dead_code)] // Retained for the external llama.cpp response schema.
    prompt_ms: f64,
    prompt_per_second: f64,
    predicted_n: usize,
    predicted_ms: f64,
    predicted_per_second: f64,
}

#[derive(Deserialize, Debug)]
struct LlamaResp {
    #[allow(dead_code)] // Retained for the external llama.cpp response schema.
    content: String,
    timings: LlamaTimings,
}

struct LlamaServerProcess {
    child: Child,
}

impl Drop for LlamaServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

async fn start_llama_server(
    model_p: &std::path::Path,
    port: u16,
) -> Result<Option<LlamaServerProcess>, Box<dyn std::error::Error>> {
    let ollama_dir = PathBuf::from(r"C:\Users\niber\AppData\Local\Programs\Ollama\lib\ollama");
    let cuda_dir = ollama_dir.join("cuda_v12");
    let mut server_exe = ollama_dir.join("llama-server.exe");
    let mut work_dir = cuda_dir.clone();

    if !server_exe.exists() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let bench_exe = manifest_dir.join("../../benchmarks/llama_cpp/llama-server.exe");
        if bench_exe.exists() {
            server_exe = bench_exe;
            work_dir = manifest_dir.join("../../benchmarks/llama_cpp");
        } else {
            let bench_exe2 = manifest_dir.join("../benchmarks/llama_cpp/llama-server.exe");
            if bench_exe2.exists() {
                server_exe = bench_exe2;
                work_dir = manifest_dir.join("../benchmarks/llama_cpp");
            } else {
                return Ok(None);
            }
        }
    }

    let path_env = format!(
        "{};{};{}",
        work_dir.display(),
        ollama_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );

    let child = Command::new(&server_exe)
        .current_dir(&work_dir)
        .env("PATH", &path_env)
        .arg("-m")
        .arg(model_p)
        .arg("-ngl")
        .arg("99")
        .arg("--port")
        .arg(port.to_string())
        .arg("-c")
        .arg("2048")
        .arg("--no-warmup")
        .spawn()?;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()?;
    let health_url = format!("http://127.0.0.1:{}/health", port);
    let start = Instant::now();
    let mut ready = false;

    while start.elapsed() < Duration::from_secs(15) {
        if let Ok(resp) = client.get(&health_url).send().await
            && resp.status().is_success()
        {
            ready = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    if !ready {
        return Ok(None);
    }

    Ok(Some(LlamaServerProcess { child }))
}

#[derive(Clone, Serialize)]
struct EngineMeasurement {
    prefill_tok_s: f64,
    decode_tok_s: f64,
    prefill_ms: f64,
    decode_ms: f64,
    latency_ms: f64,
    generated_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    telemetry: Option<DecodeTelemetry>,
}

impl EngineMeasurement {
    #[cfg(test)]
    fn fixture(prefill_tok_s: f64, decode_ms: f64) -> Self {
        Self {
            prefill_tok_s,
            decode_tok_s: 1_000.0 / decode_ms,
            prefill_ms: 100.0,
            decode_ms,
            latency_ms: 100.0 + decode_ms,
            generated_tokens: 41,
            telemetry: None,
        }
    }
}

#[derive(Clone, Serialize)]
struct PromptResult {
    prompt_index: usize,
    cache_condition: String,
    prompt_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    llama: Option<EngineMeasurement>,
    titan: EngineMeasurement,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_ratio: Option<f64>,
}

#[derive(Serialize)]
struct BenchmarkModelResult {
    model: String,
    model_path: String,
    prompts: Vec<PromptResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    llama_decode_tok_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    llama_decode_ms: Option<f64>,
    titan_decode_tok_s: f64,
    titan_decode_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_ratio: Option<f64>,
    repetitions: usize,
    statistics: BenchmarkStatistics,
    runs: Vec<BenchmarkRun>,
}

#[derive(Serialize)]
struct MetricStatistics {
    samples: usize,
    median: f64,
    variance: f64,
    stddev: f64,
}

#[derive(Serialize)]
struct ConditionStatistics {
    #[serde(skip_serializing_if = "Option::is_none")]
    llama_decode_tok_s: Option<MetricStatistics>,
    titan_decode_tok_s: MetricStatistics,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_ratio: Option<MetricStatistics>,
}

#[derive(Serialize)]
struct BenchmarkStatistics {
    cold: ConditionStatistics,
    warm: ConditionStatistics,
}

#[derive(Serialize)]
struct BenchmarkRun {
    repetition: usize,
    prompts: Vec<PromptResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    llama_decode_tok_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    llama_decode_ms: Option<f64>,
    titan_decode_tok_s: f64,
    titan_decode_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_ratio: Option<f64>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum CacheCondition {
    Cold,
    Warm,
}

#[derive(Serialize)]
struct ConfigurationMetadata {
    command: String,
    generated_tokens: usize,
    temperature: f32,
    llama_port: u16,
    repetitions: usize,
}

#[derive(Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    configuration: ConfigurationMetadata,
    results: Vec<BenchmarkModelResult>,
}

struct BenchResult {
    prompts: Vec<PromptResult>,
    llama_decode_toks: Option<f64>,
    llama_decode_lat: Option<f64>,
    titan_decode_toks: f64,
    titan_decode_lat: f64,
    ratio: Option<f64>,
}

#[cfg(test)]
impl BenchResult {
    fn fixture(llama: f64, titan: f64, cold_ratio: f64, warm_ratio: f64) -> Self {
        let prompt =
            |index: usize, condition: &str, llama_speed: f64, titan_speed: f64, ratio: f64| {
                PromptResult {
                    prompt_index: index,
                    cache_condition: condition.to_string(),
                    prompt_tokens: 1,
                    llama: Some(EngineMeasurement::fixture(
                        llama_speed,
                        1_000.0 / llama_speed,
                    )),
                    titan: EngineMeasurement::fixture(titan_speed, 1_000.0 / titan_speed),
                    decode_ratio: Some(ratio),
                }
            };
        Self {
            prompts: vec![
                prompt(1, "cold", llama, titan, cold_ratio),
                prompt(2, "warm", llama + 2.0, titan + 2.0, warm_ratio),
            ],
            llama_decode_toks: Some(llama),
            llama_decode_lat: Some(1.0),
            titan_decode_toks: titan,
            titan_decode_lat: 1.0,
            ratio: Some(titan / llama),
        }
    }
}

fn metric_statistics(values: &[f64]) -> MetricStatistics {
    assert!(!values.is_empty(), "benchmark statistics require samples");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    };
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    MetricStatistics {
        samples: values.len(),
        median,
        variance,
        stddev: variance.sqrt(),
    }
}

fn condition_statistics(runs: &[BenchResult], condition: &str) -> ConditionStatistics {
    let samples = runs
        .iter()
        .flat_map(|run| run.prompts.iter())
        .filter(|prompt| prompt.cache_condition == condition)
        .collect::<Vec<_>>();
    ConditionStatistics {
        llama_decode_tok_s: {
            let values = samples
                .iter()
                .filter_map(|p| p.llama.as_ref().map(|m| m.decode_tok_s))
                .collect::<Vec<_>>();
            (!values.is_empty()).then(|| metric_statistics(&values))
        },
        titan_decode_tok_s: metric_statistics(
            &samples
                .iter()
                .map(|p| p.titan.decode_tok_s)
                .collect::<Vec<_>>(),
        ),
        decode_ratio: {
            let values = samples
                .iter()
                .filter_map(|p| p.decode_ratio)
                .collect::<Vec<_>>();
            (!values.is_empty()).then(|| metric_statistics(&values))
        },
    }
}

fn aggregate_repeated_runs(
    model_name: &str,
    model_path: &str,
    runs: Vec<BenchResult>,
) -> Result<BenchmarkModelResult, Box<dyn std::error::Error>> {
    if runs.is_empty() {
        return Err("benchmark requires at least one repetition".into());
    }
    let first = runs.first().expect("non-empty runs");
    let median = |values: Vec<f64>| metric_statistics(&values).median;
    let optional_median =
        |values: Vec<f64>| (!values.is_empty()).then(|| metric_statistics(&values).median);
    Ok(BenchmarkModelResult {
        model: model_name.to_string(),
        model_path: model_path.to_string(),
        prompts: first.prompts.clone(),
        llama_decode_tok_s: optional_median(
            runs.iter().filter_map(|r| r.llama_decode_toks).collect(),
        ),
        llama_decode_ms: optional_median(runs.iter().filter_map(|r| r.llama_decode_lat).collect()),
        titan_decode_tok_s: median(runs.iter().map(|r| r.titan_decode_toks).collect()),
        titan_decode_ms: median(runs.iter().map(|r| r.titan_decode_lat).collect()),
        decode_ratio: optional_median(runs.iter().filter_map(|r| r.ratio).collect()),
        repetitions: runs.len(),
        statistics: BenchmarkStatistics {
            cold: condition_statistics(&runs, "cold"),
            warm: condition_statistics(&runs, "warm"),
        },
        runs: runs
            .iter()
            .enumerate()
            .map(|(index, run)| BenchmarkRun {
                repetition: index + 1,
                prompts: run.prompts.clone(),
                llama_decode_tok_s: run.llama_decode_toks,
                llama_decode_ms: run.llama_decode_lat,
                titan_decode_tok_s: run.titan_decode_toks,
                titan_decode_ms: run.titan_decode_lat,
                decode_ratio: run.ratio,
            })
            .collect(),
    })
}

fn benchmark_repetitions() -> usize {
    std::env::var("TITAN_BENCHMARK_REPETITIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(3)
}

fn benchmark_model_filter() -> Option<String> {
    std::env::var("TITAN_BENCHMARK_MODEL_FILTER")
        .ok()
        .filter(|filter| !filter.is_empty())
}

fn benchmark_model_matches_filter(model_name: &str, filter: Option<&str>) -> bool {
    filter.is_none_or(|filter| model_name.contains(filter))
}

fn benchmark_skip_llama(value: Option<&str>) -> bool {
    value == Some("1")
}

async fn run_model_benchmark(
    name: &str,
    p: PathBuf,
    test_prompts: &[&str],
    port: u16,
    skip_llama: bool,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    println!("\n================================================================================");
    println!(">>> BENCHMARKING MODEL: {} ({})", name, p.display());
    println!("================================================================================");

    let n_tokens_to_generate = 41;

    // 1. LLAMA.CPP
    if skip_llama {
        println!("[1/2] llama.cpp benchmark unavailable (TITAN_BENCHMARK_SKIP_LLAMA=1)");
    } else {
        println!("[1/2] Benchmarking official llama.cpp server...");
    }
    let mut llama_measurements = Vec::new();

    let llama_proc = if skip_llama {
        None
    } else {
        start_llama_server(&p, port).await?
    };
    if llama_proc.is_some() {
        let client = reqwest::Client::new();
        let comp_url = format!("http://127.0.0.1:{}/completion", port);

        for (i, &prompt) in test_prompts.iter().enumerate() {
            let req = LlamaReq {
                prompt,
                n_predict: n_tokens_to_generate,
                temperature: 0.0,
            };
            let body_str = serde_json::to_string(&req)?;
            let resp_text = client
                .post(&comp_url)
                .header("content-type", "application/json")
                .body(body_str)
                .send()
                .await?
                .text()
                .await?;
            let resp: LlamaResp = serde_json::from_str(&resp_text)?;

            let decode_ms = resp.timings.predicted_ms;
            let latency_ms = resp.timings.prompt_ms + decode_ms;
            println!(
                "  llama.cpp Prompt #{}: Prefill = {:.1} tok/s, Decode = {:.1} tok/s ({:.2} ms/tok)",
                i + 1,
                resp.timings.prompt_per_second,
                resp.timings.predicted_per_second,
                decode_ms / resp.timings.predicted_n.max(1) as f64
            );
            llama_measurements.push(EngineMeasurement {
                prefill_tok_s: resp.timings.prompt_per_second,
                decode_tok_s: resp.timings.predicted_per_second,
                prefill_ms: resp.timings.prompt_ms,
                decode_ms,
                latency_ms,
                generated_tokens: resp.timings.predicted_n,
                telemetry: None,
            });
        }
    } else {
        println!("  [!] llama.cpp server failed to start for {}", name);
    }
    drop(llama_proc);

    // Give time for port and memory cleanup
    if !skip_llama {
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // 2. TITAN GPU RESIDENT
    println!("\n[2/2] Benchmarking Titan GPU Resident Engine (100% Pure Rust / CUDA JIT)...");
    let reader = GgufReader::open(&p)?;
    let cfg = ModelConfig::from_reader(&reader)?;
    let pinned = load_to_pinned(&reader, &p)?;
    let tokenizer = BpeTokenizer::from_reader(&reader)?;
    let mut driver = ForwardDriver::new(&reader, &pinned, &cfg, 512)?;
    if std::env::var("TITAN_BENCHMARK_DISPATCH_TELEMETRY").as_deref() == Ok("1") {
        driver.enable_dispatch_telemetry();
    }
    let _ = driver.capture_autonomous_decode_graph();

    let mut titan_measurements = Vec::new();

    for (i, &raw_prompt) in test_prompts.iter().enumerate() {
        driver.reset_pos();
        let prompt_tokens = tokenizer.encode(raw_prompt)?;

        let t_request = Instant::now();
        // Prefill
        let t_prefill = Instant::now();
        let initial_logits = driver.prefill(&prompt_tokens)?;
        let prefill_dur = t_prefill.elapsed();

        let cur_token = {
            let mut best_i = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &v) in initial_logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_i = idx;
                }
            }
            best_i as u32
        };

        let _ = driver.decode(cur_token)?;

        let t_decode = Instant::now();
        let gen_tokens = driver.generate_autonomous_gpu(cur_token, n_tokens_to_generate)?;
        let telemetry = driver.take_decode_telemetry();
        let decode_dur = t_decode.elapsed();
        let decode_speed = gen_tokens.len() as f64 / decode_dur.as_secs_f64();
        let decode_lat_ms = (decode_dur.as_secs_f64() * 1000.0) / gen_tokens.len() as f64;

        titan_measurements.push(EngineMeasurement {
            prefill_tok_s: prompt_tokens.len() as f64 / prefill_dur.as_secs_f64(),
            decode_tok_s: decode_speed,
            prefill_ms: prefill_dur.as_secs_f64() * 1000.0,
            decode_ms: decode_dur.as_secs_f64() * 1000.0,
            latency_ms: t_request.elapsed().as_secs_f64() * 1000.0,
            generated_tokens: gen_tokens.len(),
            telemetry,
        });

        println!(
            "  Titan Prompt #{}: GPU Stream Decode = {:.1} tok/s ({:.2} ms/tok)",
            i + 1,
            decode_speed,
            decode_lat_ms
        );
    }

    let avg_llama_decode: Option<f64> = if !llama_measurements.is_empty() {
        Some(
            llama_measurements
                .iter()
                .map(|m| m.decode_tok_s)
                .sum::<f64>()
                / llama_measurements.len() as f64,
        )
    } else {
        None
    };
    let avg_llama_lat: Option<f64> = if !llama_measurements.is_empty() {
        Some(
            llama_measurements
                .iter()
                .map(|m| m.decode_ms / m.generated_tokens.max(1) as f64)
                .sum::<f64>()
                / llama_measurements.len() as f64,
        )
    } else {
        None
    };

    let avg_titan_decode: f64 = titan_measurements
        .iter()
        .map(|m| m.decode_tok_s)
        .sum::<f64>()
        / titan_measurements.len() as f64;
    let avg_titan_lat: f64 = titan_measurements
        .iter()
        .map(|m| m.decode_ms / m.generated_tokens.max(1) as f64)
        .sum::<f64>()
        / titan_measurements.len() as f64;

    let prompts = test_prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            let llama = llama_measurements.get(index).cloned();
            let titan = titan_measurements[index].clone();
            Ok(PromptResult {
                prompt_index: index + 1,
                cache_condition: if index == 0 { "cold" } else { "warm" }.to_string(),
                prompt_tokens: tokenizer.encode(prompt)?.len(),
                decode_ratio: llama
                    .as_ref()
                    .map(|measurement| titan.decode_tok_s / measurement.decode_tok_s),
                llama,
                titan,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

    Ok(BenchResult {
        prompts,
        llama_decode_toks: avg_llama_decode,
        llama_decode_lat: avg_llama_lat,
        titan_decode_toks: avg_titan_decode,
        titan_decode_lat: avg_titan_lat,
        ratio: avg_llama_decode.map(|llama| avg_titan_decode / llama),
    })
}

#[tokio::test]
#[ignore]
async fn test_multi_model_head_to_head_bench() -> Result<(), Box<dyn std::error::Error>> {
    engine_cuda::ensure_cuda_dll_paths();
    let nvrtc_dll = engine_cuda::nvrtc_dll_preflight()
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::NotFound, message))?;
    println!("NVRTC preflight passed: {}", nvrtc_dll.display());

    let models = [
        (
            "Qwen 2.5 1.5B Instruct",
            "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            [
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nGive me a numbered list of 5 historical dates.<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain quantum computing in three concise bullet points.<|im_end|>\n<|im_start|>assistant\n",
            ],
        ),
        (
            "Llama 3.2 1B Instruct",
            "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            [
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nGive me a numbered list of 5 historical dates.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nExplain quantum computing in three concise bullet points.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
            ],
        ),
        (
            "Llama 3.2 3B Instruct",
            "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            [
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nGive me a numbered list of 5 historical dates.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nExplain quantum computing in three concise bullet points.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
            ],
        ),
        (
            "DeepSeek-R1-Distill 1.5B",
            "models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
            [
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nGive me a numbered list of 5 historical dates.<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain quantum computing in three concise bullet points.<|im_end|>\n<|im_start|>assistant\n",
            ],
        ),
        (
            "Qwen3 0.6B Base/Chat",
            "testdata/Qwen3-0.6B-Q4_K_M.gguf",
            [
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nGive me a numbered list of 5 historical dates.<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain quantum computing in three concise bullet points.<|im_end|>\n<|im_start|>assistant\n",
            ],
        ),
    ];

    let repetitions = benchmark_repetitions();
    let model_filter = benchmark_model_filter();
    let skip_llama =
        benchmark_skip_llama(std::env::var("TITAN_BENCHMARK_SKIP_LLAMA").ok().as_deref());
    println!(
        "Configured benchmark repetitions per model: {}",
        repetitions
    );
    if let Some(filter) = &model_filter {
        println!("Configured model filter: {}", filter);
    }
    if skip_llama {
        println!("Configured to skip llama.cpp: TITAN_BENCHMARK_SKIP_LLAMA=1");
    }
    let mut results = Vec::new();
    let port = 8098;

    for (name, rel_path, prompts) in &models {
        if !benchmark_model_matches_filter(name, model_filter.as_deref()) {
            println!("SKIP: {} (model filter: no substring match)", name);
            continue;
        }
        if let Some(path) = resolve_model_path(rel_path) {
            let mut runs = Vec::with_capacity(repetitions);
            for repetition in 1..=repetitions {
                println!(
                    "\n--- {} repetition {}/{} ---",
                    name, repetition, repetitions
                );
                runs.push(
                    run_model_benchmark(name, path.clone(), prompts, port, skip_llama).await?,
                );
            }
            results.push(aggregate_repeated_runs(
                name,
                &path.display().to_string(),
                runs,
            )?);
        } else {
            println!("SKIP: {} (path not found: {})", name, rel_path);
        }
    }

    println!(
        "\n========================================================================================================="
    );
    println!(
        "===                                MULTI-MODEL HEAD-TO-HEAD COMPARISON                                ==="
    );
    println!(
        "========================================================================================================="
    );
    println!(
        "{:<28} | {:<20} | {:<20} | {:<12}",
        "Model Name", "llama.cpp (C++)", "Titan (Pure Rust)", "Ratio (Titan / llama.cpp)"
    );
    println!("{:-<28}-|-{:-<20}-|-{:-<20}-|-{:-<12}", "", "", "", "");

    for r in &results {
        let cold = &r.statistics.cold;
        let warm = &r.statistics.warm;
        if let (Some(cold_llama), Some(cold_ratio), Some(warm_llama), Some(warm_ratio)) = (
            cold.llama_decode_tok_s.as_ref(),
            cold.decode_ratio.as_ref(),
            warm.llama_decode_tok_s.as_ref(),
            warm.decode_ratio.as_ref(),
        ) {
            println!(
                "{:<28} | cold n={} llama {:<6.1} tok/s | Titan {:<6.1} tok/s | {:<6.2}x ±{:<5.2}",
                r.model,
                cold_llama.samples,
                cold_llama.median,
                cold.titan_decode_tok_s.median,
                cold_ratio.median,
                cold_ratio.stddev
            );
            println!(
                "{:<28} | warm n={} llama {:<6.1} tok/s | Titan {:<6.1} tok/s | {:<6.2}x ±{:<5.2}",
                "",
                warm_llama.samples,
                warm_llama.median,
                warm.titan_decode_tok_s.median,
                warm_ratio.median,
                warm_ratio.stddev
            );
        } else {
            println!(
                "{:<28} | cold llama n/a | Titan {:<6.1} tok/s | ratio n/a",
                r.model, cold.titan_decode_tok_s.median
            );
            println!(
                "{:<28} | warm llama n/a | Titan {:<6.1} tok/s | ratio n/a",
                "", warm.titan_decode_tok_s.median
            );
        }
    }
    println!(
        "=========================================================================================================\n"
    );

    let report = BenchmarkReport {
        schema_version: 1,
        configuration: ConfigurationMetadata {
            command: std::env::args().collect::<Vec<_>>().join(" "),
            generated_tokens: 41,
            temperature: 0.0,
            llama_port: port,
            repetitions,
        },
        results: results.into_iter().collect(),
    };
    let output_path = std::env::var_os("TITAN_BENCHMARK_JSON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                r"C:\Users\niber\AppData\Local\hermes\workspace\titan\local-artifacts\benchmarks\multi_model_comparison.json",
            )
        });
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_vec_pretty(&report)?;
    fs::write(&output_path, json)?;
    println!(
        "Machine-readable benchmark results: {}",
        output_path.display()
    );

    Ok(())
}
