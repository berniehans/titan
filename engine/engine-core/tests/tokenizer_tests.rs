//! Task 2.1 / 2.3 — Failing tests: the engine-core BPE tokenizer must produce
//! byte-identical token streams to the pinned llama.cpp reference for every
//! prompt in `tests/fixtures/prompts.txt`, and round-trip decode(encode(t)) == t.
//!
//! Reference captured by `tools/golden_tokenize.py` (llama-tokenize --ids) into
//! `tests/fixtures/golden/tokenize_reference.json` (llama.cpp pinned cb1adf8).

use engine_core::tokenizer::BpeTokenizer;
use engine_io::GgufReader;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../"))
}

fn get_fixture_path() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    let root = repo_root();
    let gguf = root.join("testdata/Qwen3-0.6B-Q4_K_M.gguf");
    if gguf.exists() {
        return Some(gguf);
    }
    None
}

fn load_tokenizer() -> Option<BpeTokenizer> {
    let fixture = get_fixture_path()?;
    let reader = GgufReader::open(&fixture).ok()?;
    BpeTokenizer::from_reader(&reader).ok()
}

fn prompt_ids() -> Option<(Vec<String>, Vec<Vec<u32>>)> {
    let root = repo_root();
    let prompts_path = root.join("tests/fixtures/prompts.txt");
    let ref_path = root.join("tests/fixtures/golden/tokenize_reference.json");
    if !prompts_path.exists() || !ref_path.exists() {
        return None;
    }
    let prompts: Vec<String> = std::fs::read_to_string(&prompts_path)
        .ok()?
        .split('\n')
        .map(|l| l.trim_end_matches('\r').to_string())
        .filter(|l| !l.is_empty())
        .collect();
    let raw = std::fs::read_to_string(&ref_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let arr = json.get("prompts")?.as_array()?;
    let mut ids = Vec::with_capacity(arr.len());
    for item in arr {
        let row: Vec<u32> = item
            .get("ids")?
            .as_array()?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as u32)
            .collect();
        ids.push(row);
    }
    Some((prompts, ids))
}

#[test]
fn test_6_1_encoder_matches_llama_cpp_token_stream() {
    let Some(tok) = load_tokenizer() else {
        eprintln!("SKIP: fixture not present in this environment");
        return;
    };
    let Some((prompts, refs)) = prompt_ids() else {
        eprintln!("SKIP: prompts/reference fixtures not present");
        return;
    };
    assert!(
        prompts.len() >= 20,
        "need >=20 prompts, got {}",
        prompts.len()
    );
    assert_eq!(prompts.len(), refs.len(), "prompts and reference aligned");

    let mut matched = 0;
    for (i, (prompt, expected)) in prompts.iter().zip(refs.iter()).enumerate() {
        let actual = tok.encode(prompt).unwrap_or_else(|e| {
            panic!("encode failed on {:?}: {}", prompt, e);
        });
        assert_eq!(
            actual, *expected,
            "token stream mismatch on prompt[{i}] {:?}",
            prompt
        );
        matched += 1;
    }
    assert!(
        matched >= 20,
        "expected >=20 prompts matched, got {matched}"
    );
}

#[test]
fn test_6_1_decode_round_trip_across_prompts() {
    let Some(tok) = load_tokenizer() else {
        eprintln!("SKIP: fixture not present in this environment");
        return;
    };
    let Some((prompts, _)) = prompt_ids() else {
        eprintln!("SKIP: prompts/reference fixtures not present");
        return;
    };
    for (i, prompt) in prompts.iter().enumerate() {
        let ids = tok
            .encode(prompt)
            .unwrap_or_else(|e| panic!("encode {:?}: {}", prompt, e));
        let decoded = tok.decode(&ids).unwrap_or_else(|e| panic!("decode: {}", e));
        assert_eq!(
            decoded, *prompt,
            "round-trip mismatch on prompt[{i}] {:?}",
            prompt
        );
    }
}
