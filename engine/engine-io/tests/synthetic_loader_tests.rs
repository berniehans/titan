//! Phase 6.2 — synthetic minimal GGUF loader round-trip.
//!
//! Verifies that `engine-io` loads the committed synthetic model fixture
//! `tests/fixtures/synthetic/synthetic_min.gguf` (built deterministically by
//! `tools/build_synthetic_gguf.py`) and resolves every tensor to its shape
//! (`dims`) and type (`GgmlType`), plus metadata/config round-trips.
//!
//! The fixture mirrors the real Qwen3 GGUF naming/layout conventions
//! (see `openspec/changes/6.2-cpu-reference-bank/proposal.md`):
//! `blk.N.*` layer tensors, `token_embd.weight`, `output_norm.weight`, and the
//! `qwen3.*` metadata keys that `ModelConfig` reads. Formats carried:
//! Q4_K (attentions, down), Q8_0 (mid FFN), F32 (embedding / norms / head).

use engine_io::{GgmlType, GgufError, GgufReader, LoadedLayout, ModelConfig};
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // engine/engine-io -> repo root (../../), then tests/fixtures/synthetic/
    manifest_dir.join("../../tests/fixtures/synthetic/synthetic_min.gguf")
}

const EXPECTED_TENSORS: &[(&str, GgmlType, &[u64])] = &[
    ("output_norm.weight", GgmlType::F32, &[256]),
    ("token_embd.weight", GgmlType::F32, &[256, 16]),
    ("blk.0.attn_norm.weight", GgmlType::F32, &[256]),
    ("blk.0.attn_q.weight", GgmlType::Q4_K, &[256, 256]),
    ("blk.0.attn_q_norm.weight", GgmlType::F32, &[128]),
    ("blk.0.attn_k.weight", GgmlType::Q4_K, &[256, 128]),
    ("blk.0.attn_k_norm.weight", GgmlType::F32, &[128]),
    ("blk.0.attn_v.weight", GgmlType::Q8_0, &[256, 128]),
    ("blk.0.attn_output.weight", GgmlType::Q4_K, &[256, 256]),
    ("blk.0.ffn_norm.weight", GgmlType::F32, &[256]),
    ("blk.0.ffn_gate.weight", GgmlType::Q8_0, &[256, 256]),
    ("blk.0.ffn_up.weight", GgmlType::Q8_0, &[256, 256]),
    ("blk.0.ffn_down.weight", GgmlType::Q4_K, &[256, 256]),
];

#[test]
fn synthetic_fixture_exists() {
    let path = fixture_path();
    assert!(
        path.exists(),
        "missing synthetic fixture at {} (regenerate with \
         `uv run python tools/build_synthetic_gguf.py`)",
        path.display()
    );
}

#[test]
fn synthetic_loads_and_roundtrips_tensor_shape_type() -> Result<(), GgufError> {
    let reader = GgufReader::open(fixture_path())?;

    // 2 non-layer + 2 layers * 11 tensors.
    assert_eq!(reader.header().tensor_count, 24);
    assert_eq!(reader.tensor_infos().len(), 24);

    // Round-trip each critical tensor: name -> dims + type.
    for (name, expected_type, expected_dims) in EXPECTED_TENSORS {
        let t = reader
            .get_tensor(name)
            .unwrap_or_else(|| panic!("expected tensor {name} present in synthetic fixture"));
        assert_eq!(
            t.ggml_type, *expected_type,
            "{name}: type mismatch (got {:?})",
            t.ggml_type
        );
        assert_eq!(t.dims, *expected_dims, "{name}: dims mismatch");
    }

    // Layer classification: layer 0 and 1 each hold 11 tensors.
    let layers = reader.layer_index().layers();
    assert_eq!(layers, vec![0, 1]);
    assert_eq!(reader.layer_index().by_layer(0).unwrap().len(), 11);
    assert_eq!(reader.layer_index().by_layer(1).unwrap().len(), 11);
    assert_eq!(reader.layer_index().non_layer_tensors().len(), 2);

    Ok(())
}

#[test]
fn synthetic_config_parses() -> Result<(), GgufError> {
    let reader = GgufReader::open(fixture_path())?;
    let cfg = ModelConfig::from_reader(&reader)?;
    assert_eq!(cfg.architecture, "qwen3");
    assert_eq!(cfg.n_layer, 2);
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.n_head, 2);
    assert_eq!(cfg.n_head_kv, 1);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.value_dim, 128);
    assert_eq!(cfg.intermediate_size, 256);
    assert_eq!(cfg.vocab_size, 16);
    assert_eq!(cfg.context_length, 64);
    assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-9);
    Ok(())
}

#[test]
fn synthetic_layout_accounting() -> Result<(), GgufError> {
    let reader = GgufReader::open(fixture_path())?;
    let layout = LoadedLayout::from_reader(&reader)?;

    // Contiguous blob that ends exactly at tensor_data_offset + total.
    assert_eq!(
        layout.total_size_bytes() + reader.tensor_data_offset(),
        std::fs::metadata(fixture_path()).map(|m| m.len()).unwrap(),
        "data blob + header must equal full file size"
    );

    // Sum of all tensor spans == total blob size.
    let sum: u64 = reader.tensor_infos().iter().map(|t| t.size_bytes).sum();
    assert_eq!(sum, layout.total_size_bytes());

    // Layer ranges are contiguous per layer.
    for layer in [0, 1] {
        let (start, size) = layout.layer_range(layer).unwrap();
        let tensors = reader.layer_index().by_layer(layer).unwrap();
        let exp_size: u64 = tensors.iter().map(|t| t.size_bytes).sum();
        assert_eq!(size, exp_size, "layer {layer} range size");
        assert_eq!(start, tensors.first().unwrap().offset);
    }

    // Spot-check one span.
    let (off, sz) = layout.tensor_span("blk.0.attn_q.weight").unwrap();
    assert_eq!(sz, 256 * 144);
    let info = reader.get_tensor("blk.0.attn_q.weight").unwrap();
    assert_eq!(off, info.offset);
    assert_eq!(sz, info.size_bytes);

    Ok(())
}
