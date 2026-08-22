use engine_io::GgufReader;
use std::path::PathBuf;

fn get_fixture_path() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        manifest_dir.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
    ];
    for c in &candidates {
        if c.exists() {
            return Some(c.canonicalize().unwrap_or_else(|_| c.clone()));
        }
    }
    None
}

#[test]
fn test_3_1_parse_header_of_fixture() {
    let Some(fixture) = get_fixture_path() else {
        eprintln!("SKIP: fixture not present in this environment");
        return;
    };

    let reader = GgufReader::open(&fixture).expect("Failed to open GGUF file");
    let header = reader.header();

    assert_eq!(header.magic, "GGUF", "Magic header must be 'GGUF'");
    assert_eq!(header.version, 3, "GGUF version must be 3");
    assert!(
        header.tensor_count > 0,
        "Tensor count must be > 0, got {}",
        header.tensor_count
    );
    assert!(
        header.metadata_kv_count > 0,
        "Metadata KV count must be > 0, got {}",
        header.metadata_kv_count
    );
}

#[test]
fn test_3_3_tensor_infos_and_spans() {
    let Some(fixture) = get_fixture_path() else {
        eprintln!("SKIP: fixture not present in this environment");
        return;
    };
    let file_size = std::fs::metadata(&fixture).expect("Read metadata").len();

    let reader = GgufReader::open(&fixture).expect("Failed to open GGUF file");
    let tensors = reader.tensor_infos();
    let header = reader.header();

    assert_eq!(tensors.len(), header.tensor_count as usize);
    assert!(
        tensors.len() > 100,
        "Total tensor count should be > 100 for 0.6B model, got {}",
        tensors.len()
    );

    // Assert known tensor exists
    let token_embd = reader.get_tensor("token_embd.weight");
    assert!(
        token_embd.is_some(),
        "Known tensor 'token_embd.weight' must exist"
    );
    let embd_tensor = token_embd.unwrap();
    assert!(
        !embd_tensor.dims.is_empty(),
        "token_embd.weight dims must not be empty"
    );
    assert!(
        embd_tensor.size_bytes > 0,
        "token_embd.weight size_bytes must be > 0"
    );

    // Check all tensor spans
    let tensor_data_offset = reader.tensor_data_offset();
    assert!(tensor_data_offset > 0, "tensor_data_offset must be > 0");
    assert!(
        tensor_data_offset < file_size,
        "tensor_data_offset ({}) must be < file_size ({})",
        tensor_data_offset,
        file_size
    );

    let mut max_tensor_end = 0u64;
    let mut sum_tensor_bytes = 0u64;

    for t in tensors {
        assert!(!t.name.is_empty(), "Tensor name must not be empty");
        assert!(
            !t.dims.is_empty(),
            "Tensor dims must not be empty for {}",
            t.name
        );
        assert!(
            t.size_bytes > 0,
            "Tensor size_bytes must be > 0 for {}",
            t.name
        );

        let tensor_end = t.offset + t.size_bytes;
        if tensor_end > max_tensor_end {
            max_tensor_end = tensor_end;
        }
        sum_tensor_bytes += t.size_bytes;

        // Full span must fit within file size
        assert!(
            tensor_data_offset + tensor_end <= file_size,
            "Tensor {} span [{}..{}] exceeds file size {}",
            t.name,
            tensor_data_offset + t.offset,
            tensor_data_offset + tensor_end,
            file_size
        );
    }

    assert!(sum_tensor_bytes > 0);
    assert!(sum_tensor_bytes <= file_size);

    // Aligned header + tensor data area should reach the end of the file
    let total_file_span = tensor_data_offset + max_tensor_end;
    assert_eq!(
        total_file_span, file_size,
        "Aligned header ({}) + max tensor end ({}) = {} must equal file size {}",
        tensor_data_offset, max_tensor_end, total_file_span, file_size
    );
}

#[test]
fn test_3_5_layer_pattern_and_index() {
    use engine_io::classify_layer;

    // Unit test layer classification patterns
    assert_eq!(classify_layer("blk.0.attn_q.weight"), Some(0));
    assert_eq!(classify_layer("blk.1.attn_k.weight"), Some(1));
    assert_eq!(classify_layer("blk.27.ffn_down.weight"), Some(27));
    assert_eq!(classify_layer("token_embd.weight"), None);
    assert_eq!(classify_layer("output.weight"), None);
    assert_eq!(classify_layer("output_norm.weight"), None);
    assert_eq!(classify_layer("blk.invalid.weight"), None);
    assert_eq!(classify_layer("blk."), None);

    // Test layer index on real fixture
    let Some(fixture) = get_fixture_path() else {
        eprintln!("SKIP: fixture not present in this environment");
        return;
    };
    let reader = GgufReader::open(&fixture).expect("Failed to open GGUF file");
    let layer_idx = reader.layer_index();

    let layers = layer_idx.layers();
    assert!(!layers.is_empty(), "Model should have layers");
    // Verify layers are 0, 1, 2, ...
    for (i, &layer_num) in layers.iter().enumerate() {
        assert_eq!(
            i, layer_num,
            "Layer index must be consecutive starting at 0"
        );
    }

    let mut counted_layer_tensors = 0;
    for &idx in &layers {
        let layer_tensors = layer_idx
            .by_layer(idx)
            .expect("Layer tensors must exist for indexed layer");
        assert!(
            !layer_tensors.is_empty(),
            "Layer {} should have tensors",
            idx
        );
        for t in layer_tensors {
            let prefix = format!("blk.{}.", idx);
            assert!(
                t.name.starts_with(&prefix),
                "Tensor {} must start with {}",
                t.name,
                prefix
            );
        }
        counted_layer_tensors += layer_tensors.len();
    }

    let non_layer_tensors = layer_idx.non_layer_tensors();
    assert!(
        !non_layer_tensors.is_empty(),
        "Model should have non-layer tensors"
    );
    assert!(
        non_layer_tensors
            .iter()
            .any(|t| t.name == "token_embd.weight"),
        "non_layer_tensors must contain 'token_embd.weight'"
    );

    let all_tensors = layer_idx.tensors();
    assert_eq!(all_tensors.len(), reader.header().tensor_count as usize);
    assert_eq!(
        counted_layer_tensors + non_layer_tensors.len(),
        all_tensors.len(),
        "Sum of layer and non-layer tensors must equal total tensors"
    );
}

#[test]
fn test_metadata_qwen3() {
    let Some(fixture) = get_fixture_path() else {
        eprintln!("SKIP: fixture not present in this environment");
        return;
    };
    let reader = GgufReader::open(&fixture).expect("Failed to open GGUF file");
    let metadata = reader.metadata();

    assert!(!metadata.is_empty(), "Metadata should not be empty");

    // Architecture
    let arch = metadata
        .get("general.architecture")
        .expect("general.architecture must exist");
    assert_eq!(arch.as_str(), Some("qwen3"));

    // Quantization version
    if let Some(quant_ver) = metadata.get("general.quantization_version") {
        assert!(
            quant_ver.as_u32().is_some()
                || quant_ver.as_u64().is_some()
                || quant_ver.as_i32().is_some()
        );
    }

    // Tokenizer tokens array
    if let Some(tokens) = metadata.get("tokenizer.ggml.tokens") {
        let token_list = tokens.as_string_list();
        assert!(
            token_list.is_some(),
            "tokenizer.ggml.tokens must be a string array"
        );
        let list = token_list.unwrap();
        assert!(!list.is_empty(), "Token list must not be empty");
    }
}
