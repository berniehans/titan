use engine_io::{GgmlType, GgufReader, LoadedLayout};
use std::fs::File;
use std::io::Write;
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
fn test_5_1_loaded_layout_accounting() {
    let Some(fixture) = get_fixture_path() else {
        eprintln!("SKIP: fixture not present in this environment");
        return;
    };
    let file_size = std::fs::metadata(&fixture).expect("Read metadata").len();

    let reader = GgufReader::open(&fixture).expect("Failed to open GGUF file");
    let layout = LoadedLayout::from_reader(&reader).expect("Failed to create LoadedLayout");

    assert!(
        layout.total_size_bytes() > 0,
        "total_size_bytes must be > 0"
    );

    // Key assertion: metadata size + loaded data size == total file size
    assert_eq!(
        layout.total_size_bytes() + reader.tensor_data_offset(),
        file_size,
        "Total layout bytes ({}) + tensor_data_offset ({}) must equal file_size ({})",
        layout.total_size_bytes(),
        reader.tensor_data_offset(),
        file_size
    );

    // Sum over all tensors of size_bytes == layout.total_size_bytes() (contiguous in data blob)
    let sum_tensor_bytes: u64 = reader.tensor_infos().iter().map(|t| t.size_bytes).sum();
    assert_eq!(
        sum_tensor_bytes,
        layout.total_size_bytes(),
        "Sum of all tensor size_bytes must equal total_size_bytes"
    );

    // Assert tensor_span lookup
    let embd_span = layout.tensor_span("token_embd.weight");
    assert!(embd_span.is_some(), "token_embd.weight span must exist");
    let (embd_offset, embd_size) = embd_span.unwrap();
    let embd_info = reader.get_tensor("token_embd.weight").unwrap();
    assert_eq!(embd_offset, embd_info.offset);
    assert_eq!(embd_size, embd_info.size_bytes);

    let norm_span = layout.tensor_span("output_norm.weight");
    assert!(norm_span.is_some(), "output_norm.weight span must exist");
    let (norm_offset, norm_size) = norm_span.unwrap();
    let norm_info = reader.get_tensor("output_norm.weight").unwrap();
    assert_eq!(norm_offset, norm_info.offset);
    assert_eq!(norm_size, norm_info.size_bytes);

    // Non-existent tensor returns None
    assert_eq!(layout.tensor_span("non_existent_tensor"), None);

    // Assert layer_spans and layer_range lookup for all layers
    let layers = reader.layer_index().layers();
    assert!(!layers.is_empty(), "Model must have layers");

    for &layer_num in &layers {
        let layer_spans = layout
            .layer_spans(layer_num)
            .expect("Layer spans must exist");
        let layer_tensors = reader.layer_index().by_layer(layer_num).unwrap();
        assert_eq!(layer_spans.len(), layer_tensors.len());

        let mut expected_layer_size = 0u64;
        for (span, tensor) in layer_spans.iter().zip(layer_tensors.iter()) {
            assert_eq!(span.0, tensor.offset);
            assert_eq!(span.1, tensor.size_bytes);
            expected_layer_size += tensor.size_bytes;
        }

        let (range_start, range_size) = layout
            .layer_range(layer_num)
            .expect("Layer range must exist");
        assert_eq!(range_start, layer_tensors[0].offset);
        assert_eq!(range_size, expected_layer_size);
    }

    // Non-existent layer returns None
    assert_eq!(layout.layer_spans(999999), None);
    assert_eq!(layout.layer_range(999999), None);
}

#[test]
fn test_5_1_synthetic_layout_accounting() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join(format!("test_synth_layout_{}.gguf", std::process::id()));

    // Create a synthetic GGUF file with 2 layers and 1 non-layer tensor
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
    buf.extend_from_slice(&3u64.to_le_bytes()); // 3 tensors
    buf.extend_from_slice(&0u64.to_le_bytes()); // 0 KV entries

    // Tensor 0: token_embd.weight, dims: [4], type: F32 (0), offset: 0, size: 16
    let t0_name = "token_embd.weight";
    buf.extend_from_slice(&(t0_name.len() as u64).to_le_bytes());
    buf.extend_from_slice(t0_name.as_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // 1 dim
    buf.extend_from_slice(&4u64.to_le_bytes());
    buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset 0

    // Tensor 1: blk.0.attn_q.weight, dims: [4], type: F32 (0), offset: 16, size: 16
    let t1_name = "blk.0.attn_q.weight";
    buf.extend_from_slice(&(t1_name.len() as u64).to_le_bytes());
    buf.extend_from_slice(t1_name.as_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&4u64.to_le_bytes());
    buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
    buf.extend_from_slice(&16u64.to_le_bytes()); // offset 16

    // Tensor 2: blk.1.attn_k.weight, dims: [8], type: F32 (0), offset: 32, size: 32
    let t2_name = "blk.1.attn_k.weight";
    buf.extend_from_slice(&(t2_name.len() as u64).to_le_bytes());
    buf.extend_from_slice(t2_name.as_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&8u64.to_le_bytes());
    buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
    buf.extend_from_slice(&32u64.to_le_bytes()); // offset 32

    // Align to 32 bytes
    let cur = buf.len();
    let aligned = (cur + 31) & !31;
    buf.resize(aligned, 0);

    // Tensor data: 16 + 16 + 32 = 64 bytes
    buf.resize(aligned + 64, 0xAA);

    let mut file = File::create(&file_path).expect("Create synthetic file");
    file.write_all(&buf).expect("Write synthetic file");

    let reader = GgufReader::open(&file_path).expect("Open synthetic file");
    let layout = LoadedLayout::from_reader(&reader).expect("LoadedLayout from synthetic");

    assert_eq!(layout.total_size_bytes(), 64);
    assert_eq!(layout.tensor_span("token_embd.weight"), Some((0, 16)));
    assert_eq!(layout.tensor_span("blk.0.attn_q.weight"), Some((16, 16)));
    assert_eq!(layout.tensor_span("blk.1.attn_k.weight"), Some((32, 32)));

    assert_eq!(layout.layer_spans(0), Some(&[(16, 16)][..]));
    assert_eq!(layout.layer_range(0), Some((16, 16)));
    assert_eq!(layout.layer_spans(1), Some(&[(32, 32)][..]));
    assert_eq!(layout.layer_range(1), Some((32, 32)));
    assert_eq!(layout.layer_spans(2), None);

    let _ = std::fs::remove_file(&file_path);
}
