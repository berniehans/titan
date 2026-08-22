use std::path::PathBuf;
use std::sync::Mutex;
use engine_cuda::PinnedHost;
use engine_io::{load_to_pinned, GgufReader};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn get_fixture_path() -> PathBuf {
    if let Ok(env_path) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return p;
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
            return c.canonicalize().unwrap_or_else(|_| c.clone());
        }
    }
    panic!("Fixture file Qwen3-0.6B-Q4_K_M.gguf not found at any candidate path. Set ENGINE_TESTDATA to point to it.");
}

#[test]
#[ignore]
fn test_5_2_load_fixture_to_pinned() {
    let _lock = TEST_MUTEX.lock().expect("mutex lock");
    let fixture = get_fixture_path();
    let file_size = std::fs::metadata(&fixture).expect("Read metadata").len();

    let initial_live = PinnedHost::live_allocations();

    let reader = GgufReader::open(&fixture).expect("Failed to open GGUF file");
    let tensor_data_offset = reader.tensor_data_offset();

    {
        let loaded = load_to_pinned(&reader, &fixture).expect("Failed to load fixture to pinned host memory");

        // Assert size matches file data area
        assert_eq!(
            loaded.total_size_bytes(),
            file_size - tensor_data_offset,
            "Loaded total_size_bytes must equal file_size - tensor_data_offset"
        );
        assert_eq!(
            loaded.as_slice().len(),
            loaded.total_size_bytes() as usize,
            "Pinned buffer slice length must match total_size_bytes"
        );

        // Assert live pinned allocations incremented
        assert_eq!(
            PinnedHost::live_allocations(),
            initial_live + 1,
            "PinnedHost live allocations should be incremented"
        );

        // Assert known tensor slices
        let embd_info = reader.get_tensor("token_embd.weight").expect("token_embd.weight info");
        let embd_slice = loaded.tensor("token_embd.weight").expect("token_embd.weight slice");
        assert_eq!(embd_slice.len(), embd_info.size_bytes as usize);

        let norm_info = reader.get_tensor("output_norm.weight").expect("output_norm.weight info");
        let norm_slice = loaded.tensor("output_norm.weight").expect("output_norm.weight slice");
        assert_eq!(norm_slice.len(), norm_info.size_bytes as usize);

        // Non-existent tensor returns None
        assert_eq!(loaded.tensor("non_existent_tensor"), None);

        // Assert layer 0 slice
        let layer0_tensors = reader.layer_index().by_layer(0).expect("Layer 0 tensors");
        let expected_layer0_len: usize = layer0_tensors.iter().map(|t| t.size_bytes as usize).sum();
        let layer0_slice = loaded.layer(0).expect("Layer 0 slice");
        assert_eq!(
            layer0_slice.len(),
            expected_layer0_len,
            "Layer 0 slice length must equal sum of layer 0 tensor sizes"
        );

        // Assert all layers have valid contiguous slices
        for &layer_idx in &reader.layer_index().layers() {
            let layer_tensors = reader.layer_index().by_layer(layer_idx).unwrap();
            let expected_len: usize = layer_tensors.iter().map(|t| t.size_bytes as usize).sum();
            let slice = loaded.layer(layer_idx).expect("Layer slice must exist");
            assert_eq!(slice.len(), expected_len);
        }

        // Non-existent layer returns None
        assert_eq!(loaded.layer(999999), None);

        // Assert throughput metric
        assert!(
            loaded.gb_per_second() > 0.0,
            "Throughput metric must be > 0 GB/s, got {}",
            loaded.gb_per_second()
        );
        println!(
            "Loaded {} bytes in pinned memory at {:.2} GB/s",
            loaded.total_size_bytes(),
            loaded.gb_per_second()
        );
    }

    // Assert live pinned allocations decremented on drop
    assert_eq!(
        PinnedHost::live_allocations(),
        initial_live,
        "PinnedHost live allocations should return to initial after drop"
    );
}
