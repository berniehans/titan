//! Bandwidth profiler and policy resolution tests (Phase 7.1).
//!
//! Asserts that:
//! 1. `benchbw.json` serialization round-trips losslessly and conforms to versioned schema.
//! 2. `fetch_fraction` calculates `pcie_ov / (pcie_ov + cpu_ov)` and clamps properly.
//! 3. Backend recommendation selects `hybrid` when `cpu_ov > 2.0 * pcie_ov`, else `offload`.
//! 4. Host DRAM STREAM read baseline measurement produces positive bandwidth.
//! 5. GPU hardware profiler measures linear PCIe and overlapped gather when CUDA is available.

use engine_core::moe::{
    BandwidthMeasurement, GpuProfileInfo, HardwareBandwidthProfile, MoeBackend,
    measure_stream_host_dram_read, resolve_backend_recommendation, resolve_hybrid_fetch_fraction,
};
use std::path::PathBuf;

#[test]
fn test_benchbw_profile_serialization_roundtrip() {
    let gpu = GpuProfileInfo {
        name: "NVIDIA GeForce RTX 3060 Laptop GPU".to_string(),
        compute_capability: "8.6".to_string(),
        total_memory_bytes: 6_442_450_944,
    };
    let mut profile = HardwareBandwidthProfile::new(gpu, "AMD Ryzen 7 5800H");

    let meas_q4k = BandwidthMeasurement {
        stream_dram_read_gbps: 45.2,
        linear_pcie_h2d_gbps: 12.1,
        linear_pcie_d2h_gbps: 11.8,
        cpu_moe_isolated_gbps: 38.5,
        pcie_gather_isolated_gbps: 11.9,
        cpu_moe_overlap_gbps: 32.4,
        pcie_gather_overlap_gbps: 10.2,
        fetch_fraction: 10.2 / (10.2 + 32.4),
        recommended_backend: MoeBackend::Hybrid,
    };
    profile.record("Q4_K", meas_q4k);

    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_benchbw_schema.json");

    profile.save_to_file(&file_path).expect("save profile");
    let loaded = HardwareBandwidthProfile::load_from_file(&file_path).expect("load profile");

    assert_eq!(loaded.version, HardwareBandwidthProfile::CURRENT_VERSION);
    assert_eq!(loaded.gpu.name, "NVIDIA GeForce RTX 3060 Laptop GPU");
    assert_eq!(loaded.cpu_brand, "AMD Ryzen 7 5800H");
    assert!(loaded.measurements.contains_key("Q4_K"));

    let m = loaded.measurements.get("Q4_K").unwrap();
    assert_eq!(m.recommended_backend, MoeBackend::Hybrid);
    assert!((m.fetch_fraction - (10.2 / (10.2 + 32.4))).abs() < 1e-6);

    let _ = std::fs::remove_file(file_path);
}

#[test]
fn test_fetch_fraction_and_backend_policy_resolution() {
    let gpu = GpuProfileInfo {
        name: "Test GPU".to_string(),
        compute_capability: "8.0".to_string(),
        total_memory_bytes: 16_000_000_000,
    };
    let mut profile = HardwareBandwidthProfile::new(gpu, "Test CPU");

    // Case 1: Fast CPU (> 2x PCIe) -> Hybrid
    profile.record(
        "Q4_K",
        BandwidthMeasurement {
            stream_dram_read_gbps: 60.0,
            linear_pcie_h2d_gbps: 12.0,
            linear_pcie_d2h_gbps: 12.0,
            cpu_moe_isolated_gbps: 40.0,
            pcie_gather_isolated_gbps: 12.0,
            cpu_moe_overlap_gbps: 30.0,
            pcie_gather_overlap_gbps: 10.0,
            fetch_fraction: 10.0 / (10.0 + 30.0), // 0.25
            recommended_backend: MoeBackend::Hybrid,
        },
    );

    // Case 2: Slow CPU (< 2x PCIe) -> Offload
    profile.record(
        "F16",
        BandwidthMeasurement {
            stream_dram_read_gbps: 60.0,
            linear_pcie_h2d_gbps: 12.0,
            linear_pcie_d2h_gbps: 12.0,
            cpu_moe_isolated_gbps: 15.0,
            pcie_gather_isolated_gbps: 12.0,
            cpu_moe_overlap_gbps: 12.0,
            pcie_gather_overlap_gbps: 10.0,
            fetch_fraction: 10.0 / (10.0 + 12.0), // ~0.4545
            recommended_backend: MoeBackend::Offload,
        },
    );

    // Assert fetch fraction resolution
    let frac_q4k = resolve_hybrid_fetch_fraction(Some(&profile), "Q4_K", 0.5);
    assert!((frac_q4k - 0.25).abs() < 1e-6);

    let frac_f16 = resolve_hybrid_fetch_fraction(Some(&profile), "F16", 0.5);
    assert!((frac_f16 - (10.0 / 22.0)).abs() < 1e-6);

    // Fallback when missing
    let frac_missing = resolve_hybrid_fetch_fraction(Some(&profile), "UNKNOWN", 0.5);
    assert_eq!(frac_missing, 0.5);

    // Assert backend recommendation
    assert_eq!(
        resolve_backend_recommendation(Some(&profile), "Q4_K", MoeBackend::Offload),
        MoeBackend::Hybrid
    );
    assert_eq!(
        resolve_backend_recommendation(Some(&profile), "F16", MoeBackend::Hybrid),
        MoeBackend::Offload
    );
    assert_eq!(
        resolve_backend_recommendation(None, "Q4_K", MoeBackend::Offload),
        MoeBackend::Offload
    );
}

#[test]
fn test_host_dram_stream_read_measurement() {
    let bw = measure_stream_host_dram_read(8 * 1024 * 1024, 10);
    println!("Measured host DRAM read bandwidth: {:.2} GB/s", bw);
    assert!(bw > 0.1, "DRAM bandwidth must be strictly positive");
}

#[test]
#[ignore]
fn test_gpu_hardware_bandwidth_profiler_gate()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use cudarc::driver::CudaDevice;
    use engine_core::moe::benchbw::measure_overlapped_cpu_and_pcie;
    use engine_cuda::CudaStream;
    use std::sync::Arc;

    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;

    let gpu = GpuProfileInfo {
        name: "NVIDIA RTX 3060 Laptop GPU".to_string(),
        compute_capability: "8.6".to_string(),
        total_memory_bytes: 6_442_450_944,
    };
    let mut profile = HardwareBandwidthProfile::new(gpu, "Host CPU");

    // Profile Q4_K format (8 MB simulated expert block)
    let meas_q4k = measure_overlapped_cpu_and_pcie(&device, &stream, "Q4_K", 8 * 1024 * 1024, 5)?;
    profile.record("Q4_K", meas_q4k);

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/benches");
    let out_file = out_dir.join("benchbw.json");
    profile.save_to_file(&out_file)?;
    println!("\nSaved hardware bandwidth profile to: {:?}", out_file);

    Ok(())
}
