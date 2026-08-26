//! Hardware bandwidth profile and policy resolution (Phase 7.1).
//!
//! Stores offline micro-benchmark measurements (`benchbw.json`) of:
//! - Host DRAM read bandwidth (STREAM-style)
//! - Linear PCIe H2D and D2H copy bandwidth
//! - Standalone and overlapped CPU MoE GEMV vs PCIe gather bandwidth
//!
//! Provides data-driven policy resolution:
//! - `fetch_fraction`: bandwidth-matched ratio `pcie_ov / (pcie_ov + cpu_ov)`
//! - `recommended_backend`: `hybrid` when `cpu_ov > 2.0 * pcie_ov`, else `offload`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// MoE execution backend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeBackend {
    /// Stream all missing experts over PCIe to GPU slot cache; compute GEMM on GPU.
    Offload,
    /// Compute all missing experts on host CPU; stream activations and results.
    Cpu,
    /// Balanced hybrid: stream `fetch_fraction * misses` to GPU; overflow misses compute on CPU concurrently.
    Hybrid,
}

impl std::fmt::Display for MoeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeBackend::Offload => write!(f, "offload"),
            MoeBackend::Cpu => write!(f, "cpu"),
            MoeBackend::Hybrid => write!(f, "hybrid"),
        }
    }
}

/// Metadata describing the GPU device used for profiling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuProfileInfo {
    /// Device product name (e.g. "NVIDIA GeForce RTX 3060 Laptop GPU").
    pub name: String,
    /// Major and minor compute capability (e.g. "8.6").
    pub compute_capability: String,
    /// Total VRAM bytes reported by the driver.
    pub total_memory_bytes: usize,
}

/// Measured bandwidth metrics for a specific quantization format on this hardware.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandwidthMeasurement {
    /// Host DRAM read bandwidth (GB/s).
    pub stream_dram_read_gbps: f64,
    /// Linear pinned Host->Device copy bandwidth (GB/s).
    pub linear_pcie_h2d_gbps: f64,
    /// Linear Device->Host copy bandwidth (GB/s).
    pub linear_pcie_d2h_gbps: f64,
    /// Standalone CPU MoE GEMV bandwidth (GB/s).
    pub cpu_moe_isolated_gbps: f64,
    /// Standalone PCIe gather bandwidth (GB/s).
    pub pcie_gather_isolated_gbps: f64,
    /// CPU MoE GEMV bandwidth under concurrent PCIe gather contention (GB/s).
    pub cpu_moe_overlap_gbps: f64,
    /// PCIe gather bandwidth under concurrent CPU MoE contention (GB/s).
    pub pcie_gather_overlap_gbps: f64,
    /// Calculated hybrid fetch fraction: `pcie_ov / (pcie_ov + cpu_ov)`, clamped to `[0.0, 1.0]`.
    pub fetch_fraction: f64,
    /// Recommended execution backend based on measured bandwidth ratio.
    pub recommended_backend: MoeBackend,
}

/// Root hardware bandwidth profile structure saved to `benchbw.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareBandwidthProfile {
    /// Schema version (current = 1).
    pub version: u32,
    /// Timestamp when profile was recorded.
    pub timestamp_utc: String,
    /// GPU metadata.
    pub gpu: GpuProfileInfo,
    /// CPU model string.
    pub cpu_brand: String,
    /// Measurements keyed by quantization format (e.g. "Q4_K", "Q8_0", "F16").
    pub measurements: HashMap<String, BandwidthMeasurement>,
}

impl HardwareBandwidthProfile {
    /// Current profile schema version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Creates an empty profile for the given hardware.
    pub fn new(gpu: GpuProfileInfo, cpu_brand: impl Into<String>) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            timestamp_utc: "2026-08-26T00:00:00Z".to_string(),
            gpu,
            cpu_brand: cpu_brand.into(),
            measurements: HashMap::new(),
        }
    }

    /// Records or updates a measurement for a specific quant format.
    pub fn record(&mut self, format: impl Into<String>, measurement: BandwidthMeasurement) {
        self.measurements.insert(format.into(), measurement);
    }

    /// Serializes and saves the profile to a JSON file.
    pub fn save_to_file(&self, path: &Path) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Loads and parses a profile from a JSON file.
    pub fn load_from_file(path: &Path) -> Result<Self, std::io::Error> {
        let content = fs::read_to_string(path)?;
        let prof: Self = serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(prof)
    }
}

/// Resolves the hybrid fetch fraction for a format from a profile or default fallback.
///
/// If `profile` contains overlapped measurements for `format`, calculates:
/// `pcie_ov / (pcie_ov + cpu_ov)`, clamped to `[0.0, 1.0]`.
/// Otherwise, falls back to `default_fraction` (clamped to `[0.0, 1.0]`).
pub fn resolve_hybrid_fetch_fraction(
    profile: Option<&HardwareBandwidthProfile>,
    format: &str,
    default_fraction: f64,
) -> f64 {
    if let Some(m) = profile.and_then(|p| p.measurements.get(format)) {
        let denom = m.pcie_gather_overlap_gbps + m.cpu_moe_overlap_gbps;
        if denom > 1e-6 {
            let frac = m.pcie_gather_overlap_gbps / denom;
            return frac.clamp(0.0, 1.0);
        }
        return m.fetch_fraction.clamp(0.0, 1.0);
    }
    default_fraction.clamp(0.0, 1.0)
}

/// Resolves the recommended backend for a format from a profile or default fallback.
///
/// Rule: `MoeBackend::Hybrid` if `cpu_moe_overlap_gbps > 2.0 * pcie_gather_overlap_gbps`,
/// else `MoeBackend::Offload`.
pub fn resolve_backend_recommendation(
    profile: Option<&HardwareBandwidthProfile>,
    format: &str,
    default_backend: MoeBackend,
) -> MoeBackend {
    if let Some(m) = profile.and_then(|p| p.measurements.get(format)) {
        return m.recommended_backend;
    }
    default_backend
}
