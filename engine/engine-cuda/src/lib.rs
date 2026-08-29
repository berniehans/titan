pub mod batched_gemm;
pub mod dequant;
pub mod dequant_q6k;
pub mod device_buffer;
pub mod error;
pub mod event;
pub mod flash_attention;
pub mod graphs;
pub mod logit_mask;
pub mod multiformat_gemv;
pub mod norm_rope;
pub mod paged_attention;
pub mod paged_kv;
pub mod pinned_host;
pub mod streams;

pub use batched_gemm::BatchedGEMM;
pub use cudarc::driver::CudaDevice;
pub use dequant::Q4KDequantizer;
pub use dequant_q6k::Q6KDequantizer;
pub use device_buffer::DeviceBuffer;
pub use error::CudaError;
pub use event::CudaEvent;
pub use flash_attention::FlashAttention2;
pub use graphs::{CudaGraph, CudaGraphExec};
pub use logit_mask::LogitMaskGpu;
pub use multiformat_gemv::{GemvFormat, MultiFormatGEMV};
pub use norm_rope::{MODE_BROADCAST_RESIDUAL, MODE_FUSED, MODE_NORM, MODE_ROPE, MODE_SWIGLU, NormRope};
pub use paged_attention::PagedAttention;
pub use paged_kv::{KvDataType, PagedKvGpu, PagedKvLayout};
pub use pinned_host::PinnedHost;
pub use streams::CudaStream;

/// Auto-discovers and prepends CUDA / NVRTC dynamic library search paths on Windows.
#[cfg(target_os = "windows")]
pub fn ensure_cuda_dll_paths() {
    let mut extra_dirs = Vec::new();
    let temp = std::env::temp_dir();
    if temp.exists() {
        extra_dirs.push(temp);
    }
    if let Ok(local_appdata) = std::env::var("LOCALAPPDATA") {
        let p = std::path::PathBuf::from(local_appdata).join("Temp");
        if p.exists() {
            extra_dirs.push(p);
        }
    }
    for (k, v) in std::env::vars() {
        if k.starts_with("CUDA_PATH") {
            let bin = std::path::PathBuf::from(&v).join("bin");
            if bin.exists() {
                extra_dirs.push(bin);
            }
        }
    }
    if let Ok(prog_files) = std::env::var("ProgramFiles") {
        let cuda_base = std::path::PathBuf::from(prog_files).join("NVIDIA GPU Computing Toolkit").join("CUDA");
        if let Ok(entries) = std::fs::read_dir(cuda_base) {
            for entry in entries.flatten() {
                let bin = entry.path().join("bin");
                if bin.exists() {
                    extra_dirs.push(bin);
                }
            }
        }
    }

    if let Ok(curr_path) = std::env::var("PATH") {
        let mut new_path = String::new();
        for dir in &extra_dirs {
            let s = dir.to_string_lossy();
            if !curr_path.contains(&*s) {
                new_path.push_str(&s);
                new_path.push(';');
            }
        }
        if !new_path.is_empty() {
            new_path.push_str(&curr_path);
            unsafe {
                std::env::set_var("PATH", new_path);
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub fn ensure_cuda_dll_paths() {}

pub fn version() -> &'static str {
    "0.1.0"
}
