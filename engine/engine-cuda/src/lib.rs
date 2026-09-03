pub mod batched_gemm;
pub mod dequant;
pub mod dequant_q6k;
pub mod device_buffer;
pub mod dispatch_telemetry;
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
pub use dispatch_telemetry::{DispatchRecord, DispatchTelemetry, DispatchTelemetrySnapshot};
pub use error::CudaError;
pub use event::CudaEvent;
pub use flash_attention::FlashAttention2;
pub use graphs::{CudaGraph, CudaGraphExec};
pub use logit_mask::LogitMaskGpu;
pub use multiformat_gemv::{GemvFormat, MultiFormatGEMV};
pub use norm_rope::{
    MODE_BROADCAST_RESIDUAL, MODE_FUSED, MODE_NORM, MODE_ROPE, MODE_SWIGLU, NormRope,
};
pub use paged_attention::PagedAttention;
pub use paged_kv::{KvDataType, PagedKvGpu, PagedKvLayout};
pub use pinned_host::PinnedHost;
pub use streams::CudaStream;

/// Auto-discovers and prepends CUDA / NVRTC dynamic library search paths on Windows.
#[cfg(target_os = "windows")]
pub fn ensure_cuda_dll_paths() {
    let extra_dirs = cuda_dll_search_paths_from_env();
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
    if let Some(nvrtc_dll) = find_nvrtc_dll_in_dirs(&extra_dirs) {
        let _ = load_nvrtc_dll(&nvrtc_dll);
    }
}

/// Initializes the CUDA/NVRTC runtime and returns the discovered NVRTC DLL.
///
/// This is intended for process startup and GPU tests. It performs an explicit
/// preflight so a missing runtime produces an actionable error instead of the
/// panic emitted by cudarc's lazy NVRTC loader.
#[cfg(target_os = "windows")]
pub fn initialize_cuda_runtime() -> Result<std::path::PathBuf, String> {
    let dirs = cuda_dll_search_paths_from_env();
    let nvrtc_dll = find_nvrtc_dll_in_dirs(&dirs).ok_or_else(|| {
        "NVRTC preflight failed: no nvrtc64_*.dll was found in the CUDA DLL search paths. "
            .to_owned()
            + "Set TITAN_CUDA_DLL_DIR to the directory containing the required NVRTC DLL "
            + "(for example nvrtc64_120_0.dll), then rerun the benchmark."
    })?;
    load_nvrtc_dll(&nvrtc_dll)?;
    Ok(nvrtc_dll)
}

#[cfg(target_os = "windows")]
#[allow(clippy::collapsible_if)] // Keep Rust 2021-compatible nested pattern matching.
fn load_nvrtc_dll(path: &std::path::Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{
        LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW,
    };

    static NVRTC_MODULE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    if NVRTC_MODULE.get().is_some() {
        return Ok(());
    }

    let mut dependency_paths = Vec::new();
    if let Some(parent) = path.parent() {
        if let Ok(entries) = std::fs::read_dir(parent) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("nvrtc-builtins") && name.ends_with(".dll") {
                    dependency_paths.push(entry.path());
                }
            }
        }
    }
    dependency_paths.push(path.to_owned());

    let mut module = std::ptr::null_mut();
    for dependency in dependency_paths {
        let wide_path: Vec<u16> = dependency
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect();
        module = unsafe {
            LoadLibraryExW(
                wide_path.as_ptr(),
                std::ptr::null_mut(),
                LOAD_WITH_ALTERED_SEARCH_PATH,
            )
        };
        if module.is_null() {
            return Err(format!(
                "NVRTC preflight found {} but Windows could not load dependency {}: {}.",
                path.display(),
                dependency.display(),
                std::io::Error::last_os_error()
            ));
        }
    }
    let _ = NVRTC_MODULE.set(module as usize);
    Ok(())
}

#[cfg(target_os = "windows")]
fn cuda_dll_search_paths_from_env() -> Vec<std::path::PathBuf> {
    let override_dir = std::env::var_os("TITAN_CUDA_DLL_DIR").map(std::path::PathBuf::from);
    cuda_dll_search_paths(override_dir)
}

#[cfg(target_os = "windows")]
fn cuda_dll_search_paths(override_dir: Option<std::path::PathBuf>) -> Vec<std::path::PathBuf> {
    let mut extra_dirs = Vec::new();
    if let Some(dir) = override_dir.filter(|dir| !dir.as_os_str().is_empty()) {
        extra_dirs.push(dir);
    }
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
        let cuda_base = std::path::PathBuf::from(prog_files)
            .join("NVIDIA GPU Computing Toolkit")
            .join("CUDA");
        if let Ok(entries) = std::fs::read_dir(cuda_base) {
            for entry in entries.flatten() {
                let bin = entry.path().join("bin");
                if bin.exists() {
                    extra_dirs.push(bin);
                }
            }
        }
    }
    extra_dirs
}

fn find_nvrtc_dll_in_dirs(dirs: &[std::path::PathBuf]) -> Option<std::path::PathBuf> {
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        let mut fallback = None;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("nvrtc64_") && name.ends_with(".dll") {
                if name == "nvrtc64_120_0.dll" {
                    return Some(entry.path());
                }
                if !name.contains(".alt.") {
                    fallback = Some(entry.path());
                }
            }
        }
        if fallback.is_some() {
            return fallback;
        }
    }
    None
}

/// Checks for an NVRTC runtime DLL without loading CUDA or creating a device.
#[cfg(target_os = "windows")]
pub fn nvrtc_dll_preflight() -> Result<std::path::PathBuf, String> {
    let dirs = cuda_dll_search_paths_from_env();
    find_nvrtc_dll_in_dirs(&dirs).ok_or_else(|| {
        "NVRTC preflight failed: no nvrtc64_*.dll was found in the CUDA DLL search paths. "
            .to_owned()
            + "Set TITAN_CUDA_DLL_DIR to the directory containing the required NVRTC DLL "
            + "(for example nvrtc64_120_0.dll), then rerun the benchmark."
    })
}

#[cfg(not(target_os = "windows"))]
pub fn nvrtc_dll_preflight() -> Result<std::path::PathBuf, String> {
    Ok(std::path::PathBuf::new())
}

#[cfg(not(target_os = "windows"))]
pub fn ensure_cuda_dll_paths() {}

pub fn version() -> &'static str {
    "0.1.0"
}

#[cfg(test)]
mod tests {
    use super::find_nvrtc_dll_in_dirs;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn finds_nvrtc_dll_candidate_in_a_search_directory() {
        let dir = unique_temp_dir("nvrtc-present");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("nvrtc64_120_0.dll"), b"test").unwrap();

        let result = find_nvrtc_dll_in_dirs(std::slice::from_ref(&dir));

        assert_eq!(result, Some(dir.join("nvrtc64_120_0.dll")));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ignores_non_nvrtc_files_when_preflighting() {
        let dir = unique_temp_dir("nvrtc-absent");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cudart64_120.dll"), b"test").unwrap();

        assert_eq!(find_nvrtc_dll_in_dirs(std::slice::from_ref(&dir)), None);
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn explicit_cuda_dll_dir_is_the_first_search_path() {
        let override_dir = PathBuf::from(r"C:\custom\cuda\bin");

        let paths = super::cuda_dll_search_paths(Some(override_dir.clone()));

        assert_eq!(paths.first(), Some(&override_dir));
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "titan-cuda-preflight-{name}-{}",
            std::process::id()
        ))
    }
}
