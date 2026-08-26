//! Q6_K GPU dequantization kernel and RAII launcher (Phase 8, task 1.2).
//!
//! Compiles [`../kernels/dequant_q6k.cu`](kernels) to PTX at runtime with NVRTC
//! (`cudarc::nvrtc`), loads the module through the raw CUDA driver API
//! (`cuModuleLoadData`/`cuModuleGetFunction`), and launches with `cuLaunchKernel`
//! on the crate's [`CudaStream`]. 8 threads per 256-weight super-block.

use crate::DeviceBuffer;
use crate::error::CudaError;
use crate::streams::CudaStream;
use cudarc::driver::CudaDevice;
use cudarc::driver::sys::{self, CUfunction, CUmodule, CUresult};
use cudarc::nvrtc::{self, Ptx};
use std::ffi::CString;
use std::sync::Arc;

/// NVRTC target architecture for the RTX 3060 Laptop (compute capability 8.6).
const KERNEL_ARCH: &str = "compute_86";
/// Kernel symbol name exported by `dequant_q6k.cu`.
const FUNC_NAME: &str = "dequant_q6k_kernel";
/// Canonical CUDA kernel source, compiled to PTX at runtime via NVRTC.
const KERNEL_SRC: &str = include_str!("../kernels/dequant_q6k.cu");

/// RAII wrapper around a compiled and loaded Q6_K dequantization kernel.
///
/// Owns the CUDA `CUmodule` (freed on [`Drop`]) and the resolved `CUfunction`.
pub struct Q6KDequantizer {
    device: Arc<CudaDevice>,
    cu_module: CUmodule,
    cu_function: CUfunction,
}

// SAFETY: `CUmodule`/`CUfunction` are opaque CUDA driver handles tied to the
// `Arc<CudaDevice>`; they can be safely transferred and shared across threads.
unsafe impl Send for Q6KDequantizer {}
unsafe impl Sync for Q6KDequantizer {}

impl Q6KDequantizer {
    /// Compiles `dequant_q6k.cu` with NVRTC and loads it into `device`.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CudaError> {
        device.bind_to_thread()?;

        let ptx: Ptx = nvrtc::compile_ptx_with_opts(
            KERNEL_SRC,
            nvrtc::CompileOptions {
                arch: Some(KERNEL_ARCH),
                ..Default::default()
            },
        )
        .map_err(|e| CudaError::KernelCompile(format!("{e}")))?;

        let ptx_c = CString::new(ptx.to_src())
            .map_err(|_| CudaError::KernelCompile("NUL byte in PTX".into()))?;

        let mut cu_module: CUmodule = std::ptr::null_mut();
        let res = unsafe {
            let lib = sys::lib();
            lib.cuModuleLoadData(&mut cu_module, ptx_c.as_ptr() as *const std::ffi::c_void)
        };
        if res != CUresult::CUDA_SUCCESS || cu_module.is_null() {
            return Err(CudaError::KernelLoad("cuModuleLoadData", res));
        }

        let func_name_c = CString::new(FUNC_NAME)
            .map_err(|_| CudaError::KernelCompile("NUL byte in function name".into()))?;
        let mut cu_function: CUfunction = std::ptr::null_mut();
        let res = unsafe {
            let lib = sys::lib();
            lib.cuModuleGetFunction(&mut cu_function, cu_module, func_name_c.as_ptr())
        };
        if res != CUresult::CUDA_SUCCESS || cu_function.is_null() {
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuModuleUnload(cu_module);
            }
            return Err(CudaError::KernelLoad("cuModuleGetFunction", res));
        }

        Ok(Self {
            device,
            cu_module,
            cu_function,
        })
    }

    /// Launches the kernel asynchronously on `stream`, dequantizing `src`
    /// (`n_blocks * 210` bytes) into `dst` (`n_blocks * 256` `f32`).
    ///
    /// Synchronize `stream` (or copy D2H on it) before reading `dst`.
    pub fn launch(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
    ) -> Result<(), CudaError> {
        const BLOCK_X: u32 = 1024;

        let src_bytes = src.size();
        if src_bytes == 0 || !src_bytes.is_multiple_of(210) {
            return Err(CudaError::InvalidSize {
                expected: 210,
                actual: src_bytes,
            });
        }
        let n_blocks = src_bytes / 210;
        let expected_dst = n_blocks * 256 * 4;
        if dst.size() < expected_dst {
            return Err(CudaError::InvalidSize {
                expected: expected_dst,
                actual: dst.size(),
            });
        }

        let n_blocks_i: i32 = n_blocks as i32;
        let total_threads = (n_blocks * 8) as u32;
        let grid_x = total_threads.div_ceil(BLOCK_X);

        let src_addr: u64 = src.device_ptr();
        let dst_addr: u64 = dst.device_ptr();
        let args: [*mut std::ffi::c_void; 3] = [
            &src_addr as *const u64 as *mut std::ffi::c_void,
            &dst_addr as *const u64 as *mut std::ffi::c_void,
            &n_blocks_i as *const i32 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        let res = unsafe {
            let lib = sys::lib();
            lib.cuLaunchKernel(
                self.cu_function,
                grid_x,
                1,
                1,
                BLOCK_X,
                1,
                1,
                0,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            )
        };
        if res != CUresult::CUDA_SUCCESS {
            return Err(CudaError::KernelLaunch("dequant_q6k_kernel", res));
        }

        Ok(())
    }

    /// Accessor for the underlying `CudaDevice`.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

impl Drop for Q6KDequantizer {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = self.device.bind_to_thread();
            // SAFETY: `self.cu_module` is non-null and was produced by
            // `cuModuleLoadData` on `self.device`.
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuModuleUnload(self.cu_module);
            }
            self.cu_module = std::ptr::null_mut();
        }
    }
}
