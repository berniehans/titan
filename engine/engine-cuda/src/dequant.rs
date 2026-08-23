//! Q4_K GPU dequantization kernel and RAII launcher (f3-gpu-dequant, task 2.2).
//!
//! The CUDA kernel source lives at [`../kernels/dequant_q4k.cu`](kernels)
//! (embedded via `include_str!`), is compiled to PTX at runtime with NVRTC
//! (`cudarc::nvrtc`), loads the module through the raw CUDA driver API
//! (`cuModuleLoadData`/`cuModuleGetFunction`), and launches with `cuLaunchKernel`
//! on the crate's own [`CudaStream`]. One thread per 32-weight sub-block.
//!
//! NVRTC loads an `nvrtc64_*.dll` dynamically (cudarc's `driver` feature pulls
//! in `nvrtc`); the DLL must be discoverable on the machine (e.g. via `PATH`).
//! Output is not bit-exact against any FP16 materialization: it matches the CPU
//! reference `dequant_q4k_cpu` in `engine-core` for parity (`< 1e-5` per element).

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
/// Kernel symbol name exported by `dequant_q4k.cu`.
const FUNC_NAME: &str = "dequant_q4k_kernel";
/// Canonical CUDA kernel source, compiled to PTX at runtime via NVRTC.
const KERNEL_SRC: &str = include_str!("../kernels/dequant_q4k.cu");

/// RAII wrapper around a compiled and loaded Q4_K dequantization kernel.
///
/// Owns the CUDA `CUmodule` (freed on [`Drop`]) and the resolved `CUfunction`.
pub struct Q4KDequantizer {
    device: Arc<CudaDevice>,
    cu_module: CUmodule,
    cu_function: CUfunction,
}

// SAFETY: `CUmodule`/`CUfunction` are opaque CUDA driver handles tied to the
// `Arc<CudaDevice>`; they can be safely transferred and shared across threads.
unsafe impl Send for Q4KDequantizer {}
unsafe impl Sync for Q4KDequantizer {}

impl Q4KDequantizer {
    /// Compiles `dequant_q4k.cu` with NVRTC and loads it into `device`.
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
        // SAFETY:
        // `device.bind_to_thread()` ensured a valid CUDA context on this thread.
        // `&mut cu_module` is a valid stack pointer to receive the module handle.
        // `ptx_c.as_ptr()` points to valid UTF-8 PTX bytes (terminated by NUL).
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
        // SAFETY:
        // `cu_module` was produced by `cuModuleLoadData` above and is non-null.
        // `&mut cu_function` is a valid stack pointer.
        // `func_name_c.as_ptr()` points to a NUL-terminated symbol name.
        let res = unsafe {
            let lib = sys::lib();
            lib.cuModuleGetFunction(&mut cu_function, cu_module, func_name_c.as_ptr())
        };
        if res != CUresult::CUDA_SUCCESS || cu_function.is_null() {
            // SAFETY: `cu_module` was produced by `cuModuleLoadData` above and is valid.
            // Free the already-loaded module to avoid leaking it on this error path.
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
    /// (`n_blocks * 144` bytes) into `dst` (`n_blocks * 256` `f32`).
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
        if src_bytes == 0 || !src_bytes.is_multiple_of(144) {
            return Err(CudaError::InvalidSize {
                expected: 144,
                actual: src_bytes,
            });
        }
        let n_blocks = src_bytes / 144;
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
        // kernelParams expects each element to be a POINTER TO the argument
        // storage (the driver reads the value through that pointer). The device
        // addresses are u64 locals; the int is an i32 local. All three locals
        // stay alive for the duration of the launch call.
        let src_addr: u64 = src.device_ptr();
        let dst_addr: u64 = dst.device_ptr();
        let args: [*mut std::ffi::c_void; 3] = [
            &src_addr as *const u64 as *mut std::ffi::c_void,
            &dst_addr as *const u64 as *mut std::ffi::c_void,
            &n_blocks_i as *const i32 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured a valid CUDA context on this thread.
        // `self.cu_function` is a valid `CUfunction` from a live module.
        // `stream.raw()` is a valid `CUstream`. `args` points to 3 raw pointers that
        // map to the kernel parameters `(src, dst, n_blocks)`; `src`/`dst` are device
        // pointers into live `DeviceBuffer` allocations and `n_blocks_i` is a local
        // `i32` alive for the duration of this call (the driver copies the value at
        // launch time). `extra = null` selects the `kernelParams` array above.
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
            return Err(CudaError::KernelLaunch("cuLaunchKernel", res));
        }
        Ok(())
    }

    /// Reference to the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

impl Drop for Q4KDequantizer {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = self.device.bind_to_thread();
            // SAFETY:
            // `self.cu_module` was created by `cuModuleLoadData` in `Q4KDequantizer::new`
            // and has not been unloaded yet.
            unsafe {
                let lib = sys::lib();
                let _res = lib.cuModuleUnload(self.cu_module);
            }
            self.cu_module = std::ptr::null_mut();
        }
    }
}
