//! Multi-format GEMV GPU kernel and RAII launcher (Phase 6.3).
//!
//! The CUDA kernel source lives at [`../kernels/gemv_q4k.cu`](kernels)
//! (embedded via `include_str!`), is compiled to PTX at runtime with NVRTC
//! (`cudarc::nvrtc`), loads the module through the raw CUDA driver API
//! (`cuModuleLoadData`/`cuModuleGetFunction`), and launches with `cuLaunchKernel`
//! on the crate's [`CudaStream`].
//!
//! Supports Q4_K, Q8_0, and F16 quantized/unquantized weight formats with
//! per-column reduction in registers.

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
/// Kernel symbol names exported by `gemv_q4k.cu`.
const FUNC_NAME_Q4K: &str = "gemv_q4k_kernel";
const FUNC_NAME_Q6K: &str = "gemv_q6k_kernel";
const FUNC_NAME_Q8: &str = "gemv_q8_kernel";
const FUNC_NAME_F16: &str = "gemv_f16_kernel";
/// Canonical CUDA kernel source, compiled to PTX at runtime via NVRTC.
const KERNEL_SRC: &str = include_str!("../kernels/gemv_q4k.cu");
const BLOCK_X: u32 = 256;

/// Quantization format of the weight matrix for GEMV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemvFormat {
    /// Q4_K quantized weights (256 weights per 144-byte super-block).
    Q4K,
    /// Q6_K quantized weights (256 weights per 210-byte super-block).
    Q6K,
    /// Q8_0 quantized weights (32 weights per 34-byte block).
    Q8,
    /// 16-bit float weights (2 bytes per weight).
    F16,
}

/// RAII wrapper around compiled and loaded multi-format GEMV kernels.
///
/// Owns the CUDA `CUmodule` (freed on [`Drop`]) and resolved `CUfunction`s.
pub struct MultiFormatGEMV {
    device: Arc<CudaDevice>,
    cu_module: CUmodule,
    fn_q4k: CUfunction,
    fn_q6k: CUfunction,
    fn_q8: CUfunction,
    fn_f16: CUfunction,
}

// SAFETY: `CUmodule`/`CUfunction` are opaque CUDA driver handles tied to the
// `Arc<CudaDevice>`; they can be safely transferred and shared across threads.
unsafe impl Send for MultiFormatGEMV {}
unsafe impl Sync for MultiFormatGEMV {}

impl MultiFormatGEMV {
    /// Compiles `gemv_q4k.cu` with NVRTC and loads all three kernel symbols into `device`.
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

        let load_fn = |name: &str| -> Result<CUfunction, CudaError> {
            let name_c = CString::new(name)
                .map_err(|_| CudaError::KernelCompile("NUL byte in function name".into()))?;
            let mut func: CUfunction = std::ptr::null_mut();
            // SAFETY:
            // `cu_module` was produced by `cuModuleLoadData` above and is non-null.
            // `&mut func` is a valid stack pointer.
            // `name_c.as_ptr()` points to a NUL-terminated symbol name.
            let res = unsafe {
                let lib = sys::lib();
                lib.cuModuleGetFunction(&mut func, cu_module, name_c.as_ptr())
            };
            if res != CUresult::CUDA_SUCCESS || func.is_null() {
                Err(CudaError::KernelLoad("cuModuleGetFunction", res))
            } else {
                Ok(func)
            }
        };

        let fn_q4k = match load_fn(FUNC_NAME_Q4K) {
            Ok(f) => f,
            Err(e) => {
                unsafe {
                    let lib = sys::lib();
                    let _ = lib.cuModuleUnload(cu_module);
                }
                return Err(e);
            }
        };

        let fn_q6k = match load_fn(FUNC_NAME_Q6K) {
            Ok(f) => f,
            Err(e) => {
                unsafe {
                    let lib = sys::lib();
                    let _ = lib.cuModuleUnload(cu_module);
                }
                return Err(e);
            }
        };

        let fn_q8 = match load_fn(FUNC_NAME_Q8) {
            Ok(f) => f,
            Err(e) => {
                unsafe {
                    let lib = sys::lib();
                    let _ = lib.cuModuleUnload(cu_module);
                }
                return Err(e);
            }
        };

        let fn_f16 = match load_fn(FUNC_NAME_F16) {
            Ok(f) => f,
            Err(e) => {
                unsafe {
                    let lib = sys::lib();
                    let _ = lib.cuModuleUnload(cu_module);
                }
                return Err(e);
            }
        };

        Ok(Self {
            device,
            cu_module,
            fn_q4k,
            fn_q6k,
            fn_q8,
            fn_f16,
        })
    }

    /// Reference to the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Launches matrix-vector multiplication asynchronously on `stream`.
    ///
    /// Computes `out[j] = dot(weights[:, j], x)` for `j in 0..ne1`,
    /// where `ne0` is the reduction dimension and `ne1` is the number of columns.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv(
        &self,
        stream: &CudaStream,
        format: GemvFormat,
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        out: &DeviceBuffer,
        ne0: usize,
        ne1: usize,
    ) -> Result<(), CudaError> {
        if ne0 == 0 {
            return Err(CudaError::InvalidSize {
                expected: 1,
                actual: 0,
            });
        }
        if ne1 == 0 {
            return Err(CudaError::InvalidSize {
                expected: 1,
                actual: 0,
            });
        }

        let expected_weight_bytes = match format {
            GemvFormat::Q4K => {
                if !ne0.is_multiple_of(256) {
                    return Err(CudaError::InvalidSize {
                        expected: 256,
                        actual: ne0,
                    });
                }
                (ne0 / 256) * 144 * ne1
            }
            GemvFormat::Q6K => {
                if !ne0.is_multiple_of(256) {
                    return Err(CudaError::InvalidSize {
                        expected: 256,
                        actual: ne0,
                    });
                }
                (ne0 / 256) * 210 * ne1
            }
            GemvFormat::Q8 => {
                if !ne0.is_multiple_of(32) {
                    return Err(CudaError::InvalidSize {
                        expected: 32,
                        actual: ne0,
                    });
                }
                (ne0 / 32) * 34 * ne1
            }
            GemvFormat::F16 => ne0 * 2 * ne1,
        };

        if weights.size() < expected_weight_bytes {
            return Err(CudaError::InvalidSize {
                expected: expected_weight_bytes,
                actual: weights.size(),
            });
        }

        let expected_x_bytes = ne0 * 4;
        if x.size() < expected_x_bytes {
            return Err(CudaError::InvalidSize {
                expected: expected_x_bytes,
                actual: x.size(),
            });
        }

        let expected_out_bytes = ne1 * 4;
        if out.size() < expected_out_bytes {
            return Err(CudaError::InvalidSize {
                expected: expected_out_bytes,
                actual: out.size(),
            });
        }

        let ne0_i: i32 = i32::try_from(ne0).map_err(|_| CudaError::InvalidSize {
            expected: i32::MAX as usize,
            actual: ne0,
        })?;
        let ne1_i: i32 = i32::try_from(ne1).map_err(|_| CudaError::InvalidSize {
            expected: i32::MAX as usize,
            actual: ne1,
        })?;

        let grid_x = (ne1 as u32).div_ceil(BLOCK_X);

        let weights_addr: u64 = weights.device_ptr();
        let x_addr: u64 = x.device_ptr();
        let out_addr: u64 = out.device_ptr();
        let args: [*mut std::ffi::c_void; 5] = [
            &weights_addr as *const u64 as *mut std::ffi::c_void,
            &x_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &ne1_i as *const i32 as *mut std::ffi::c_void,
        ];

        let cu_function = match format {
            GemvFormat::Q4K => self.fn_q4k,
            GemvFormat::Q6K => self.fn_q6k,
            GemvFormat::Q8 => self.fn_q8,
            GemvFormat::F16 => self.fn_f16,
        };

        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured a valid CUDA context on this thread.
        // `cu_function` is a valid `CUfunction` from a live module.
        // `stream.raw()` is a valid `CUstream`. `args` points to 5 raw pointers that
        // map to the kernel parameters `(weights, x, out, ne0, ne1)`; `weights`/`x`/`out`
        // are device pointers into live `DeviceBuffer` allocations and `ne0_i`/`ne1_i`
        // are local `i32` values alive for the duration of this call (the driver copies
        // the values at launch time). `extra = null` selects the `kernelParams` array.
        let res = unsafe {
            let lib = sys::lib();
            lib.cuLaunchKernel(
                cu_function,
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
}

impl Drop for MultiFormatGEMV {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = self.device.bind_to_thread();
            // SAFETY:
            // `self.cu_module` was created by `cuModuleLoadData` in `MultiFormatGEMV::new`
            // and has not been unloaded yet.
            unsafe {
                let lib = sys::lib();
                let _res = lib.cuModuleUnload(self.cu_module);
            }
            self.cu_module = std::ptr::null_mut();
        }
    }
}
