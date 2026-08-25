//! Fused RMSNorm, RoPE, and SwiGLU GPU kernel and RAII launcher (Phase 6.4).
//!
//! The CUDA kernel source lives at [`../kernels/norm_rope.cu`](kernels)
//! (embedded via `include_str!`), is compiled to PTX at runtime with NVRTC
//! (`cudarc::nvrtc`), loads the module through the raw CUDA driver API
//! (`cuModuleLoadData`/`cuModuleGetFunction`), and launches with `cuLaunchKernel`
//! on the crate's [`CudaStream`].
//!
//! Bitmask modes for op isolation / fused execution:
//! - [`MODE_NORM`]: Bit 0 (RMSNorm + residual add)
//! - [`MODE_ROPE`]: Bit 1 (partial NeoX RoPE rotation)
//! - [`MODE_SWIGLU`]: Bit 2 (SwiGLU gating `silu(y) * up`)
//! - [`MODE_FUSED`]: All 3 ops fused (`1 | 2 | 4 = 7`)

use crate::DeviceBuffer;
use crate::error::CudaError;
use crate::streams::CudaStream;
use cudarc::driver::CudaDevice;
use cudarc::driver::sys::{self, CUfunction, CUmodule, CUresult};
use cudarc::nvrtc::{self, Ptx};
use std::ffi::CString;
use std::sync::Arc;

/// Mode bit 0: RMSNorm + residual addition.
pub const MODE_NORM: u8 = 1;
/// Mode bit 1: Partial NeoX RoPE rotation.
pub const MODE_ROPE: u8 = 2;
/// Mode bit 2: SwiGLU gating (`silu(y) * up`).
pub const MODE_SWIGLU: u8 = 4;
/// Full fused mode (RMSNorm + residual + RoPE + SwiGLU).
pub const MODE_FUSED: u8 = 7;

/// NVRTC target architecture for the RTX 3060 Laptop (compute capability 8.6).
const KERNEL_ARCH: &str = "compute_86";
/// Kernel symbol name exported by `norm_rope.cu`.
const FUNC_NAME: &str = "norm_rope_swiglu_kernel";
/// Canonical CUDA kernel source, compiled to PTX at runtime via NVRTC.
const KERNEL_SRC: &str = include_str!("../kernels/norm_rope.cu");
const BLOCK_X: u32 = 256;

/// RAII wrapper around compiled and loaded fused norm/rope/swiglu kernel.
///
/// Owns the CUDA `CUmodule` (freed on [`Drop`]) and resolved `CUfunction`.
pub struct NormRope {
    device: Arc<CudaDevice>,
    cu_module: CUmodule,
    func: CUfunction,
}

// SAFETY: `CUmodule`/`CUfunction` are opaque CUDA driver handles tied to the
// `Arc<CudaDevice>`; they can be safely transferred and shared across threads.
unsafe impl Send for NormRope {}
unsafe impl Sync for NormRope {}

impl NormRope {
    /// Compiles `norm_rope.cu` with NVRTC and loads the kernel into `device`.
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

        let name_c = CString::new(FUNC_NAME)
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
            // SAFETY: Free the already-loaded module to avoid leaking it on this error path.
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuModuleUnload(cu_module);
            }
            return Err(CudaError::KernelLoad("cuModuleGetFunction", res));
        }

        Ok(Self {
            device,
            cu_module,
            func,
        })
    }

    /// Reference to the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Launches the fused norm/rope/swiglu kernel asynchronously on `stream`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer,
        residual: &DeviceBuffer,
        w: &DeviceBuffer,
        up: &DeviceBuffer,
        out: &DeviceBuffer,
        eps: f32,
        n: usize,
        n_dims: usize,
        freq_base: f32,
        pos: u32,
        mode: u8,
    ) -> Result<(), CudaError> {
        if n == 0 {
            return Err(CudaError::InvalidSize {
                expected: 1,
                actual: 0,
            });
        }
        if n_dims > n {
            return Err(CudaError::InvalidSize {
                expected: n,
                actual: n_dims,
            });
        }
        if mode == 0 {
            return Err(CudaError::InvalidSize {
                expected: 1,
                actual: 0,
            });
        }

        let expected_bytes = n * 4;
        if x.size() < expected_bytes {
            return Err(CudaError::InvalidSize {
                expected: expected_bytes,
                actual: x.size(),
            });
        }
        if residual.size() < expected_bytes {
            return Err(CudaError::InvalidSize {
                expected: expected_bytes,
                actual: residual.size(),
            });
        }
        if w.size() < expected_bytes {
            return Err(CudaError::InvalidSize {
                expected: expected_bytes,
                actual: w.size(),
            });
        }
        if up.size() < expected_bytes {
            return Err(CudaError::InvalidSize {
                expected: expected_bytes,
                actual: up.size(),
            });
        }
        if out.size() < expected_bytes {
            return Err(CudaError::InvalidSize {
                expected: expected_bytes,
                actual: out.size(),
            });
        }

        let n_i: i32 = i32::try_from(n).map_err(|_| CudaError::InvalidSize {
            expected: i32::MAX as usize,
            actual: n,
        })?;
        let n_dims_i: i32 = i32::try_from(n_dims).map_err(|_| CudaError::InvalidSize {
            expected: i32::MAX as usize,
            actual: n_dims,
        })?;
        let freq_base_f: f32 = freq_base;
        let pos_u: u32 = pos;
        let eps_f: f32 = eps;
        let mode_i: i32 = mode as i32;

        let grid_x = 1u32;

        let x_addr: u64 = x.device_ptr();
        let resid_addr: u64 = residual.device_ptr();
        let w_addr: u64 = w.device_ptr();
        let up_addr: u64 = up.device_ptr();
        let out_addr: u64 = out.device_ptr();

        let args: [*mut std::ffi::c_void; 11] = [
            &x_addr as *const u64 as *mut std::ffi::c_void,
            &resid_addr as *const u64 as *mut std::ffi::c_void,
            &w_addr as *const u64 as *mut std::ffi::c_void,
            &n_i as *const i32 as *mut std::ffi::c_void,
            &n_dims_i as *const i32 as *mut std::ffi::c_void,
            &freq_base_f as *const f32 as *mut std::ffi::c_void,
            &pos_u as *const u32 as *mut std::ffi::c_void,
            &up_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &eps_f as *const f32 as *mut std::ffi::c_void,
            &mode_i as *const i32 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured a valid CUDA context on this thread.
        // `self.func` is a valid `CUfunction` from a live module.
        // `stream.raw()` is a valid `CUstream`. `args` points to 11 raw pointers that
        // map to the kernel parameters; device buffer addresses are live and local
        // scalars are alive for the duration of this call.
        let res = unsafe {
            let lib = sys::lib();
            lib.cuLaunchKernel(
                self.func,
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

impl Drop for NormRope {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = self.device.bind_to_thread();
            // SAFETY:
            // `self.cu_module` was created by `cuModuleLoadData` in `NormRope::new`
            // and has not been unloaded yet.
            unsafe {
                let lib = sys::lib();
                let _res = lib.cuModuleUnload(self.cu_module);
            }
            self.cu_module = std::ptr::null_mut();
        }
    }
}
