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
/// Mode bit 3: Broadcast residual across batches/heads (repeats every token).
pub const MODE_BROADCAST_RESIDUAL: u8 = 8;

/// NVRTC target architecture for the RTX 3060 Laptop (compute capability 8.6).
const KERNEL_ARCH: &str = "compute_86";
/// Kernel symbol name exported by `norm_rope.cu`.
const FUNC_NAME: &str = "norm_rope_swiglu_kernel";
/// Canonical CUDA kernel source, compiled to PTX at runtime via NVRTC.
const KERNEL_SRC: &str = include_str!("../kernels/norm_rope.cu");
const BLOCK_X: u32 = 256;

const FUNC_NAME_FUSED_QK: &str = "fused_qk_norm_rope_kernel";

/// RAII wrapper around compiled and loaded fused norm/rope/swiglu kernel.
///
/// Owns the CUDA `CUmodule` (freed on [`Drop`]) and resolved `CUfunction`.
pub struct NormRope {
    device: Arc<CudaDevice>,
    cu_module: CUmodule,
    func: CUfunction,
    func_fused_qk: CUfunction,
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
                use_fast_math: Some(true),
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

        let name_c = CString::new(FUNC_NAME)
            .map_err(|_| CudaError::KernelCompile("NUL byte in function name".into()))?;
        let name_fused_qk_c = CString::new(FUNC_NAME_FUSED_QK)
            .map_err(|_| CudaError::KernelCompile("NUL byte in function name".into()))?;
        let mut func: CUfunction = std::ptr::null_mut();
        let mut func_fused_qk: CUfunction = std::ptr::null_mut();

        let (r1, r2) = unsafe {
            let lib = sys::lib();
            let r1 = lib.cuModuleGetFunction(&mut func, cu_module, name_c.as_ptr());
            let r2 = lib.cuModuleGetFunction(&mut func_fused_qk, cu_module, name_fused_qk_c.as_ptr());
            (r1, r2)
        };
        if r1 != CUresult::CUDA_SUCCESS || r2 != CUresult::CUDA_SUCCESS || func.is_null() || func_fused_qk.is_null() {
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuModuleUnload(cu_module);
            }
            return Err(CudaError::KernelLoad("cuModuleGetFunction", r1));
        }

        Ok(Self {
            device,
            cu_module,
            func,
            func_fused_qk,
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
        self.launch_with_pos_ptr(
            stream, x, residual, w, up, out, eps, n, n_dims, freq_base, pos, mode, None,
        )
    }

    /// Launches the fused norm/rope/swiglu kernel with an optional dynamic device-side position pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_with_pos_ptr(
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
        pos_ptr: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        let expected_bytes = n * 4;
        let n_heads = (out.size() / expected_bytes).max(1);
        self.launch_batched_with_pos_ptr(
            stream, x, residual, w, up, out, eps, n, n_dims, freq_base, pos, mode, pos_ptr, n_heads, n_heads,
        )
    }

    /// Launches the fused norm/rope/swiglu kernel for batched multi-head sequences.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_with_pos_ptr(
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
        pos_ptr: Option<&DeviceBuffer>,
        batch_count: usize,
        n_heads_per_token: usize,
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
        let n_heads_i: i32 = n_heads_per_token as i32;
        let grid_x = (batch_count as u32).max(1);

        let x_addr: u64 = x.device_ptr();
        let resid_addr: u64 = residual.device_ptr();
        let w_addr: u64 = w.device_ptr();
        let up_addr: u64 = up.device_ptr();
        let out_addr: u64 = out.device_ptr();
        let pos_ptr_addr: u64 = pos_ptr.map(|p| p.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 13] = [
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
            &pos_ptr_addr as *const u64 as *mut std::ffi::c_void,
            &n_heads_i as *const i32 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured a valid CUDA context on this thread.
        // `self.func` is a valid `CUfunction` from a live module.
        // `stream.raw()` is a valid `CUstream`. `args` points to 13 raw pointers that
        // map to the kernel parameters; device buffer addresses are live and local
        // scalars are alive for the duration of this call.
        let res = unsafe {
            let lib = sys::lib();
            lib.cuLaunchKernel(
                self.func,
                grid_x,
                1,
                1,
                32,
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

    /// Fused Q + K RMSNorm + RoPE + KV Cache Append kernel: Computes RMSNorm + RoPE for both Q and K and appends K and V into the paged KV pool simultaneously in 1 kernel launch.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_qk(
        &self,
        stream: &CudaStream,
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        qn_w: &DeviceBuffer,
        kn_w: &DeviceBuffer,
        n_head_q: usize,
        n_head_k: usize,
        head_dim: usize,
        n_rot: usize,
        freq_base: f32,
        eps: f32,
        mode: u8,
        pos_ptr: Option<&DeviceBuffer>,
        v: Option<&DeviceBuffer>,
        pool: Option<&DeviceBuffer>,
        block_table: Option<&DeviceBuffer>,
        block_tokens: usize,
    ) -> Result<(), CudaError> {
        let n_head_q_i: i32 = n_head_q as i32;
        let n_head_k_i: i32 = n_head_k as i32;
        let head_dim_i: i32 = head_dim as i32;
        let n_rot_i: i32 = n_rot as i32;
        let mode_i: i32 = mode as i32;
        let block_tokens_i: i32 = block_tokens as i32;

        let q_addr: u64 = q.device_ptr();
        let k_addr: u64 = k.device_ptr();
        let qn_addr: u64 = qn_w.device_ptr();
        let kn_addr: u64 = kn_w.device_ptr();
        let pos_ptr_addr: u64 = pos_ptr.map(|p| p.device_ptr()).unwrap_or(0);
        let v_addr: u64 = v.map(|p| p.device_ptr()).unwrap_or(0);
        let pool_addr: u64 = pool.map(|p| p.device_ptr()).unwrap_or(0);
        let bt_addr: u64 = block_table.map(|p| p.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 16] = [
            &q_addr as *const u64 as *mut std::ffi::c_void,
            &k_addr as *const u64 as *mut std::ffi::c_void,
            &qn_addr as *const u64 as *mut std::ffi::c_void,
            &kn_addr as *const u64 as *mut std::ffi::c_void,
            &n_head_q_i as *const i32 as *mut std::ffi::c_void,
            &n_head_k_i as *const i32 as *mut std::ffi::c_void,
            &head_dim_i as *const i32 as *mut std::ffi::c_void,
            &n_rot_i as *const i32 as *mut std::ffi::c_void,
            &freq_base as *const f32 as *mut std::ffi::c_void,
            &eps as *const f32 as *mut std::ffi::c_void,
            &mode_i as *const i32 as *mut std::ffi::c_void,
            &pos_ptr_addr as *const u64 as *mut std::ffi::c_void,
            &v_addr as *const u64 as *mut std::ffi::c_void,
            &pool_addr as *const u64 as *mut std::ffi::c_void,
            &bt_addr as *const u64 as *mut std::ffi::c_void,
            &block_tokens_i as *const i32 as *mut std::ffi::c_void,
        ];

        let grid_x = (n_head_q + n_head_k) as u32;
        let block_x: u32 = 32;

        self.device.bind_to_thread()?;

        let res = unsafe {
            let lib = sys::lib();
            lib.cuLaunchKernel(
                self.func_fused_qk,
                grid_x,
                1,
                1,
                block_x,
                1,
                1,
                0,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            )
        };
        if res != CUresult::CUDA_SUCCESS {
            return Err(CudaError::KernelLaunch("cuLaunchKernel (FusedQKNormRoPE)", res));
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
