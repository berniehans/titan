//! PagedAttention decode GPU kernel and RAII launcher (Phase 6.5).
//!
//! The CUDA kernel source lives at [`../kernels/paged_attention.cu`](kernels)
//! (embedded via `include_str!`), is compiled to PTX at runtime with NVRTC
//! (`cudarc::nvrtc`), loads the module through the raw CUDA driver API
//! (`cuModuleLoadData`/`cuModuleGetFunction`), and launches with `cuLaunchKernel`
//! on the crate's [`CudaStream`].
//!
//! Single-pass decode attention with online softmax, supporting multi-query/grouped-query
//! attention (GQA) and causal masking without dynamic GPU memory allocation.

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
/// Kernel symbol name exported by `paged_attention.cu`.
const FUNC_NAME: &str = "paged_attention_decode_kernel";
/// Canonical CUDA kernel source, compiled to PTX at runtime via NVRTC.
const KERNEL_SRC: &str = include_str!("../kernels/paged_attention.cu");
/// Number of threads per block (one warp).
const BLOCK_DIM: u32 = 32;

/// RAII wrapper around compiled and loaded PagedAttention decode kernel.
///
/// Owns the CUDA `CUmodule` (freed on [`Drop`]) and resolved `CUfunction`.
pub struct PagedAttention {
    device: Arc<CudaDevice>,
    cu_module: CUmodule,
    func: CUfunction,
}

// SAFETY: `CUmodule`/`CUfunction` are opaque CUDA driver handles tied to the
// `Arc<CudaDevice>`; they can be safely transferred and shared across threads.
unsafe impl Send for PagedAttention {}
unsafe impl Sync for PagedAttention {}

impl PagedAttention {
    /// Compiles `paged_attention.cu` with NVRTC and loads the kernel into `device`.
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

    /// Launches the PagedAttention decode kernel asynchronously on `stream`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &CudaStream,
        q: &DeviceBuffer,
        pool: &DeviceBuffer,
        block_table: &DeviceBuffer,
        out: &DeviceBuffer,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        block_tokens: usize,
        seq_tokens: usize,
        query_pos: usize,
        causal: bool,
    ) -> Result<(), CudaError> {
        self.launch_with_pos_ptr(
            stream,
            q,
            pool,
            block_table,
            out,
            n_head,
            n_head_kv,
            head_dim,
            block_tokens,
            seq_tokens,
            query_pos,
            causal,
            None,
        )
    }

    /// Launches the PagedAttention decode kernel with an optional dynamic device-side position pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_with_pos_ptr(
        &self,
        stream: &CudaStream,
        q: &DeviceBuffer,
        pool: &DeviceBuffer,
        block_table: &DeviceBuffer,
        out: &DeviceBuffer,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        block_tokens: usize,
        seq_tokens: usize,
        query_pos: usize,
        causal: bool,
        pos_ptr: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        if n_head < 1 {
            return Err(CudaError::InvalidSize {
                expected: 1,
                actual: n_head,
            });
        }
        if n_head_kv < 1 {
            return Err(CudaError::InvalidSize {
                expected: 1,
                actual: n_head_kv,
            });
        }
        if !n_head.is_multiple_of(n_head_kv) {
            return Err(CudaError::InvalidSize {
                expected: n_head_kv,
                actual: n_head % n_head_kv,
            });
        }
        if head_dim < 1 {
            return Err(CudaError::InvalidSize {
                expected: 1,
                actual: head_dim,
            });
        }
        if block_tokens < 1 {
            return Err(CudaError::InvalidSize {
                expected: 1,
                actual: block_tokens,
            });
        }
        if seq_tokens < 1 {
            return Err(CudaError::InvalidSize {
                expected: 1,
                actual: seq_tokens,
            });
        }

        let q_expected = n_head * head_dim * 4;
        if q.size() < q_expected {
            return Err(CudaError::InvalidSize {
                expected: q_expected,
                actual: q.size(),
            });
        }
        if out.size() < q_expected {
            return Err(CudaError::InvalidSize {
                expected: q_expected,
                actual: out.size(),
            });
        }
        if pool.size() < 4 {
            return Err(CudaError::InvalidSize {
                expected: 4,
                actual: pool.size(),
            });
        }

        let n_blocks = seq_tokens.div_ceil(block_tokens);
        let bt_expected = n_blocks * 4;
        if block_table.size() < bt_expected {
            return Err(CudaError::InvalidSize {
                expected: bt_expected,
                actual: block_table.size(),
            });
        }

        let n_head_i: i32 = i32::try_from(n_head).map_err(|_| CudaError::InvalidSize {
            expected: i32::MAX as usize,
            actual: n_head,
        })?;
        let n_head_kv_i: i32 = i32::try_from(n_head_kv).map_err(|_| CudaError::InvalidSize {
            expected: i32::MAX as usize,
            actual: n_head_kv,
        })?;
        let head_dim_i: i32 = i32::try_from(head_dim).map_err(|_| CudaError::InvalidSize {
            expected: i32::MAX as usize,
            actual: head_dim,
        })?;
        let block_tokens_i: i32 =
            i32::try_from(block_tokens).map_err(|_| CudaError::InvalidSize {
                expected: i32::MAX as usize,
                actual: block_tokens,
            })?;
        let seq_tokens_i: i32 = i32::try_from(seq_tokens).map_err(|_| CudaError::InvalidSize {
            expected: i32::MAX as usize,
            actual: seq_tokens,
        })?;
        let query_pos_i: i32 = i32::try_from(query_pos).map_err(|_| CudaError::InvalidSize {
            expected: i32::MAX as usize,
            actual: query_pos,
        })?;
        let causal_i: i32 = if causal { 1 } else { 0 };

        let scale: f32 = 1.0f32 / (head_dim as f32).sqrt();
        let grid_x: u32 = n_head as u32;
        let shared_bytes = 0u32;

        let q_addr: u64 = q.device_ptr();
        let pool_addr: u64 = pool.device_ptr();
        let block_table_addr: u64 = block_table.device_ptr();
        let out_addr: u64 = out.device_ptr();
        let pos_ptr_addr: u64 = pos_ptr.map(|p| p.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 13] = [
            &q_addr as *const u64 as *mut std::ffi::c_void,
            &pool_addr as *const u64 as *mut std::ffi::c_void,
            &block_table_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &n_head_i as *const i32 as *mut std::ffi::c_void,
            &n_head_kv_i as *const i32 as *mut std::ffi::c_void,
            &head_dim_i as *const i32 as *mut std::ffi::c_void,
            &block_tokens_i as *const i32 as *mut std::ffi::c_void,
            &seq_tokens_i as *const i32 as *mut std::ffi::c_void,
            &query_pos_i as *const i32 as *mut std::ffi::c_void,
            &causal_i as *const i32 as *mut std::ffi::c_void,
            &scale as *const f32 as *mut std::ffi::c_void,
            &pos_ptr_addr as *const u64 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured a valid CUDA context on this thread.
        // `self.func` is a valid `CUfunction` from a live module.
        // `stream.raw()` is a valid `CUstream`. `args` points to 13 raw pointers that
        // map to the kernel parameters.
        let res = unsafe {
            let lib = sys::lib();
            lib.cuLaunchKernel(
                self.func,
                grid_x,
                1,
                1,
                BLOCK_DIM,
                1,
                1,
                shared_bytes,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            )
        };

        if res != CUresult::CUDA_SUCCESS {
            return Err(CudaError::KernelLaunch("cuLaunchKernel (paged_attn)", res));
        }

        Ok(())
    }
}

impl Drop for PagedAttention {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = self.device.bind_to_thread();
            // SAFETY:
            // `self.cu_module` was created by `cuModuleLoadData` in `PagedAttention::new`
            // and has not been unloaded yet.
            unsafe {
                let lib = sys::lib();
                let _res = lib.cuModuleUnload(self.cu_module);
            }
            self.cu_module = std::ptr::null_mut();
        }
    }
}
