//! GPU paged KV-cache kernels and RAII launcher (f4-paged-kvcache).
//!
//! The CUDA kernel source lives at [`../kernels/paged_kv.cu`](kernels)
//! (embedded via `include_str!`), is compiled to PTX at runtime with NVRTC
//! (`cudarc::nvrtc`), loads the module through the raw CUDA driver API
//! (`cuModuleLoadData`/`cuModuleGetFunction`), and launches with `cuLaunchKernel`
//! on the crate's own [`CudaStream`].
//!
//! Pool layout is bit-identical to the CPU flat buffer in `engine-kvcache`:
//!   physical block `b`  -> float offset `b * floats_per_block`
//!   token slot `s`      -> `+ s * floats_per_token` (key row then value row)
//!   key row             -> `+ 0 .. row_len`
//!   value row           -> `+ row_len .. 2 * row_len`
//!   `row_len = heads * head_dim` ; `floats_per_token = 2 * row_len` ;
//!   `floats_per_block = block_tokens * floats_per_token`.

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
/// Kernel symbol name for append exported by `paged_kv.cu`.
const FUNC_APPEND: &str = "paged_append_kv_kernel";
/// Kernel symbol name for gather exported by `paged_kv.cu`.
const FUNC_GATHER: &str = "paged_gather_kernel";
/// Canonical CUDA kernel source, compiled to PTX at runtime via NVRTC.
const KERNEL_SRC: &str = include_str!("../kernels/paged_kv.cu");

/// Layout specification for a paged KV cache device pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvLayout {
    /// Total number of physical blocks in the pool.
    pub n_blocks: usize,
    /// Logical tokens per block.
    pub block_tokens: usize,
    /// Floats per key / value row (`heads * head_dim`).
    pub row_len: usize,
}

impl PagedKvLayout {
    /// Number of floats stored per token (key row + value row).
    pub fn floats_per_token(&self) -> usize {
        2 * self.row_len
    }

    /// Number of floats stored per physical block.
    pub fn floats_per_block(&self) -> usize {
        self.block_tokens * self.floats_per_token()
    }

    /// Total pool capacity in floats.
    pub fn floats_total(&self) -> usize {
        self.n_blocks * self.floats_per_block()
    }
}

/// RAII wrapper around compiled and loaded paged KV-cache kernels.
///
/// Owns the CUDA `CUmodule` (freed on [`Drop`]) and the resolved `CUfunction`
/// handles for append and gather operations.
pub struct PagedKvGpu {
    device: Arc<CudaDevice>,
    cu_module: CUmodule,
    func_append: CUfunction,
    func_gather: CUfunction,
}

// SAFETY: `CUmodule`/`CUfunction` are opaque CUDA driver handles tied to the
// `Arc<CudaDevice>`; they can be safely transferred and shared across threads.
unsafe impl Send for PagedKvGpu {}
unsafe impl Sync for PagedKvGpu {}

impl PagedKvGpu {
    /// Compiles `paged_kv.cu` with NVRTC and loads it into `device`.
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

        let func_append_c = CString::new(FUNC_APPEND)
            .map_err(|_| CudaError::KernelCompile("NUL byte in function name".into()))?;
        let mut func_append: CUfunction = std::ptr::null_mut();
        // SAFETY:
        // `cu_module` was produced by `cuModuleLoadData` above and is non-null.
        // `&mut func_append` is a valid stack pointer.
        // `func_append_c.as_ptr()` points to a NUL-terminated symbol name.
        let res = unsafe {
            let lib = sys::lib();
            lib.cuModuleGetFunction(&mut func_append, cu_module, func_append_c.as_ptr())
        };
        if res != CUresult::CUDA_SUCCESS || func_append.is_null() {
            // SAFETY: Free the already-loaded module to avoid leaking it on this error path.
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuModuleUnload(cu_module);
            }
            return Err(CudaError::KernelLoad("cuModuleGetFunction (append)", res));
        }

        let func_gather_c = CString::new(FUNC_GATHER)
            .map_err(|_| CudaError::KernelCompile("NUL byte in function name".into()))?;
        let mut func_gather: CUfunction = std::ptr::null_mut();
        // SAFETY:
        // `cu_module` was produced by `cuModuleLoadData` above and is non-null.
        // `&mut func_gather` is a valid stack pointer.
        // `func_gather_c.as_ptr()` points to a NUL-terminated symbol name.
        let res = unsafe {
            let lib = sys::lib();
            lib.cuModuleGetFunction(&mut func_gather, cu_module, func_gather_c.as_ptr())
        };
        if res != CUresult::CUDA_SUCCESS || func_gather.is_null() {
            // SAFETY: Free the already-loaded module to avoid leaking it on this error path.
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuModuleUnload(cu_module);
            }
            return Err(CudaError::KernelLoad("cuModuleGetFunction (gather)", res));
        }

        Ok(Self {
            device,
            cu_module,
            func_append,
            func_gather,
        })
    }

    /// Appends `n_tokens` key and value rows into the paged `pool` according to `block_table`.
    #[allow(clippy::too_many_arguments)]
    pub fn append_kv(
        &self,
        stream: &CudaStream,
        layout: &PagedKvLayout,
        pool: &DeviceBuffer,
        keys: &DeviceBuffer,
        values: &DeviceBuffer,
        block_table: &DeviceBuffer,
        start_token: usize,
        n_tokens: usize,
    ) -> Result<(), CudaError> {
        const BLOCK_X: u32 = 1024;

        let pool_expected = layout.floats_total() * 4;
        if pool.size() < pool_expected {
            return Err(CudaError::InvalidSize {
                expected: pool_expected,
                actual: pool.size(),
            });
        }

        let kv_expected = n_tokens * layout.row_len * 4;
        if keys.size() < kv_expected {
            return Err(CudaError::InvalidSize {
                expected: kv_expected,
                actual: keys.size(),
            });
        }
        if values.size() < kv_expected {
            return Err(CudaError::InvalidSize {
                expected: kv_expected,
                actual: values.size(),
            });
        }

        if block_table.size() == 0 {
            return Err(CudaError::InvalidSize {
                expected: 4,
                actual: 0,
            });
        }

        if n_tokens == 0 {
            return Ok(());
        }

        let total_threads = (n_tokens * layout.row_len) as u32;
        let grid_x = total_threads.div_ceil(BLOCK_X);

        let keys_addr: u64 = keys.device_ptr();
        let values_addr: u64 = values.device_ptr();
        let block_table_addr: u64 = block_table.device_ptr();
        let pool_addr: u64 = pool.device_ptr();
        let n_tokens_i: i32 = n_tokens as i32;
        let start_token_i: i32 = start_token as i32;
        let row_len_i: i32 = layout.row_len as i32;
        let block_tokens_i: i32 = layout.block_tokens as i32;

        let args: [*mut std::ffi::c_void; 8] = [
            &keys_addr as *const u64 as *mut std::ffi::c_void,
            &values_addr as *const u64 as *mut std::ffi::c_void,
            &block_table_addr as *const u64 as *mut std::ffi::c_void,
            &pool_addr as *const u64 as *mut std::ffi::c_void,
            &n_tokens_i as *const i32 as *mut std::ffi::c_void,
            &start_token_i as *const i32 as *mut std::ffi::c_void,
            &row_len_i as *const i32 as *mut std::ffi::c_void,
            &block_tokens_i as *const i32 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured a valid CUDA context on this thread.
        // `self.func_append` is a valid `CUfunction` from a live module.
        // `stream.raw()` is a valid `CUstream`. `args` points to 8 raw pointers that
        // map to the kernel parameters.
        let res = unsafe {
            let lib = sys::lib();
            lib.cuLaunchKernel(
                self.func_append,
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

    /// Gathers contiguous `[n_tokens, row_len]` key rows from `pool` into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn read_keys(
        &self,
        stream: &CudaStream,
        layout: &PagedKvLayout,
        pool: &DeviceBuffer,
        block_table: &DeviceBuffer,
        out: &DeviceBuffer,
        start_token: usize,
        n_tokens: usize,
    ) -> Result<(), CudaError> {
        self.gather(
            stream,
            layout,
            pool,
            block_table,
            out,
            start_token,
            n_tokens,
            0,
        )
    }

    /// Gathers contiguous `[n_tokens, row_len]` value rows from `pool` into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn read_values(
        &self,
        stream: &CudaStream,
        layout: &PagedKvLayout,
        pool: &DeviceBuffer,
        block_table: &DeviceBuffer,
        out: &DeviceBuffer,
        start_token: usize,
        n_tokens: usize,
    ) -> Result<(), CudaError> {
        self.gather(
            stream,
            layout,
            pool,
            block_table,
            out,
            start_token,
            n_tokens,
            1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gather(
        &self,
        stream: &CudaStream,
        layout: &PagedKvLayout,
        pool: &DeviceBuffer,
        block_table: &DeviceBuffer,
        out: &DeviceBuffer,
        start_token: usize,
        n_tokens: usize,
        is_value: i32,
    ) -> Result<(), CudaError> {
        const BLOCK_X: u32 = 1024;

        let pool_expected = layout.floats_total() * 4;
        if pool.size() < pool_expected {
            return Err(CudaError::InvalidSize {
                expected: pool_expected,
                actual: pool.size(),
            });
        }

        let out_expected = n_tokens * layout.row_len * 4;
        if out.size() < out_expected {
            return Err(CudaError::InvalidSize {
                expected: out_expected,
                actual: out.size(),
            });
        }

        if block_table.size() == 0 {
            return Err(CudaError::InvalidSize {
                expected: 4,
                actual: 0,
            });
        }

        if n_tokens == 0 {
            return Ok(());
        }

        let total_threads = (n_tokens * layout.row_len) as u32;
        let grid_x = total_threads.div_ceil(BLOCK_X);

        let pool_addr: u64 = pool.device_ptr();
        let block_table_addr: u64 = block_table.device_ptr();
        let out_addr: u64 = out.device_ptr();
        let n_tokens_i: i32 = n_tokens as i32;
        let start_token_i: i32 = start_token as i32;
        let row_len_i: i32 = layout.row_len as i32;
        let block_tokens_i: i32 = layout.block_tokens as i32;
        let is_value_i: i32 = is_value;

        let args: [*mut std::ffi::c_void; 8] = [
            &pool_addr as *const u64 as *mut std::ffi::c_void,
            &block_table_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &n_tokens_i as *const i32 as *mut std::ffi::c_void,
            &start_token_i as *const i32 as *mut std::ffi::c_void,
            &row_len_i as *const i32 as *mut std::ffi::c_void,
            &block_tokens_i as *const i32 as *mut std::ffi::c_void,
            &is_value_i as *const i32 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured a valid CUDA context on this thread.
        // `self.func_gather` is a valid `CUfunction` from a live module.
        // `stream.raw()` is a valid `CUstream`. `args` points to 8 raw pointers that
        // map to the kernel parameters.
        let res = unsafe {
            let lib = sys::lib();
            lib.cuLaunchKernel(
                self.func_gather,
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

impl Drop for PagedKvGpu {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = self.device.bind_to_thread();
            // SAFETY:
            // `self.cu_module` was created by `cuModuleLoadData` in `PagedKvGpu::new`
            // and has not been unloaded yet.
            unsafe {
                let lib = sys::lib();
                let _res = lib.cuModuleUnload(self.cu_module);
            }
            self.cu_module = std::ptr::null_mut();
        }
    }
}
