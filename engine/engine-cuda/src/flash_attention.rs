//! FlashAttention-2 Causal Prefill GPU Launcher (Phase 11, Sub-change 11.2).
//!
//! Computes batched causal self-attention directly over resident paged KV blocks
//! with online softmax renormalization in registers.
//!
//! Kernel source lives at `../kernels/flash_attention_2.cu`.

use crate::DeviceBuffer;
use crate::error::CudaError;
use crate::streams::CudaStream;
use cudarc::driver::CudaDevice;
use cudarc::driver::sys::{self, CUfunction, CUmodule, CUresult};
use cudarc::nvrtc::{self, Ptx};
use std::ffi::CString;
use std::sync::Arc;

const KERNEL_ARCH: &str = "compute_86";
const FUNC_NAME: &str = "flash_attention_2_kernel";
const KERNEL_SRC: &str = include_str!("../kernels/flash_attention_2.cu");

/// RAII wrapper around compiled FlashAttention-2 prefill kernel.
pub struct FlashAttention2 {
    device: Arc<CudaDevice>,
    cu_module: CUmodule,
    func: CUfunction,
}

unsafe impl Send for FlashAttention2 {}
unsafe impl Sync for FlashAttention2 {}

impl Drop for FlashAttention2 {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuModuleUnload(self.cu_module);
            }
        }
    }
}

impl FlashAttention2 {
    /// Compiles `flash_attention_2.cu` via NVRTC and loads kernel function.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CudaError> {
        device.bind_to_thread()?;

        let ptx: Ptx = nvrtc::compile_ptx_with_opts(
            KERNEL_SRC,
            nvrtc::CompileOptions {
                arch: Some(KERNEL_ARCH),
                include_paths: vec![],
                use_fast_math: Some(true),
                ..Default::default()
            },
        )
        .map_err(|e| CudaError::KernelCompile(format!("{e:?}")))?;

        let ptx_src = ptx.to_src();
        let ptx_c = CString::new(ptx_src.as_str()).map_err(|_| {
            CudaError::KernelLoad("CString::new", CUresult::CUDA_ERROR_INVALID_VALUE)
        })?;

        let mut cu_module: CUmodule = std::ptr::null_mut();
        let mut func: CUfunction = std::ptr::null_mut();

        unsafe {
            let lib = sys::lib();
            let res = lib.cuModuleLoadData(
                &mut cu_module,
                ptx_c.as_ptr() as *const std::ffi::c_void,
            );
            if res != CUresult::CUDA_SUCCESS || cu_module.is_null() {
                return Err(CudaError::KernelLoad(
                    "cuModuleLoadData (flash_attention_2)",
                    res,
                ));
            }

            let c_func = CString::new(FUNC_NAME).unwrap();
            let res_f = lib.cuModuleGetFunction(&mut func, cu_module, c_func.as_ptr());
            if res_f != CUresult::CUDA_SUCCESS || func.is_null() {
                let _ = lib.cuModuleUnload(cu_module);
                return Err(CudaError::KernelLoad("cuModuleGetFunction", res_f));
            }
        }

        Ok(Self {
            device,
            cu_module,
            func,
        })
    }

    /// Launches FlashAttention-2 causal prefill kernel.
    #[allow(clippy::too_many_arguments)]
    /// Launches FlashAttention-2 causal kernel with a static host pos_offset.
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
        q_tokens: usize,
        pos_offset: usize,
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
            q_tokens,
            pos_offset,
            None,
        )
    }

    /// Launches FlashAttention-2 causal kernel with an optional dynamic device pos_ptr (for CUDA Graphs).
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
        q_tokens: usize,
        pos_offset: usize,
        pos_ptr: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        let q_expected = q_tokens * n_head * head_dim * 4;
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

        let total_tokens = pos_offset + q_tokens;
        let n_blocks = total_tokens.div_ceil(block_tokens);
        let bt_expected = n_blocks * 4;
        if block_table.size() < bt_expected {
            return Err(CudaError::InvalidSize {
                expected: bt_expected,
                actual: block_table.size(),
            });
        }

        let scale: f32 = 1.0f32 / (head_dim as f32).sqrt();
        let grid_x: u32 = n_head as u32;
        let grid_y: u32 = q_tokens as u32;
        const BLOCK_X: u32 = 32;

        let n_head_i: i32 = n_head as i32;
        let n_head_kv_i: i32 = n_head_kv as i32;
        let head_dim_i: i32 = head_dim as i32;
        let block_tokens_i: i32 = block_tokens as i32;
        let q_tokens_i: i32 = q_tokens as i32;
        let pos_offset_i: i32 = pos_offset as i32;

        let q_addr: u64 = q.device_ptr();
        let pool_addr: u64 = pool.device_ptr();
        let block_table_addr: u64 = block_table.device_ptr();
        let out_addr: u64 = out.device_ptr();
        let pos_ptr_addr: u64 = pos_ptr.map(|p| p.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 12] = [
            &q_addr as *const u64 as *mut std::ffi::c_void,
            &pool_addr as *const u64 as *mut std::ffi::c_void,
            &block_table_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &n_head_i as *const i32 as *mut std::ffi::c_void,
            &n_head_kv_i as *const i32 as *mut std::ffi::c_void,
            &head_dim_i as *const i32 as *mut std::ffi::c_void,
            &block_tokens_i as *const i32 as *mut std::ffi::c_void,
            &q_tokens_i as *const i32 as *mut std::ffi::c_void,
            &pos_offset_i as *const i32 as *mut std::ffi::c_void,
            &scale as *const f32 as *mut std::ffi::c_void,
            &pos_ptr_addr as *const u64 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.func,
                grid_x,
                grid_y,
                1,
                BLOCK_X,
                1,
                1,
                0,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch(
                    "cuLaunchKernel (FlashAttention2)",
                    res,
                ));
            }
        }

        Ok(())
    }
}
