//! GPU logit bitmasking kernel and RAII launcher for grammar-constrained decoding.

use crate::DeviceBuffer;
use crate::error::CudaError;
use crate::streams::CudaStream;
use cudarc::driver::CudaDevice;
use cudarc::driver::sys::{self, CUfunction, CUmodule, CUresult};
use cudarc::nvrtc::{self, Ptx};
use std::ffi::CString;
use std::sync::Arc;

const KERNEL_ARCH: &str = "compute_86";
const FUNC_NAME: &str = "apply_logit_mask_kernel";
const KERNEL_SRC: &str = include_str!("../kernels/logit_mask.cu");

/// RAII wrapper around compiled `logit_mask.cu` kernel.
pub struct LogitMaskGpu {
    device: Arc<CudaDevice>,
    cu_module: CUmodule,
    func: CUfunction,
}

// SAFETY: CUmodule / CUfunction are thread-safe opaque CUDA driver handles.
unsafe impl Send for LogitMaskGpu {}
unsafe impl Sync for LogitMaskGpu {}

impl LogitMaskGpu {
    /// Compiles `logit_mask.cu` with NVRTC and loads the kernel into `device`.
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
        let mut func: CUfunction = std::ptr::null_mut();
        let r = unsafe {
            let lib = sys::lib();
            lib.cuModuleGetFunction(&mut func, cu_module, name_c.as_ptr())
        };
        if r != CUresult::CUDA_SUCCESS || func.is_null() {
            return Err(CudaError::KernelLoad("cuModuleGetFunction", r));
        }

        Ok(Self {
            device,
            cu_module,
            func,
        })
    }

    /// Applies the packed `u32` bitmask in-place to `logits_dev` on `stream`.
    pub fn apply_mask(
        &self,
        stream: &CudaStream,
        logits_dev: &DeviceBuffer,
        mask_dev: &DeviceBuffer,
        vocab_size: usize,
    ) -> Result<(), CudaError> {
        self.device.bind_to_thread()?;

        let block_x: u32 = 256;
        let grid_x: u32 = ((vocab_size as u32) + block_x - 1) / block_x;

        let logits_addr = logits_dev.device_ptr();
        let mask_addr = mask_dev.device_ptr();
        let vocab_i32 = vocab_size as i32;

        let args: [*mut std::ffi::c_void; 3] = [
            &logits_addr as *const u64 as *mut std::ffi::c_void,
            &mask_addr as *const u64 as *mut std::ffi::c_void,
            &vocab_i32 as *const i32 as *mut std::ffi::c_void,
        ];

        let res = unsafe {
            let lib = sys::lib();
            lib.cuLaunchKernel(
                self.func,
                grid_x,
                1,
                1,
                block_x,
                1,
                1,
                0,
                stream.raw(),
                args.as_ptr() as *mut _,
                std::ptr::null_mut(),
            )
        };

        if res != CUresult::CUDA_SUCCESS {
            return Err(CudaError::KernelLaunch(FUNC_NAME, res));
        }

        Ok(())
    }
}

impl Drop for LogitMaskGpu {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuModuleUnload(self.cu_module);
            }
            self.cu_module = std::ptr::null_mut();
        }
    }
}
