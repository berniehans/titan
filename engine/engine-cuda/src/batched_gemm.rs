//! Batched Quantized Matrix Multiplication (GEMM) GPU launcher (Phase 11, Sub-change 11.1).
//!
//! Computes `Y = X * W^T` for arbitrary batch size `M` (e.g. sequence lengths 1..512).
//!
//! Kernel source lives at `../kernels/gemm_quant.cu`, compiled via NVRTC to PTX at runtime.

use crate::DeviceBuffer;
use crate::error::CudaError;
use crate::multiformat_gemv::GemvFormat;
use crate::streams::CudaStream;
use cudarc::driver::CudaDevice;
use cudarc::driver::sys::{self, CUfunction, CUmodule, CUresult};
use cudarc::nvrtc::{self, Ptx};
use std::ffi::CString;
use std::sync::Arc;

const KERNEL_ARCH: &str = "compute_86";
const FUNC_NAME_Q4K: &str = "gemm_q4k_kernel";
const FUNC_NAME_Q6K: &str = "gemm_q6k_kernel";
const FUNC_NAME_Q8: &str = "gemm_q8_kernel";
const FUNC_NAME_F16: &str = "gemm_f16_kernel";
const FUNC_NAME_F32: &str = "gemm_f32_kernel";

const KERNEL_SRC: &str = include_str!("../kernels/gemm_quant.cu");
const BLOCK_X: u32 = 256;

/// RAII launcher for Batched Quantized Matrix Multiplication kernels.
pub struct BatchedGEMM {
    device: Arc<CudaDevice>,
    cu_module: CUmodule,
    fn_q4k: CUfunction,
    fn_q6k: CUfunction,
    fn_q8: CUfunction,
    fn_f16: CUfunction,
    _fn_f32: CUfunction,
}

unsafe impl Send for BatchedGEMM {}
unsafe impl Sync for BatchedGEMM {}

impl Drop for BatchedGEMM {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuModuleUnload(self.cu_module);
            }
        }
    }
}

impl BatchedGEMM {
    /// Compiles `gemm_quant.cu` via NVRTC and loads driver functions.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CudaError> {
        device.bind_to_thread()?;

        let ptx: Ptx = nvrtc::compile_ptx_with_opts(
            KERNEL_SRC,
            nvrtc::CompileOptions {
                arch: Some(KERNEL_ARCH),
                include_paths: vec![],
                ..Default::default()
            },
        )
        .map_err(|e| CudaError::KernelCompile(format!("{e:?}")))?;

        let ptx_src = ptx.to_src();
        let ptx_c = CString::new(ptx_src.as_str()).map_err(|_| CudaError::KernelLoad("CString::new", CUresult::CUDA_ERROR_INVALID_VALUE))?;

        let mut cu_module: CUmodule = std::ptr::null_mut();
        let mut fn_q4k: CUfunction = std::ptr::null_mut();
        let mut fn_q6k: CUfunction = std::ptr::null_mut();
        let mut fn_q8: CUfunction = std::ptr::null_mut();
        let mut fn_f16: CUfunction = std::ptr::null_mut();
        let mut fn_f32: CUfunction = std::ptr::null_mut();

        unsafe {
            let lib = sys::lib();
            let res = lib.cuModuleLoadData(&mut cu_module, ptx_c.as_ptr() as *const std::ffi::c_void);
            if res != CUresult::CUDA_SUCCESS || cu_module.is_null() {
                return Err(CudaError::KernelLoad("cuModuleLoadData (gemm_quant)", res));
            }

            let c_q4k = CString::new(FUNC_NAME_Q4K).unwrap();
            let c_q6k = CString::new(FUNC_NAME_Q6K).unwrap();
            let c_q8  = CString::new(FUNC_NAME_Q8).unwrap();
            let c_f16 = CString::new(FUNC_NAME_F16).unwrap();
            let c_f32 = CString::new(FUNC_NAME_F32).unwrap();

            let r1 = lib.cuModuleGetFunction(&mut fn_q4k, cu_module, c_q4k.as_ptr());
            let r2 = lib.cuModuleGetFunction(&mut fn_q6k, cu_module, c_q6k.as_ptr());
            let r3 = lib.cuModuleGetFunction(&mut fn_q8,  cu_module, c_q8.as_ptr());
            let r4 = lib.cuModuleGetFunction(&mut fn_f16, cu_module, c_f16.as_ptr());
            let r5 = lib.cuModuleGetFunction(&mut fn_f32, cu_module, c_f32.as_ptr());

            if r1 != CUresult::CUDA_SUCCESS || r2 != CUresult::CUDA_SUCCESS || r3 != CUresult::CUDA_SUCCESS || r4 != CUresult::CUDA_SUCCESS || r5 != CUresult::CUDA_SUCCESS {
                let _ = lib.cuModuleUnload(cu_module);
                return Err(CudaError::KernelLoad("cuModuleGetFunction", r1));
            }
        }

        Ok(Self {
            device,
            cu_module,
            fn_q4k,
            fn_q6k,
            fn_q8,
            fn_f16,
            _fn_f32: fn_f32,
        })
    }

    /// Launches batched quantized GEMM `Out[M, ne1] = X[M, ne0] * W[ne1, ne0]^T`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        stream: &CudaStream,
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        out: &DeviceBuffer,
        ne0: usize,
        ne1: usize,
        batch_size: usize,
        format: GemvFormat,
    ) -> Result<(), CudaError> {
        if batch_size == 0 || ne0 == 0 || ne1 == 0 {
            return Ok(());
        }

        let expected_weight_bytes = match format {
            GemvFormat::Q4K => (ne0 / 256) * 144 * ne1,
            GemvFormat::Q6K => (ne0 / 256) * 210 * ne1,
            GemvFormat::Q8  => (ne0 / 32) * 34 * ne1,
            GemvFormat::F16 => ne0 * 2 * ne1,
        };

        if weights.size() < expected_weight_bytes {
            return Err(CudaError::InvalidSize {
                expected: expected_weight_bytes,
                actual: weights.size(),
            });
        }

        let expected_x_bytes = batch_size * ne0 * std::mem::size_of::<f32>();
        if x.size() < expected_x_bytes {
            return Err(CudaError::InvalidSize {
                expected: expected_x_bytes,
                actual: x.size(),
            });
        }

        let expected_out_bytes = batch_size * ne1 * std::mem::size_of::<f32>();
        if out.size() < expected_out_bytes {
            return Err(CudaError::InvalidSize {
                expected: expected_out_bytes,
                actual: out.size(),
            });
        }

        let func = match format {
            GemvFormat::Q4K => self.fn_q4k,
            GemvFormat::Q6K => self.fn_q6k,
            GemvFormat::Q8  => self.fn_q8,
            GemvFormat::F16 => self.fn_f16,
        };

        let ne0_i: i32 = ne0 as i32;
        let ne1_i: i32 = ne1 as i32;
        let batch_i: i32 = batch_size as i32;

        let w_addr: u64 = weights.device_ptr();
        let x_addr: u64 = x.device_ptr();
        let out_addr: u64 = out.device_ptr();

        let args: [*mut std::ffi::c_void; 6] = [
            &w_addr as *const u64 as *mut std::ffi::c_void,
            &x_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &ne1_i as *const i32 as *mut std::ffi::c_void,
            &batch_i as *const i32 as *mut std::ffi::c_void,
        ];

        let grid_x = (ne1 as u32).div_ceil(BLOCK_X);
        let grid_y = batch_size as u32;

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                func,
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
                return Err(CudaError::KernelLaunch("cuLaunchKernel (BatchedGEMM)", res));
            }
        }

        Ok(())
    }
}
