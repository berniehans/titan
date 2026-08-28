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
const FUNC_NAME_QUANTIZE_Q8_1: &str = "quantize_row_q8_1_kernel";
const FUNC_NAME_Q4K: &str = "gemm_q4k_kernel";
const FUNC_NAME_Q6K: &str = "gemm_q6k_kernel";
const FUNC_NAME_Q8: &str = "gemm_q8_kernel";
const FUNC_NAME_F16: &str = "gemm_f16_kernel";
const FUNC_NAME_F32: &str = "gemm_f32_kernel";
const FUNC_NAME_Q4K_FUSED_GATE_UP_SWIGLU: &str = "gemm_q4k_fused_gate_up_swiglu_kernel";
const FUNC_NAME_Q4K_BATCHED_GATE_UP_SWIGLU: &str = "gemm_q4k_batched_gate_up_swiglu_kernel";
const FUNC_NAME_Q4K_BATCHED: &str = "gemm_q4k_batched_kernel";
const FUNC_NAME_Q6K_BATCHED: &str = "gemm_q6k_batched_kernel";
const FUNC_NAME_Q6K_SPLITK: &str = "gemm_q6k_splitk_kernel";
const FUNC_NAME_Q4K_SPLITK: &str = "gemm_q4k_splitk_kernel";
const FUNC_NAME_FUSED_QKV: &str = "gemm_fused_qkv_kernel";
const FUNC_NAME_FUSED_QKV_MULTI_ROW: &str = "gemm_fused_qkv_multi_row_kernel";
const FUNC_NAME_FUSED_QKV_Q4K: &str = "gemm_fused_qkv_q4k_kernel";
const FUNC_NAME_FUSED_QKV_BATCHED: &str = "gemm_fused_qkv_batched_kernel";
const FUNC_NAME_GET_ROWS_Q4K: &str = "get_rows_q4k_kernel";
const FUNC_NAME_GET_ROWS_Q6K: &str = "get_rows_q6k_kernel";
const FUNC_NAME_GET_ROWS_Q8_0: &str = "get_rows_q8_0_kernel";
const FUNC_NAME_GET_ROWS_F16: &str = "get_rows_f16_kernel";
const FUNC_NAME_SAMPLE_GREEDY: &str = "gpu_sample_greedy_kernel";
const FUNC_NAME_ADVANCE_TOKEN: &str = "gpu_advance_token_step_kernel";
const FUNC_NAME_Q4K_MMA: &str = "gemm_q4k_mma_kernel";
const FUNC_NAME_Q4K_MULTI_ROW: &str = "gemm_q4k_multi_row_kernel";
const FUNC_NAME_Q6K_MULTI_ROW: &str = "gemm_q6k_multi_row_kernel";
const FUNC_NAME_FUSED_QKV_Q4K_MULTI_ROW: &str = "gemm_fused_qkv_q4k_multi_row_kernel";
const FUNC_NAME_Q4K_FUSED_GATE_UP_SWIGLU_MULTI_ROW: &str = "gemm_q4k_fused_gate_up_swiglu_multi_row_kernel";
const FUNC_NAME_Q4K_FUSED_GATE_UP_SWIGLU_MMA: &str = "gemm_q4k_fused_gate_up_swiglu_mma_kernel";
const FUNC_NAME_REDUCE_SPLITK: &str = "reduce_splitk_kernel";

const KERNEL_SRC: &str = include_str!("../kernels/gemm_quant.cu");
const BLOCK_X: u32 = 128;

/// RAII launcher for Batched Quantized Matrix Multiplication kernels.
pub struct BatchedGEMM {
    device: Arc<CudaDevice>,
    cu_module: CUmodule,
    fn_quantize_q8_1: CUfunction,
    fn_q4k: CUfunction,
    fn_q4k_batched: CUfunction,
    fn_q4k_multi_row: CUfunction,
    fn_q6k: CUfunction,
    fn_q6k_multi_row: CUfunction,
    fn_q8: CUfunction,
    fn_f16: CUfunction,
    _fn_f32: CUfunction,
    fn_q4k_fused_gate_up_swiglu: CUfunction,
    fn_q4k_batched_gate_up_swiglu: CUfunction,
    fn_q4k_fused_gate_up_swiglu_multi_row: CUfunction,
    fn_q6k_batched: CUfunction,
    fn_q6k_splitk: CUfunction,
    fn_q4k_splitk: CUfunction,
    fn_fused_qkv: CUfunction,
    fn_fused_qkv_multi_row: CUfunction,
    fn_fused_qkv_q4k: CUfunction,
    fn_fused_qkv_q4k_multi_row: CUfunction,
    fn_fused_qkv_batched: CUfunction,
    fn_get_rows_q4k: CUfunction,
    fn_get_rows_q6k: CUfunction,
    fn_get_rows_q8_0: CUfunction,
    fn_get_rows_f16: CUfunction,
    fn_sample_greedy: CUfunction,
    fn_advance_token: CUfunction,
    fn_q4k_mma: CUfunction,
    fn_q4k_fused_gate_up_swiglu_mma: CUfunction,
    fn_reduce_splitk: CUfunction,
}

unsafe impl Send for BatchedGEMM {}
unsafe impl Sync for BatchedGEMM {}

impl Drop for BatchedGEMM {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = self.device.bind_to_thread();
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuModuleUnload(self.cu_module);
            }
            self.cu_module = std::ptr::null_mut();
        }
    }
}

impl BatchedGEMM {
    /// Compiles `gemm_quant.cu` via NVRTC and returns a new `BatchedGEMM`.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CudaError> {
        device.bind_to_thread()?;

        let ptx: Ptx = nvrtc::compile_ptx_with_opts(
            KERNEL_SRC,
            nvrtc::CompileOptions {
                arch: Some("compute_86"),
                use_fast_math: Some(true),
                options: vec!["--maxrregcount=48".to_string()],
                ..Default::default()
            },
        )
        .map_err(|e| CudaError::KernelCompile(format!("{e:?}")))?;

        let ptx_src = ptx.to_src();
        let ptx_c = CString::new(ptx_src.as_str()).map_err(|_| CudaError::KernelLoad("CString::new", CUresult::CUDA_ERROR_INVALID_VALUE))?;

        let mut cu_module: CUmodule = std::ptr::null_mut();
        let mut fn_quantize_q8_1: CUfunction = std::ptr::null_mut();
        let mut fn_q4k: CUfunction = std::ptr::null_mut();
        let mut fn_q4k_batched: CUfunction = std::ptr::null_mut();
        let mut fn_q6k: CUfunction = std::ptr::null_mut();
        let mut fn_q8: CUfunction = std::ptr::null_mut();
        let mut fn_f16: CUfunction = std::ptr::null_mut();
        let mut fn_f32: CUfunction = std::ptr::null_mut();
        let mut fn_q4k_fused_gate_up_swiglu: CUfunction = std::ptr::null_mut();
        let mut fn_q4k_batched_gate_up_swiglu: CUfunction = std::ptr::null_mut();
        let mut fn_q6k_batched: CUfunction = std::ptr::null_mut();
        let mut fn_q6k_splitk: CUfunction = std::ptr::null_mut();
        let mut fn_q4k_splitk: CUfunction = std::ptr::null_mut();
        let mut fn_fused_qkv: CUfunction = std::ptr::null_mut();
        let mut fn_fused_qkv_multi_row: CUfunction = std::ptr::null_mut();
        let mut fn_fused_qkv_q4k: CUfunction = std::ptr::null_mut();
        let mut fn_fused_qkv_batched: CUfunction = std::ptr::null_mut();
        let mut fn_get_rows_q4k: CUfunction = std::ptr::null_mut();
        let mut fn_get_rows_q6k: CUfunction = std::ptr::null_mut();
        let mut fn_get_rows_q8_0: CUfunction = std::ptr::null_mut();
        let mut fn_get_rows_f16: CUfunction = std::ptr::null_mut();
        let mut fn_sample_greedy: CUfunction = std::ptr::null_mut();
        let mut fn_advance_token: CUfunction = std::ptr::null_mut();

        let mut fn_q4k_mma: CUfunction = std::ptr::null_mut();
        let mut fn_q4k_multi_row: CUfunction = std::ptr::null_mut();
        let mut fn_q6k_multi_row: CUfunction = std::ptr::null_mut();
        let mut fn_fused_qkv_q4k_multi_row: CUfunction = std::ptr::null_mut();
        let mut fn_q4k_fused_gate_up_swiglu_multi_row: CUfunction = std::ptr::null_mut();
        let mut fn_q4k_fused_gate_up_swiglu_mma: CUfunction = std::ptr::null_mut();
        let mut fn_reduce_splitk: CUfunction = std::ptr::null_mut();

        unsafe {
            let lib = sys::lib();
            let res = lib.cuModuleLoadData(&mut cu_module, ptx_c.as_ptr() as *const std::ffi::c_void);
            if res != CUresult::CUDA_SUCCESS || cu_module.is_null() {
                return Err(CudaError::KernelLoad("cuModuleLoadData (gemm_quant)", res));
            }

            let c_quant = CString::new(FUNC_NAME_QUANTIZE_Q8_1).unwrap();
            let c_q4k = CString::new(FUNC_NAME_Q4K).unwrap();
            let c_q4k_b = CString::new(FUNC_NAME_Q4K_BATCHED).unwrap();
            let c_q6k = CString::new(FUNC_NAME_Q6K).unwrap();
            let c_q8  = CString::new(FUNC_NAME_Q8).unwrap();
            let c_f16 = CString::new(FUNC_NAME_F16).unwrap();
            let c_f32 = CString::new(FUNC_NAME_F32).unwrap();
            let c_fused = CString::new(FUNC_NAME_Q4K_FUSED_GATE_UP_SWIGLU).unwrap();
            let c_b_fused = CString::new(FUNC_NAME_Q4K_BATCHED_GATE_UP_SWIGLU).unwrap();
            let c_b_q6k = CString::new(FUNC_NAME_Q6K_BATCHED).unwrap();
            let c_splitk = CString::new(FUNC_NAME_Q6K_SPLITK).unwrap();
            let c_splitk_q4 = CString::new(FUNC_NAME_Q4K_SPLITK).unwrap();
            let c_fused_qkv = CString::new(FUNC_NAME_FUSED_QKV).unwrap();
            let c_fused_qkv_mr = CString::new(FUNC_NAME_FUSED_QKV_MULTI_ROW).unwrap();
            let c_fused_qkv_q4k = CString::new(FUNC_NAME_FUSED_QKV_Q4K).unwrap();
            let c_fused_qkv_b = CString::new(FUNC_NAME_FUSED_QKV_BATCHED).unwrap();
            let c_get_rows = CString::new(FUNC_NAME_GET_ROWS_Q4K).unwrap();
            let c_get_rows_q6k = CString::new(FUNC_NAME_GET_ROWS_Q6K).unwrap();
            let c_get_rows_q8 = CString::new(FUNC_NAME_GET_ROWS_Q8_0).unwrap();
            let c_get_rows_f16 = CString::new(FUNC_NAME_GET_ROWS_F16).unwrap();
            let c_sample = CString::new(FUNC_NAME_SAMPLE_GREEDY).unwrap();
            let c_advance = CString::new(FUNC_NAME_ADVANCE_TOKEN).unwrap();
            let c_q4k_mma = CString::new(FUNC_NAME_Q4K_MMA).unwrap();
            let c_q4k_mr = CString::new(FUNC_NAME_Q4K_MULTI_ROW).unwrap();
            let c_q6k_mr = CString::new(FUNC_NAME_Q6K_MULTI_ROW).unwrap();
            let c_qkv_mr = CString::new(FUNC_NAME_FUSED_QKV_Q4K_MULTI_ROW).unwrap();
            let c_swiglu_mr = CString::new(FUNC_NAME_Q4K_FUSED_GATE_UP_SWIGLU_MULTI_ROW).unwrap();
            let c_fused_mma = CString::new(FUNC_NAME_Q4K_FUSED_GATE_UP_SWIGLU_MMA).unwrap();

            let r_q = lib.cuModuleGetFunction(&mut fn_quantize_q8_1, cu_module, c_quant.as_ptr());
            let r1 = lib.cuModuleGetFunction(&mut fn_q4k, cu_module, c_q4k.as_ptr());
            let r1b = lib.cuModuleGetFunction(&mut fn_q4k_batched, cu_module, c_q4k_b.as_ptr());
            let _ = lib.cuModuleGetFunction(&mut fn_q4k_multi_row, cu_module, c_q4k_mr.as_ptr());
            let _ = lib.cuModuleGetFunction(&mut fn_q6k_multi_row, cu_module, c_q6k_mr.as_ptr());
            let _ = lib.cuModuleGetFunction(&mut fn_fused_qkv_q4k_multi_row, cu_module, c_qkv_mr.as_ptr());
            let _ = lib.cuModuleGetFunction(&mut fn_fused_qkv_multi_row, cu_module, c_fused_qkv_mr.as_ptr());
            let _ = lib.cuModuleGetFunction(&mut fn_q4k_fused_gate_up_swiglu_multi_row, cu_module, c_swiglu_mr.as_ptr());
            let r2 = lib.cuModuleGetFunction(&mut fn_q6k, cu_module, c_q6k.as_ptr());
            let r3 = lib.cuModuleGetFunction(&mut fn_q8,  cu_module, c_q8.as_ptr());
            let r4 = lib.cuModuleGetFunction(&mut fn_f16, cu_module, c_f16.as_ptr());
            let r5 = lib.cuModuleGetFunction(&mut fn_f32, cu_module, c_f32.as_ptr());
            let r6 = lib.cuModuleGetFunction(&mut fn_q4k_fused_gate_up_swiglu, cu_module, c_fused.as_ptr());
            let r6b = lib.cuModuleGetFunction(&mut fn_q4k_batched_gate_up_swiglu, cu_module, c_b_fused.as_ptr());
            let r6c = lib.cuModuleGetFunction(&mut fn_q6k_batched, cu_module, c_b_q6k.as_ptr());
            let r7 = lib.cuModuleGetFunction(&mut fn_q6k_splitk, cu_module, c_splitk.as_ptr());
            let r8 = lib.cuModuleGetFunction(&mut fn_q4k_splitk, cu_module, c_splitk_q4.as_ptr());
            let r9 = lib.cuModuleGetFunction(&mut fn_fused_qkv, cu_module, c_fused_qkv.as_ptr());
            let r9_q4 = lib.cuModuleGetFunction(&mut fn_fused_qkv_q4k, cu_module, c_fused_qkv_q4k.as_ptr());
            let r9b = lib.cuModuleGetFunction(&mut fn_fused_qkv_batched, cu_module, c_fused_qkv_b.as_ptr());
            let r10 = lib.cuModuleGetFunction(&mut fn_get_rows_q4k, cu_module, c_get_rows.as_ptr());
            let _ = lib.cuModuleGetFunction(&mut fn_get_rows_q6k, cu_module, c_get_rows_q6k.as_ptr());
            let _ = lib.cuModuleGetFunction(&mut fn_get_rows_q8_0, cu_module, c_get_rows_q8.as_ptr());
            let _ = lib.cuModuleGetFunction(&mut fn_get_rows_f16, cu_module, c_get_rows_f16.as_ptr());
            let r11 = lib.cuModuleGetFunction(&mut fn_sample_greedy, cu_module, c_sample.as_ptr());
            let r12 = lib.cuModuleGetFunction(&mut fn_advance_token, cu_module, c_advance.as_ptr());
            let _ = lib.cuModuleGetFunction(&mut fn_q4k_mma, cu_module, c_q4k_mma.as_ptr());
            let _ = lib.cuModuleGetFunction(&mut fn_q4k_fused_gate_up_swiglu_mma, cu_module, c_fused_mma.as_ptr());

            let c_reduce_splitk = CString::new(FUNC_NAME_REDUCE_SPLITK).unwrap();
            let _ = lib.cuModuleGetFunction(&mut fn_reduce_splitk, cu_module, c_reduce_splitk.as_ptr());

            if r_q != CUresult::CUDA_SUCCESS
                || r1 != CUresult::CUDA_SUCCESS
                || r1b != CUresult::CUDA_SUCCESS
                || r2 != CUresult::CUDA_SUCCESS
                || r3 != CUresult::CUDA_SUCCESS
                || r4 != CUresult::CUDA_SUCCESS
                || r5 != CUresult::CUDA_SUCCESS
                || r6 != CUresult::CUDA_SUCCESS
                || r6b != CUresult::CUDA_SUCCESS
                || r6c != CUresult::CUDA_SUCCESS
                || r7 != CUresult::CUDA_SUCCESS
                || r8 != CUresult::CUDA_SUCCESS
                || r9 != CUresult::CUDA_SUCCESS
                || r9_q4 != CUresult::CUDA_SUCCESS
                || r9b != CUresult::CUDA_SUCCESS
                || r10 != CUresult::CUDA_SUCCESS
                || r11 != CUresult::CUDA_SUCCESS
                || r12 != CUresult::CUDA_SUCCESS
            {
                let _ = lib.cuModuleUnload(cu_module);
                return Err(CudaError::KernelLoad("cuModuleGetFunction", r1));
            }
        }

        Ok(Self {
            device,
            cu_module,
            fn_quantize_q8_1,
            fn_q4k,
            fn_q4k_batched,
            fn_q4k_multi_row,
            fn_q6k,
            fn_q6k_multi_row,
            fn_q8,
            fn_f16,
            _fn_f32: fn_f32,
            fn_q4k_fused_gate_up_swiglu,
            fn_q4k_batched_gate_up_swiglu,
            fn_q4k_fused_gate_up_swiglu_multi_row,
            fn_q6k_batched,
            fn_q6k_splitk,
            fn_q4k_splitk,
            fn_fused_qkv,
            fn_fused_qkv_multi_row,
            fn_fused_qkv_q4k,
            fn_fused_qkv_q4k_multi_row,
            fn_fused_qkv_batched,
            fn_get_rows_q4k,
            fn_get_rows_q6k,
            fn_get_rows_q8_0,
            fn_get_rows_f16,
            fn_sample_greedy,
            fn_advance_token,
            fn_q4k_mma,
            fn_q4k_fused_gate_up_swiglu_mma,
            fn_reduce_splitk,
        })
    }

    /// Quantizes `x` (floats) on-the-fly into `Q8_1` (int8_t + float scales + float sums) in a single fast 1-block kernel.
    pub fn quantize_q8_1_batched(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer,
        norm_weight: Option<&DeviceBuffer>,
        out_qx: &DeviceBuffer,
        out_qd: &DeviceBuffer,
        out_qs: &DeviceBuffer,
        ne0: usize,
        batch_size: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        if batch_size == 0 || ne0 == 0 {
            return Ok(());
        }
        let x_addr: u64 = x.device_ptr();
        let norm_addr: u64 = norm_weight.map(|n| n.device_ptr()).unwrap_or(0);
        let qx_addr: u64 = out_qx.device_ptr();
        let qd_addr: u64 = out_qd.device_ptr();
        let qs_addr: u64 = out_qs.device_ptr();
        let ne0_i: i32 = ne0 as i32;
        let eps_f: f32 = eps;

        let args: [*mut std::ffi::c_void; 7] = [
            &x_addr as *const u64 as *mut std::ffi::c_void,
            &norm_addr as *const u64 as *mut std::ffi::c_void,
            &qx_addr as *const u64 as *mut std::ffi::c_void,
            &qd_addr as *const u64 as *mut std::ffi::c_void,
            &qs_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &eps_f as *const f32 as *mut std::ffi::c_void,
        ];

        let num_blocks_32 = (ne0 / 32) as u32;
        let grid_x = if norm_weight.is_some() { 1 } else { num_blocks_32.div_ceil(8) };
        let grid_y = batch_size as u32;

        self.device.bind_to_thread()?;
        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_quantize_q8_1,
                grid_x,
                grid_y,
                1,
                256,
                1,
                1,
                0,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (quantize_q8_1)", res));
            }
        }
        Ok(())
    }

    /// Performs on-the-fly Q8_1 activation quantization of row `x[ne0]` with optional fused RMSNorm:
    /// `qx[ne0] = round(RMSNorm(x)[ne0] * 127 / amax)`, `qd[ne0/32] = amax / 127`, `qs[ne0/32] = sum(RMSNorm(x))`.
    pub fn quantize_q8_1(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer,
        norm_weight: Option<&DeviceBuffer>,
        out_qx: &DeviceBuffer,
        out_qd: &DeviceBuffer,
        out_qs: &DeviceBuffer,
        ne0: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        self.quantize_q8_1_batched(stream, x, norm_weight, out_qx, out_qd, out_qs, ne0, 1, eps)
    }

    /// Evaluates quantized matrix multiplication with fused in-kernel activation quantization:
    /// `Out[M, ne1] = RMSNorm(X)[M, ne0] * W[ne1, ne0]^T + Residual[M, ne1]`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q8_act_with_residual(
        &self,
        stream: &CudaStream,
        weights: &DeviceBuffer,
        qx: &DeviceBuffer,
        qd: &DeviceBuffer,
        qs: &DeviceBuffer,
        out: &DeviceBuffer,
        ne0: usize,
        ne1: usize,
        batch_size: usize,
        format: GemvFormat,
        residual: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        match format {
            GemvFormat::Q4K => {
                if !self.fn_q4k_mma.is_null() {
                    self.gemm_q4k_mma(stream, weights, qx, qd, qs, out, ne0, ne1, batch_size, residual)
                } else {
                    self.gemm_q4k(stream, weights, qx, qd, qs, out, ne0, ne1, batch_size, residual)
                }
            }
            GemvFormat::Q6K => self.gemm_q6k(stream, weights, qx, qd, qs, out, ne0, ne1, batch_size, residual),
            _ => self.gemm_q4k(stream, weights, qx, qd, qs, out, ne0, ne1, batch_size, residual),
        }
    }

    /// Evaluates Q4_K matrix multiplication with pre-quantized activations in shared memory:
    /// `Out[M, ne1] = Q8(X)[M, ne0] * W[ne1, ne0]^T + Residual[M, ne1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4k(
        &self,
        stream: &CudaStream,
        weights: &DeviceBuffer,
        qx: &DeviceBuffer,
        qd: &DeviceBuffer,
        qs: &DeviceBuffer,
        out: &DeviceBuffer,
        ne0: usize,
        ne1: usize,
        batch_size: usize,
        residual: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        if batch_size == 0 || ne0 == 0 || ne1 == 0 {
            return Ok(());
        }

        let shared_bytes = (((ne0 * 5) / 4) + 128) as u32;
        let ne0_i: i32 = ne0 as i32;
        let ne1_i: i32 = ne1 as i32;
        let batch_i: i32 = batch_size as i32;

        let w_addr: u64 = weights.device_ptr();
        let qx_addr: u64 = qx.device_ptr();
        let qd_addr: u64 = qd.device_ptr();
        let qs_addr: u64 = qs.device_ptr();
        let out_addr: u64 = out.device_ptr();
        let res_addr: u64 = residual.map(|r| r.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 9] = [
            &w_addr as *const u64 as *mut std::ffi::c_void,
            &qx_addr as *const u64 as *mut std::ffi::c_void,
            &qd_addr as *const u64 as *mut std::ffi::c_void,
            &qs_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &ne1_i as *const i32 as *mut std::ffi::c_void,
            &batch_i as *const i32 as *mut std::ffi::c_void,
            &res_addr as *const u64 as *mut std::ffi::c_void,
        ];

        let cols_per_block = 8u32;
        let grid_x = (ne1 as u32).div_ceil(cols_per_block);

        self.device.bind_to_thread()?;

        if batch_size > 1 && batch_size <= 4 && !self.fn_q4k_multi_row.is_null() {
            let shared_bytes = (((ne0 * 5) / 4) * batch_size + 256) as u32;
            let grid_y = 1u32;
            unsafe {
                let lib = sys::lib();
                let res = lib.cuLaunchKernel(
                    self.fn_q4k_multi_row,
                    grid_x,
                    grid_y,
                    1,
                    256,
                    1,
                    1,
                    shared_bytes,
                    stream.raw(),
                    args.as_ptr() as *mut *mut std::ffi::c_void,
                    std::ptr::null_mut(),
                );
                if res != CUresult::CUDA_SUCCESS {
                    return Err(CudaError::KernelLaunch("cuLaunchKernel (GEMM Q4_K Multi-Row)", res));
                }
            }
            return Ok(());
        }

        let grid_y = batch_size as u32;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_q4k,
                grid_x,
                grid_y,
                1,
                256,
                1,
                1,
                shared_bytes,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (GEMM Q4_K)", res));
            }
        }

        Ok(())
    }

    /// Evaluates Q4_K matrix multiplication using Tensor Cores (PTX mma.sync):
    /// `Out[M, ne1] = Q8(X)[M, ne0] * W[ne1, ne0]^T + Residual[M, ne1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4k_mma(
        &self,
        stream: &CudaStream,
        weights: &DeviceBuffer,
        qx: &DeviceBuffer,
        qd: &DeviceBuffer,
        qs: &DeviceBuffer,
        out: &DeviceBuffer,
        ne0: usize,
        ne1: usize,
        batch_size: usize,
        residual: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        if batch_size == 0 || ne0 == 0 || ne1 == 0 {
            return Ok(());
        }

        if self.fn_q4k_mma.is_null() {
            return self.gemm_q4k(stream, weights, qx, qd, qs, out, ne0, ne1, batch_size, residual);
        }

        let ne0_i: i32 = ne0 as i32;
        let ne1_i: i32 = ne1 as i32;
        let batch_i: i32 = batch_size as i32;

        let w_addr: u64 = weights.device_ptr();
        let qx_addr: u64 = qx.device_ptr();
        let qd_addr: u64 = qd.device_ptr();
        let qs_addr: u64 = qs.device_ptr();
        let out_addr: u64 = out.device_ptr();
        let res_addr: u64 = residual.map(|r| r.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 9] = [
            &w_addr as *const u64 as *mut std::ffi::c_void,
            &qx_addr as *const u64 as *mut std::ffi::c_void,
            &qd_addr as *const u64 as *mut std::ffi::c_void,
            &qs_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &ne1_i as *const i32 as *mut std::ffi::c_void,
            &batch_i as *const i32 as *mut std::ffi::c_void,
            &res_addr as *const u64 as *mut std::ffi::c_void,
        ];

        let cols_per_block = 4u32;
        let grid_x = (ne1 as u32).div_ceil(cols_per_block);
        let grid_y = batch_size as u32;
        let shared_bytes = (((ne0 * 5) / 4) + 256) as u32;

        self.device.bind_to_thread()?;

        if batch_size > 1 && batch_size <= 4 && !self.fn_q4k_multi_row.is_null() {
            let cols_mr = 8u32;
            let grid_x_mr = (ne1 as u32).div_ceil(cols_mr);
            let shared_bytes_mr = (((ne0 * 5) / 4) * batch_size + 256) as u32;
            unsafe {
                let lib = sys::lib();
                let res = lib.cuLaunchKernel(
                    self.fn_q4k_multi_row,
                    grid_x_mr,
                    1,
                    1,
                    256,
                    1,
                    1,
                    shared_bytes_mr,
                    stream.raw(),
                    args.as_ptr() as *mut *mut std::ffi::c_void,
                    std::ptr::null_mut(),
                );
                if res != CUresult::CUDA_SUCCESS {
                    return Err(CudaError::KernelLaunch("cuLaunchKernel (GEMM Q4_K Multi-Row)", res));
                }
            }
            return Ok(());
        }

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_q4k_mma,
                grid_x,
                grid_y,
                1,
                256,
                1,
                1,
                shared_bytes,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (GEMM Q4_K MMA)", res));
            }
        }

        Ok(())
    }

    /// Evaluates Split-K Q4_K matrix multiplication:
    /// Partitions reduction dimension `K` into `split_k` slices across threadblocks,
    /// writing partial results to `scratch_out` and then executing a fast reduction pass.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4k_splitk(
        &self,
        stream: &CudaStream,
        weights: &DeviceBuffer,
        qx: &DeviceBuffer,
        qd: &DeviceBuffer,
        qs: &DeviceBuffer,
        scratch_out: &DeviceBuffer,
        out: &DeviceBuffer,
        ne0: usize,
        ne1: usize,
        batch_size: usize,
        split_k: usize,
        residual: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        if batch_size == 0 || ne0 == 0 || ne1 == 0 {
            return Ok(());
        }

        if split_k <= 1 || self.fn_q4k_splitk.is_null() || self.fn_reduce_splitk.is_null() {
            return self.gemm_q4k_mma(stream, weights, qx, qd, qs, out, ne0, ne1, batch_size, residual);
        }

        let ne0_i: i32 = ne0 as i32;
        let ne1_i: i32 = ne1 as i32;
        let batch_i: i32 = batch_size as i32;
        let split_k_i: i32 = split_k as i32;

        let w_addr: u64 = weights.device_ptr();
        let qx_addr: u64 = qx.device_ptr();
        let qd_addr: u64 = qd.device_ptr();
        let qs_addr: u64 = qs.device_ptr();
        let scratch_addr: u64 = scratch_out.device_ptr();

        let args: [*mut std::ffi::c_void; 9] = [
            &w_addr as *const u64 as *mut std::ffi::c_void,
            &qx_addr as *const u64 as *mut std::ffi::c_void,
            &qd_addr as *const u64 as *mut std::ffi::c_void,
            &qs_addr as *const u64 as *mut std::ffi::c_void,
            &scratch_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &ne1_i as *const i32 as *mut std::ffi::c_void,
            &batch_i as *const i32 as *mut std::ffi::c_void,
            &split_k_i as *const i32 as *mut std::ffi::c_void,
        ];

        let cols_per_block = 8u32;
        let grid_x = (ne1 as u32).div_ceil(cols_per_block);
        let grid_y = batch_size as u32;
        let grid_z = split_k as u32;
        let shared_bytes = (((ne0 * 5) / 4) + 128) as u32;

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_q4k_splitk,
                grid_x,
                grid_y,
                grid_z,
                256,
                1,
                1,
                shared_bytes,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (GEMM Q4_K Split-K)", res));
            }
        }

        // Reduction Pass
        let total_elems = (batch_size * ne1) as u32;
        let block_sz = 256u32;
        let grid_reduce = total_elems.div_ceil(block_sz);

        let out_addr: u64 = out.device_ptr();
        let res_addr: u64 = residual.map(|r| r.device_ptr()).unwrap_or(0);
        let size_i: i32 = (batch_size * ne1) as i32;

        let reduce_args: [*mut std::ffi::c_void; 5] = [
            &scratch_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &res_addr as *const u64 as *mut std::ffi::c_void,
            &size_i as *const i32 as *mut std::ffi::c_void,
            &split_k_i as *const i32 as *mut std::ffi::c_void,
        ];

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_reduce_splitk,
                grid_reduce,
                1,
                1,
                block_sz,
                1,
                1,
                0,
                stream.raw(),
                reduce_args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (Reduce Split-K)", res));
            }
        }

        Ok(())
    }

    /// Evaluates Fused Gate + Up SwiGLU projection using Tensor Cores (PTX mma.sync):
    /// `Out[M, ne1] = silu(Q8(X) * Wgate^T) * (Q8(X) * Wup^T)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4k_fused_gate_up_swiglu_mma(
        &self,
        stream: &CudaStream,
        wgate: &DeviceBuffer,
        wup: &DeviceBuffer,
        qx: &DeviceBuffer,
        qd: &DeviceBuffer,
        qs: &DeviceBuffer,
        out: &DeviceBuffer,
        ne0: usize,
        ne1: usize,
        batch_size: usize,
    ) -> Result<(), CudaError> {
        if batch_size == 0 || ne0 == 0 || ne1 == 0 {
            return Ok(());
        }

        if self.fn_q4k_fused_gate_up_swiglu_mma.is_null() {
            return self.gemm_fused_gate_up_swiglu(stream, wgate, wup, qx, qd, qs, out, ne0, ne1, batch_size);
        }

        let ne0_i: i32 = ne0 as i32;
        let ne1_i: i32 = ne1 as i32;
        let batch_i: i32 = batch_size as i32;

        let wgate_addr: u64 = wgate.device_ptr();
        let wup_addr: u64 = wup.device_ptr();
        let qx_addr: u64 = qx.device_ptr();
        let qd_addr: u64 = qd.device_ptr();
        let qs_addr: u64 = qs.device_ptr();
        let out_addr: u64 = out.device_ptr();

        let args: [*mut std::ffi::c_void; 9] = [
            &wgate_addr as *const u64 as *mut std::ffi::c_void,
            &wup_addr as *const u64 as *mut std::ffi::c_void,
            &qx_addr as *const u64 as *mut std::ffi::c_void,
            &qd_addr as *const u64 as *mut std::ffi::c_void,
            &qs_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &ne1_i as *const i32 as *mut std::ffi::c_void,
            &batch_i as *const i32 as *mut std::ffi::c_void,
        ];

        let cols_per_block = 4u32;
        let grid_x = (ne1 as u32).div_ceil(cols_per_block);

        self.device.bind_to_thread()?;

        if batch_size > 1 && batch_size <= 4 && !self.fn_q4k_fused_gate_up_swiglu_multi_row.is_null() {
            let shared_bytes = (((ne0 * 5) / 4) * batch_size + 256) as u32;
            let cols_mr = 8u32;
            let grid_x_mr = (ne1 as u32).div_ceil(cols_mr);
            let grid_y_mr = 1u32;
            unsafe {
                let lib = sys::lib();
                let res = lib.cuLaunchKernel(
                    self.fn_q4k_fused_gate_up_swiglu_multi_row,
                    grid_x_mr,
                    grid_y_mr,
                    1,
                    256,
                    1,
                    1,
                    shared_bytes,
                    stream.raw(),
                    args.as_ptr() as *mut *mut std::ffi::c_void,
                    std::ptr::null_mut(),
                );
                if res != CUresult::CUDA_SUCCESS {
                    return Err(CudaError::KernelLaunch("cuLaunchKernel (FusedGateUpSwiGLU Multi-Row)", res));
                }
            }
            return Ok(());
        }

        let grid_y = batch_size as u32;
        let shared_bytes = (((ne0 * 5) / 4) + 256) as u32;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_q4k_fused_gate_up_swiglu_mma,
                grid_x,
                grid_y,
                1,
                256,
                1,
                1,
                shared_bytes,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (FusedGateUpSwiGLU MMA)", res));
            }
        }

        Ok(())
    }

    /// Evaluates quantized matrix multiplication:
    /// `Out[M, ne1] = X[M, ne0] * W[ne1, ne0]^T`.
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
        self.gemm_with_residual(stream, weights, x, out, ne0, ne1, batch_size, format, None)
    }

    /// Evaluates quantized matrix multiplication with fused in-place residual addition:
    /// `Out[M, ne1] = X[M, ne0] * W[ne1, ne0]^T + Residual[M, ne1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_with_residual(
        &self,
        stream: &CudaStream,
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        out: &DeviceBuffer,
        ne0: usize,
        ne1: usize,
        batch_size: usize,
        format: GemvFormat,
        residual: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        if batch_size == 0 || ne0 == 0 || ne1 == 0 {
            return Ok(());
        }

        let expected_weight_bytes = match format {
            GemvFormat::Q4K => (ne0 / 256) * 144 * ne1,
            GemvFormat::Q6K => (ne0 / 256) * 210 * ne1,
            GemvFormat::Q8  => (ne0 / 32)  * 34  * ne1,
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

        if let Some(r) = residual {
            if r.size() < expected_out_bytes {
                return Err(CudaError::InvalidSize {
                    expected: expected_out_bytes,
                    actual: r.size(),
                });
            }
        }

        let (func, block_dim_y) = match format {
            GemvFormat::Q4K => (self.fn_q4k_batched, 8u32),
            GemvFormat::Q6K => (self.fn_q6k_batched, 8u32),
            _ => (self.fn_q4k_batched, 8u32),
        };

        let ne0_i: i32 = ne0 as i32;
        let ne1_i: i32 = ne1 as i32;
        let batch_i: i32 = batch_size as i32;

        let w_addr: u64 = weights.device_ptr();
        let x_addr: u64 = x.device_ptr();
        let out_addr: u64 = out.device_ptr();
        let res_addr: u64 = residual.map(|r| r.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 7] = [
            &w_addr as *const u64 as *mut std::ffi::c_void,
            &x_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &ne1_i as *const i32 as *mut std::ffi::c_void,
            &batch_i as *const i32 as *mut std::ffi::c_void,
            &res_addr as *const u64 as *mut std::ffi::c_void,
        ];

        let cols_per_block = 8u32;
        let grid_x = (ne1 as u32).div_ceil(cols_per_block);
        let grid_y = (batch_size as u32).div_ceil(block_dim_y);

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                func,
                grid_x,
                grid_y,
                1,
                128,
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

    /// Fused Q6_K LM Head projection with pre-quantized Q8_1 activations:
    /// Computes `Out[M, ne1] = Q8(X)[M, ne0] * W[ne1, ne0]^T`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q6k(
        &self,
        stream: &CudaStream,
        weights: &DeviceBuffer,
        qx: &DeviceBuffer,
        qd: &DeviceBuffer,
        qs: &DeviceBuffer,
        out: &DeviceBuffer,
        ne0: usize,
        ne1: usize,
        batch_size: usize,
        residual: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        let ne0_i: i32 = ne0 as i32;
        let ne1_i: i32 = ne1 as i32;
        let batch_i: i32 = batch_size as i32;

        let w_addr: u64 = weights.device_ptr();
        let qx_addr: u64 = qx.device_ptr();
        let qd_addr: u64 = qd.device_ptr();
        let qs_addr: u64 = qs.device_ptr();
        let out_addr: u64 = out.device_ptr();
        let res_addr: u64 = residual.map(|r| r.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 9] = [
            &w_addr as *const u64 as *mut std::ffi::c_void,
            &qx_addr as *const u64 as *mut std::ffi::c_void,
            &qd_addr as *const u64 as *mut std::ffi::c_void,
            &qs_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &ne1_i as *const i32 as *mut std::ffi::c_void,
            &batch_i as *const i32 as *mut std::ffi::c_void,
            &res_addr as *const u64 as *mut std::ffi::c_void,
        ];

        let cols_per_block = 8u32;
        let shared_bytes = (((ne0 * 5) / 4) + 128) as u32;
        let grid_x = (ne1 as u32).div_ceil(cols_per_block);

        self.device.bind_to_thread()?;

        if batch_size > 1 && batch_size <= 4 && !self.fn_q6k_multi_row.is_null() {
            let shared_bytes_mr = (((ne0 * 5) / 4) * batch_size + 256) as u32;
            let grid_y_mr = 1u32;
            unsafe {
                let lib = sys::lib();
                let res = lib.cuLaunchKernel(
                    self.fn_q6k_multi_row,
                    grid_x,
                    grid_y_mr,
                    1,
                    256,
                    1,
                    1,
                    shared_bytes_mr,
                    stream.raw(),
                    args.as_ptr() as *mut *mut std::ffi::c_void,
                    std::ptr::null_mut(),
                );
                if res != CUresult::CUDA_SUCCESS {
                    return Err(CudaError::KernelLaunch("cuLaunchKernel (GEMM Q6_K Multi-Row)", res));
                }
            }
            return Ok(());
        }

        let grid_y = batch_size as u32;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_q6k,
                grid_x,
                grid_y,
                1,
                256,
                1,
                1,
                shared_bytes,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (GEMM Q6_K)", res));
            }
        }

        Ok(())
    }

    /// Fused Q4_K Gate + Up projection with in-kernel SwiGLU activation and pre-quantized Q8_1 activations:
    /// Computes `Out[M, ne1] = silu(Q8(X)[M, ne0] * Wgate^T) * (Q8(X)[M, ne0] * Wup^T)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fused_gate_up_swiglu(
        &self,
        stream: &CudaStream,
        wgate: &DeviceBuffer,
        wup: &DeviceBuffer,
        qx: &DeviceBuffer,
        qd: &DeviceBuffer,
        qs: &DeviceBuffer,
        out: &DeviceBuffer,
        ne0: usize,
        ne1: usize,
        batch_size: usize,
    ) -> Result<(), CudaError> {
        if batch_size == 0 || ne0 == 0 || ne1 == 0 {
            return Ok(());
        }

        let ne0_i: i32 = ne0 as i32;
        let ne1_i: i32 = ne1 as i32;
        let batch_i: i32 = batch_size as i32;

        let wgate_addr: u64 = wgate.device_ptr();
        let wup_addr: u64 = wup.device_ptr();
        let qx_addr: u64 = qx.device_ptr();
        let qd_addr: u64 = qd.device_ptr();
        let qs_addr: u64 = qs.device_ptr();
        let out_addr: u64 = out.device_ptr();

        let args: [*mut std::ffi::c_void; 9] = [
            &wgate_addr as *const u64 as *mut std::ffi::c_void,
            &wup_addr as *const u64 as *mut std::ffi::c_void,
            &qx_addr as *const u64 as *mut std::ffi::c_void,
            &qd_addr as *const u64 as *mut std::ffi::c_void,
            &qs_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &ne1_i as *const i32 as *mut std::ffi::c_void,
            &batch_i as *const i32 as *mut std::ffi::c_void,
        ];

        let cols_per_block = 8u32;
        let grid_x = (ne1 as u32).div_ceil(cols_per_block);
        let grid_y = batch_size as u32;
        let shared_bytes = (((ne0 * 5) / 4) + 128) as u32;

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_q4k_fused_gate_up_swiglu,
                grid_x,
                grid_y,
                1,
                256,
                1,
                1,
                shared_bytes,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (FusedGateUpSwiGLU)", res));
            }
        }

        Ok(())
    }

    /// Launches Split-K (4-way K-slice parallel across 4 warps per col) Q4_K matrix multiplication with fused in-place residual addition:
    /// `Out[M, ne1] = Q8(X)[M, ne0] * W[ne1, ne0]^T + Residual[M, ne1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4k_splitk_with_residual(
        &self,
        stream: &CudaStream,
        weights: &DeviceBuffer,
        qx: &DeviceBuffer,
        qd: &DeviceBuffer,
        qs: &DeviceBuffer,
        out: &DeviceBuffer,
        ne0: usize,
        ne1: usize,
        batch_size: usize,
        residual: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        if batch_size == 0 || ne0 == 0 || ne1 == 0 {
            return Ok(());
        }

        let ne0_i: i32 = ne0 as i32;
        let ne1_i: i32 = ne1 as i32;
        let batch_i: i32 = batch_size as i32;

        let w_addr: u64 = weights.device_ptr();
        let qx_addr: u64 = qx.device_ptr();
        let qd_addr: u64 = qd.device_ptr();
        let qs_addr: u64 = qs.device_ptr();
        let out_addr: u64 = out.device_ptr();
        let res_addr: u64 = residual.map(|r| r.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 9] = [
            &w_addr as *const u64 as *mut std::ffi::c_void,
            &qx_addr as *const u64 as *mut std::ffi::c_void,
            &qd_addr as *const u64 as *mut std::ffi::c_void,
            &qs_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &ne1_i as *const i32 as *mut std::ffi::c_void,
            &batch_i as *const i32 as *mut std::ffi::c_void,
            &res_addr as *const u64 as *mut std::ffi::c_void,
        ];

        let grid_x = ne1 as u32;
        let grid_y = batch_size as u32;
        let shared_bytes = (((ne0 * 5) / 4) + 160) as u32;

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_q4k_splitk,
                grid_x,
                grid_y,
                1,
                256,
                1,
                1,
                shared_bytes,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (gemm_q4k_splitk)", res));
            }
        }

        Ok(())
    }

    /// Launches fused QKV projection: projects pre-quantized $X$ into $Q, K, V$ simultaneously in 1 kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fused_qkv(
        &self,
        stream: &CudaStream,
        wq: &DeviceBuffer,
        wk: &DeviceBuffer,
        wv: &DeviceBuffer,
        qx: &DeviceBuffer,
        qd: &DeviceBuffer,
        qs: &DeviceBuffer,
        q_out: &DeviceBuffer,
        k_out: &DeviceBuffer,
        v_out: &DeviceBuffer,
        ne0: usize,
        qdim: usize,
        kvd: usize,
        batch_size: usize,
        qb: Option<&DeviceBuffer>,
        kb: Option<&DeviceBuffer>,
        vb: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        if batch_size == 0 || ne0 == 0 || qdim == 0 || kvd == 0 {
            return Ok(());
        }

        let ne0_i: i32 = ne0 as i32;
        let qdim_i: i32 = qdim as i32;
        let kvd_i: i32 = kvd as i32;
        let batch_i: i32 = batch_size as i32;

        let wq_addr: u64 = wq.device_ptr();
        let wk_addr: u64 = wk.device_ptr();
        let wv_addr: u64 = wv.device_ptr();
        let qx_addr: u64 = qx.device_ptr();
        let qd_addr: u64 = qd.device_ptr();
        let qs_addr: u64 = qs.device_ptr();
        let q_addr: u64 = q_out.device_ptr();
        let k_addr: u64 = k_out.device_ptr();
        let v_addr: u64 = v_out.device_ptr();
        let qb_addr: u64 = qb.map(|b| b.device_ptr()).unwrap_or(0);
        let kb_addr: u64 = kb.map(|b| b.device_ptr()).unwrap_or(0);
        let vb_addr: u64 = vb.map(|b| b.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 16] = [
            &wq_addr as *const u64 as *mut std::ffi::c_void,
            &wk_addr as *const u64 as *mut std::ffi::c_void,
            &wv_addr as *const u64 as *mut std::ffi::c_void,
            &qx_addr as *const u64 as *mut std::ffi::c_void,
            &qd_addr as *const u64 as *mut std::ffi::c_void,
            &qs_addr as *const u64 as *mut std::ffi::c_void,
            &q_addr as *const u64 as *mut std::ffi::c_void,
            &k_addr as *const u64 as *mut std::ffi::c_void,
            &v_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &qdim_i as *const i32 as *mut std::ffi::c_void,
            &kvd_i as *const i32 as *mut std::ffi::c_void,
            &batch_i as *const i32 as *mut std::ffi::c_void,
            &qb_addr as *const u64 as *mut std::ffi::c_void,
            &kb_addr as *const u64 as *mut std::ffi::c_void,
            &vb_addr as *const u64 as *mut std::ffi::c_void,
        ];

        let cols_per_block = 8u32;
        let total_cols = (qdim + kvd + kvd) as u32;
        let grid_x = total_cols.div_ceil(cols_per_block);

        self.device.bind_to_thread()?;

        if batch_size > 1 && batch_size <= 4 && !self.fn_fused_qkv_multi_row.is_null() {
            let shared_bytes_mr = (((ne0 * 5) / 4) * batch_size + 256) as u32;
            let grid_y_mr = 1u32;
            unsafe {
                let lib = sys::lib();
                let res = lib.cuLaunchKernel(
                    self.fn_fused_qkv_multi_row,
                    grid_x,
                    grid_y_mr,
                    1,
                    256,
                    1,
                    1,
                    shared_bytes_mr,
                    stream.raw(),
                    args.as_ptr() as *mut *mut std::ffi::c_void,
                    std::ptr::null_mut(),
                );
                if res != CUresult::CUDA_SUCCESS {
                    return Err(CudaError::KernelLaunch("cuLaunchKernel (FusedQKV Multi-Row)", res));
                }
            }
            return Ok(());
        }

        let grid_y = batch_size as u32;
        let shared_bytes = (((ne0 * 5) / 4) + 128) as u32;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_fused_qkv,
                grid_x,
                grid_y,
                1,
                256,
                1,
                1,
                shared_bytes,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (FusedQKV)", res));
            }
        }

        Ok(())
    }

    /// Launches fused QKV projection where Wq, Wk, Wv are all Q4_K.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fused_qkv_q4k(
        &self,
        stream: &CudaStream,
        wq: &DeviceBuffer,
        wk: &DeviceBuffer,
        wv: &DeviceBuffer,
        qx: &DeviceBuffer,
        qd: &DeviceBuffer,
        qs: &DeviceBuffer,
        q_out: &DeviceBuffer,
        k_out: &DeviceBuffer,
        v_out: &DeviceBuffer,
        ne0: usize,
        qdim: usize,
        kvd: usize,
        batch_size: usize,
        qb: Option<&DeviceBuffer>,
        kb: Option<&DeviceBuffer>,
        vb: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        if batch_size == 0 || ne0 == 0 || qdim == 0 || kvd == 0 {
            return Ok(());
        }

        let ne0_i: i32 = ne0 as i32;
        let qdim_i: i32 = qdim as i32;
        let kvd_i: i32 = kvd as i32;
        let batch_i: i32 = batch_size as i32;

        let wq_addr: u64 = wq.device_ptr();
        let wk_addr: u64 = wk.device_ptr();
        let wv_addr: u64 = wv.device_ptr();
        let qx_addr: u64 = qx.device_ptr();
        let qd_addr: u64 = qd.device_ptr();
        let qs_addr: u64 = qs.device_ptr();
        let q_addr: u64 = q_out.device_ptr();
        let k_addr: u64 = k_out.device_ptr();
        let v_addr: u64 = v_out.device_ptr();
        let qb_addr: u64 = qb.map(|b| b.device_ptr()).unwrap_or(0);
        let kb_addr: u64 = kb.map(|b| b.device_ptr()).unwrap_or(0);
        let vb_addr: u64 = vb.map(|b| b.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 16] = [
            &wq_addr as *const u64 as *mut std::ffi::c_void,
            &wk_addr as *const u64 as *mut std::ffi::c_void,
            &wv_addr as *const u64 as *mut std::ffi::c_void,
            &qx_addr as *const u64 as *mut std::ffi::c_void,
            &qd_addr as *const u64 as *mut std::ffi::c_void,
            &qs_addr as *const u64 as *mut std::ffi::c_void,
            &q_addr as *const u64 as *mut std::ffi::c_void,
            &k_addr as *const u64 as *mut std::ffi::c_void,
            &v_addr as *const u64 as *mut std::ffi::c_void,
            &ne0_i as *const i32 as *mut std::ffi::c_void,
            &qdim_i as *const i32 as *mut std::ffi::c_void,
            &kvd_i as *const i32 as *mut std::ffi::c_void,
            &batch_i as *const i32 as *mut std::ffi::c_void,
            &qb_addr as *const u64 as *mut std::ffi::c_void,
            &kb_addr as *const u64 as *mut std::ffi::c_void,
            &vb_addr as *const u64 as *mut std::ffi::c_void,
        ];

        let cols_per_block = 4u32;
        let total_cols = (qdim + kvd + kvd) as u32;
        let grid_x = total_cols.div_ceil(cols_per_block);

        self.device.bind_to_thread()?;

        if batch_size > 1 && batch_size <= 4 && !self.fn_fused_qkv_q4k_multi_row.is_null() {
            let shared_bytes_mr = (((ne0 * 5) / 4) * batch_size + 256) as u32;
            let cols_mr = 8u32;
            let grid_x_mr = total_cols.div_ceil(cols_mr);
            let grid_y_mr = 1u32;
            unsafe {
                let lib = sys::lib();
                let res = lib.cuLaunchKernel(
                    self.fn_fused_qkv_q4k_multi_row,
                    grid_x_mr,
                    grid_y_mr,
                    1,
                    256,
                    1,
                    1,
                    shared_bytes_mr,
                    stream.raw(),
                    args.as_ptr() as *mut *mut std::ffi::c_void,
                    std::ptr::null_mut(),
                );
                if res != CUresult::CUDA_SUCCESS {
                    return Err(CudaError::KernelLaunch("cuLaunchKernel (FusedQKV_Q4K Multi-Row)", res));
                }
            }
            return Ok(());
        }

        let grid_y = batch_size as u32;
        let shared_bytes = (((ne0 * 5) / 4) + 256) as u32;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_fused_qkv_q4k,
                grid_x,
                grid_y,
                1,
                256,
                1,
                1,
                shared_bytes,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (FusedQKV_Q4K)", res));
            }
        }

        Ok(())
    }

    /// GPU embedding row lookup from Q4_K quantized weights table:
    /// Directly dequantizes `n_tokens` embeddings onto `x_out` in parallel on GPU.
    pub fn get_rows_q4k(
        &self,
        stream: &CudaStream,
        emb_weights: &DeviceBuffer,
        token_ids_dev: &DeviceBuffer,
        x_out: &DeviceBuffer,
        hidden_dim: usize,
        n_tokens: usize,
    ) -> Result<(), CudaError> {
        let hidden_dim_i: i32 = hidden_dim as i32;
        let n_tokens_i: i32 = n_tokens as i32;
        let emb_addr: u64 = emb_weights.device_ptr();
        let tok_addr: u64 = token_ids_dev.device_ptr();
        let x_addr: u64 = x_out.device_ptr();

        let args: [*mut std::ffi::c_void; 5] = [
            &emb_addr as *const u64 as *mut std::ffi::c_void,
            &tok_addr as *const u64 as *mut std::ffi::c_void,
            &x_addr as *const u64 as *mut std::ffi::c_void,
            &hidden_dim_i as *const i32 as *mut std::ffi::c_void,
            &n_tokens_i as *const i32 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_get_rows_q4k,
                n_tokens as u32,
                1,
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
                return Err(CudaError::KernelLaunch("cuLaunchKernel (GetRowsQ4K)", res));
            }
        }

        Ok(())
    }

    /// GPU embedding row lookup from Q6_K quantized weights table:
    /// Directly dequantizes `n_tokens` embeddings onto `x_out` in parallel on GPU.
    pub fn get_rows_q6k(
        &self,
        stream: &CudaStream,
        emb_weights: &DeviceBuffer,
        token_ids_dev: &DeviceBuffer,
        x_out: &DeviceBuffer,
        hidden_dim: usize,
        n_tokens: usize,
    ) -> Result<(), CudaError> {
        let hidden_dim_i: i32 = hidden_dim as i32;
        let n_tokens_i: i32 = n_tokens as i32;
        let emb_addr: u64 = emb_weights.device_ptr();
        let tok_addr: u64 = token_ids_dev.device_ptr();
        let x_addr: u64 = x_out.device_ptr();

        let args: [*mut std::ffi::c_void; 5] = [
            &emb_addr as *const u64 as *mut std::ffi::c_void,
            &tok_addr as *const u64 as *mut std::ffi::c_void,
            &x_addr as *const u64 as *mut std::ffi::c_void,
            &hidden_dim_i as *const i32 as *mut std::ffi::c_void,
            &n_tokens_i as *const i32 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_get_rows_q6k,
                n_tokens as u32,
                1,
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
                return Err(CudaError::KernelLaunch("cuLaunchKernel (GetRowsQ6K)", res));
            }
        }

        Ok(())
    }

    /// Fast GPU dequantized token embedding lookup for Q8_0 formatted weights.
    pub fn get_rows_q8_0(
        &self,
        stream: &CudaStream,
        emb_weights: &DeviceBuffer,
        token_ids_dev: &DeviceBuffer,
        x_out: &DeviceBuffer,
        hidden_dim: usize,
        n_tokens: usize,
    ) -> Result<(), CudaError> {
        let hidden_dim_i: i32 = hidden_dim as i32;
        let n_tokens_i: i32 = n_tokens as i32;
        let w_addr: u64 = emb_weights.device_ptr();
        let tok_addr: u64 = token_ids_dev.device_ptr();
        let out_addr: u64 = x_out.device_ptr();

        let args: [*mut std::ffi::c_void; 5] = [
            &w_addr as *const u64 as *mut std::ffi::c_void,
            &tok_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &hidden_dim_i as *const i32 as *mut std::ffi::c_void,
            &n_tokens_i as *const i32 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_get_rows_q8_0,
                n_tokens as u32,
                1,
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
                return Err(CudaError::KernelLaunch("cuLaunchKernel (GetRowsQ8_0)", res));
            }
        }

        Ok(())
    }

    /// Fast GPU dequantized token embedding lookup for F16 formatted weights.
    pub fn get_rows_f16(
        &self,
        stream: &CudaStream,
        emb_weights: &DeviceBuffer,
        token_ids_dev: &DeviceBuffer,
        x_out: &DeviceBuffer,
        hidden_dim: usize,
        n_tokens: usize,
    ) -> Result<(), CudaError> {
        let hidden_dim_i: i32 = hidden_dim as i32;
        let n_tokens_i: i32 = n_tokens as i32;
        let w_addr: u64 = emb_weights.device_ptr();
        let tok_addr: u64 = token_ids_dev.device_ptr();
        let out_addr: u64 = x_out.device_ptr();

        let args: [*mut std::ffi::c_void; 5] = [
            &w_addr as *const u64 as *mut std::ffi::c_void,
            &tok_addr as *const u64 as *mut std::ffi::c_void,
            &out_addr as *const u64 as *mut std::ffi::c_void,
            &hidden_dim_i as *const i32 as *mut std::ffi::c_void,
            &n_tokens_i as *const i32 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_get_rows_f16,
                n_tokens as u32,
                1,
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
                return Err(CudaError::KernelLaunch("cuLaunchKernel (GetRowsF16)", res));
            }
        }

        Ok(())
    }

    /// Unified GPU embedding row lookup dispatching to format-specific kernels.
    pub fn get_rows(
        &self,
        stream: &CudaStream,
        emb_weights: &DeviceBuffer,
        token_ids_dev: &DeviceBuffer,
        x_out: &DeviceBuffer,
        hidden_dim: usize,
        n_tokens: usize,
        format: GemvFormat,
    ) -> Result<(), CudaError> {
        match format {
            GemvFormat::Q6K => self.get_rows_q6k(stream, emb_weights, token_ids_dev, x_out, hidden_dim, n_tokens),
            GemvFormat::Q8  => self.get_rows_q8_0(stream, emb_weights, token_ids_dev, x_out, hidden_dim, n_tokens),
            GemvFormat::F16 => self.get_rows_f16(stream, emb_weights, token_ids_dev, x_out, hidden_dim, n_tokens),
            _ => self.get_rows_q4k(stream, emb_weights, token_ids_dev, x_out, hidden_dim, n_tokens),
        }
    }

    /// Fast GPU greedy argmax sampling directly from `logits_dev` into `selected_token_dev`.
    pub fn sample_greedy(
        &self,
        stream: &CudaStream,
        logits_dev: &DeviceBuffer,
        selected_token_dev: &DeviceBuffer,
        vocab_size: usize,
    ) -> Result<(), CudaError> {
        self.sample_greedy_batched(stream, logits_dev, selected_token_dev, vocab_size, 1)
    }

    /// Fast GPU greedy argmax sampling directly from `logits_dev` into `selected_token_dev` for `batch_size` rows.
    pub fn sample_greedy_batched(
        &self,
        stream: &CudaStream,
        logits_dev: &DeviceBuffer,
        selected_token_dev: &DeviceBuffer,
        vocab_size: usize,
        batch_size: usize,
    ) -> Result<(), CudaError> {
        let vocab_size_i: i32 = vocab_size as i32;
        let batch_size_i: i32 = batch_size as i32;
        let logits_addr: u64 = logits_dev.device_ptr();
        let tok_addr: u64 = selected_token_dev.device_ptr();

        let args: [*mut std::ffi::c_void; 4] = [
            &logits_addr as *const u64 as *mut std::ffi::c_void,
            &tok_addr as *const u64 as *mut std::ffi::c_void,
            &vocab_size_i as *const i32 as *mut std::ffi::c_void,
            &batch_size_i as *const i32 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_sample_greedy,
                batch_size as u32,
                1,
                1,
                1024,
                1,
                1,
                0,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (SampleGreedy)", res));
            }
        }

        Ok(())
    }

    /// Advances the generated token step directly on GPU:
    /// Copies `selected_token_dev` -> `token_id_dev`, increments `pos_dev`, and records into `output_history_dev[step_idx]`.
    pub fn advance_token_step(
        &self,
        stream: &CudaStream,
        selected_token_dev: &DeviceBuffer,
        token_id_dev: &DeviceBuffer,
        pos_dev: &DeviceBuffer,
        output_history_dev: Option<&DeviceBuffer>,
        step_counter_dev: Option<&DeviceBuffer>,
    ) -> Result<(), CudaError> {
        let sel_addr: u64 = selected_token_dev.device_ptr();
        let tok_addr: u64 = token_id_dev.device_ptr();
        let pos_addr: u64 = pos_dev.device_ptr();
        let hist_addr: u64 = output_history_dev.map(|h| h.device_ptr()).unwrap_or(0);
        let step_counter_addr: u64 = step_counter_dev.map(|s| s.device_ptr()).unwrap_or(0);

        let args: [*mut std::ffi::c_void; 5] = [
            &sel_addr as *const u64 as *mut std::ffi::c_void,
            &tok_addr as *const u64 as *mut std::ffi::c_void,
            &pos_addr as *const u64 as *mut std::ffi::c_void,
            &hist_addr as *const u64 as *mut std::ffi::c_void,
            &step_counter_addr as *const u64 as *mut std::ffi::c_void,
        ];

        self.device.bind_to_thread()?;

        unsafe {
            let lib = sys::lib();
            let res = lib.cuLaunchKernel(
                self.fn_advance_token,
                1,
                1,
                1,
                32,
                1,
                1,
                0,
                stream.raw(),
                args.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if res != CUresult::CUDA_SUCCESS {
                return Err(CudaError::KernelLaunch("cuLaunchKernel (AdvanceTokenStep)", res));
            }
        }

        Ok(())
    }
}
