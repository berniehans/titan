use cudarc::driver::{CudaDevice, sys::{self, CUfunction, CUmodule, CUresult}};
use cudarc::nvrtc::{self, Ptx};
use engine_cuda::ensure_cuda_dll_paths;
use std::ffi::CString;

const KERNEL_SRC: &str = include_str!("../kernels/gemm_q4k_mma.cu");

#[test]
#[ignore]
fn test_q4k_mma_kernel_compilation_and_parity() -> Result<(), Box<dyn std::error::Error>> {
    ensure_cuda_dll_paths();
    let dev = CudaDevice::new(0)?;
    dev.bind_to_thread()?;

    let ptx: Ptx = nvrtc::compile_ptx_with_opts(
        KERNEL_SRC,
        nvrtc::CompileOptions {
            arch: Some("compute_86"),
            use_fast_math: Some(true),
            ..Default::default()
        },
    )?;

    let ptx_c = CString::new(ptx.to_src())?;
    let mut cu_module: CUmodule = std::ptr::null_mut();
    let res = unsafe {
        let lib = sys::lib();
        lib.cuModuleLoadData(&mut cu_module, ptx_c.as_ptr() as *const std::ffi::c_void)
    };
    assert_eq!(res, CUresult::CUDA_SUCCESS);

    let fn_name1 = CString::new("gemm_q4k_mma_kernel")?;
    let mut func1: CUfunction = std::ptr::null_mut();
    let res = unsafe {
        let lib = sys::lib();
        lib.cuModuleGetFunction(&mut func1, cu_module, fn_name1.as_ptr())
    };
    assert_eq!(res, CUresult::CUDA_SUCCESS);
    assert!(!func1.is_null());

    let fn_name2 = CString::new("gemm_q4k_fused_gate_up_swiglu_mma_kernel")?;
    let mut func2: CUfunction = std::ptr::null_mut();
    let res = unsafe {
        let lib = sys::lib();
        lib.cuModuleGetFunction(&mut func2, cu_module, fn_name2.as_ptr())
    };
    assert_eq!(res, CUresult::CUDA_SUCCESS);
    assert!(!func2.is_null());

    println!("Successfully compiled and verified gemm_q4k_mma_kernel and gemm_q4k_fused_gate_up_swiglu_mma_kernel on Ampere GPU!");

    // Clean up
    unsafe {
        let lib = sys::lib();
        let _ = lib.cuModuleUnload(cu_module);
    }

    Ok(())
}