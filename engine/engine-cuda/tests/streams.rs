use cudarc::driver::CudaDevice;
use engine_cuda::{CudaError, CudaStream};
use std::sync::Arc;

#[test]
#[ignore]
fn test_stream_memset_sync() -> Result<(), CudaError> {
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;

    assert!(!stream.raw().is_null());

    // Verify sync on empty/fresh stream succeeds
    stream.sync()?;

    // Verify cuStreamQuery returns success (CUDA_SUCCESS) after sync
    // SAFETY: stream.raw() is a valid CUstream handle.
    unsafe {
        let lib = cudarc::driver::sys::lib();
        let res = lib.cuStreamQuery(stream.raw());
        assert_eq!(
            res,
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "cuStreamQuery should return CUDA_SUCCESS after sync"
        );
    }

    Ok(())
}
