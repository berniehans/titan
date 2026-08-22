use crate::error::CudaError;
use cudarc::driver::CudaDevice;
use cudarc::driver::sys::CUstream;
use std::sync::Arc;

/// RAII wrapper for a CUDA stream (`CUstream`) allocated via CUDA driver API `cuStreamCreate`.
#[derive(Debug)]
pub struct CudaStream {
    stream: CUstream,
    device: Arc<CudaDevice>,
}

// SAFETY: `CUstream` is a CUDA driver stream handle tied to an `Arc<CudaDevice>`.
// CUDA stream handles can be safely sent and shared across threads.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Creates a new CUDA stream on the specified CUDA device with default flags (flags = 0).
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CudaError> {
        device.bind_to_thread()?;

        let mut stream: CUstream = std::ptr::null_mut();

        // SAFETY:
        // `device.bind_to_thread()` ensured that a valid CUDA context is bound to this thread.
        // `&mut stream` is a valid pointer on the stack to receive the allocated stream handle.
        // `cuStreamCreate` is called with non-blocking flags = 0.
        let res = unsafe {
            let lib = cudarc::driver::sys::lib();
            lib.cuStreamCreate(&mut stream, 0)
        };

        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS || stream.is_null() {
            return Err(CudaError::StreamFailed("cuStreamCreate", res));
        }

        Ok(Self { stream, device })
    }

    /// Returns the raw `CUstream` handle.
    pub fn raw(&self) -> CUstream {
        self.stream
    }

    /// Synchronizes the CUDA stream, blocking until all preceding commands in this stream have completed.
    pub fn sync(&self) -> Result<(), CudaError> {
        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured valid CUDA context on this thread.
        // `self.stream` is a valid `CUstream` handle created by `cuStreamCreate`.
        let res = unsafe {
            let lib = cudarc::driver::sys::lib();
            lib.cuStreamSynchronize(self.stream)
        };

        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(CudaError::StreamFailed("cuStreamSynchronize", res));
        }

        Ok(())
    }

    /// Reference to the underlying `CudaDevice`.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let _ = self.device.bind_to_thread();

            // SAFETY:
            // `self.stream` was created by `cuStreamCreate` in `CudaStream::new`,
            // is non-null, and has not been destroyed yet.
            unsafe {
                let lib = cudarc::driver::sys::lib();
                let _res = lib.cuStreamDestroy_v2(self.stream);
            }

            self.stream = std::ptr::null_mut();
        }
    }
}
