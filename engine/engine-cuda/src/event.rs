use crate::error::CudaError;
use crate::streams::CudaStream;
use cudarc::driver::CudaDevice;
use cudarc::driver::sys::CUevent;
use std::sync::Arc;

/// RAII wrapper for a CUDA event (`CUevent`) allocated via CUDA driver API `cuEventCreate`.
#[derive(Debug)]
pub struct CudaEvent {
    event: CUevent,
    device: Arc<CudaDevice>,
}

// SAFETY: `CUevent` is a CUDA driver event handle tied to an `Arc<CudaDevice>`.
// CUDA event handles can be safely sent and shared across threads.
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    /// Creates a new CUDA event on the specified CUDA device with default flags (flags = 0).
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CudaError> {
        device.bind_to_thread()?;

        let mut event: CUevent = std::ptr::null_mut();

        // SAFETY:
        // `device.bind_to_thread()` ensured that a valid CUDA context is bound to this thread.
        // `&mut event` is a valid pointer on the stack to receive the allocated event handle.
        // `cuEventCreate` is called with default flags (0).
        let res = unsafe {
            let lib = cudarc::driver::sys::lib();
            lib.cuEventCreate(&mut event, 0)
        };

        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS || event.is_null() {
            return Err(CudaError::EventFailed("cuEventCreate", res));
        }

        Ok(Self { event, device })
    }

    /// Records an event on the specified CUDA stream.
    pub fn record(&self, stream: &CudaStream) -> Result<(), CudaError> {
        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured valid CUDA context on this thread.
        // `self.event` is a valid `CUevent` handle created by `cuEventCreate`.
        // `stream.raw()` is a valid `CUstream` handle.
        let res = unsafe {
            let lib = cudarc::driver::sys::lib();
            lib.cuEventRecord(self.event, stream.raw())
        };

        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(CudaError::EventFailed("cuEventRecord", res));
        }

        Ok(())
    }

    /// Makes the specified CUDA stream wait for this event to complete before executing
    /// subsequent operations queued on that stream.
    pub fn stream_wait(&self, stream: &CudaStream) -> Result<(), CudaError> {
        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured valid CUDA context on this thread.
        // `stream.raw()` is a valid `CUstream` handle.
        // `self.event` is a valid `CUevent` handle created by `cuEventCreate`.
        // Flags argument is set to default (0).
        let res = unsafe {
            let lib = cudarc::driver::sys::lib();
            lib.cuStreamWaitEvent(stream.raw(), self.event, 0)
        };

        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(CudaError::EventFailed("cuStreamWaitEvent", res));
        }

        Ok(())
    }

    /// Queries the completion status of the event without blocking.
    ///
    /// Returns `true` if the event has completed, or `false` otherwise.
    pub fn query(&self) -> bool {
        let _ = self.device.bind_to_thread();

        // SAFETY:
        // `self.device.bind_to_thread()` ensured valid CUDA context on this thread.
        // `self.event` is a valid `CUevent` handle.
        let res = unsafe {
            let lib = cudarc::driver::sys::lib();
            lib.cuEventQuery(self.event)
        };

        res == cudarc::driver::sys::CUresult::CUDA_SUCCESS
    }

    /// Computes the elapsed time in milliseconds between `earlier` and `self`.
    pub fn elapsed_ms(&self, earlier: &CudaEvent) -> Result<f32, CudaError> {
        self.device.bind_to_thread()?;

        let mut ms: f32 = 0.0;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured valid CUDA context on this thread.
        // `&mut ms` is a valid stack pointer to receive the result.
        // `earlier.event` and `self.event` are valid `CUevent` handles recorded on streams.
        let res = unsafe {
            let lib = cudarc::driver::sys::lib();
            lib.cuEventElapsedTime(&mut ms, earlier.event, self.event)
        };

        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(CudaError::EventFailed("cuEventElapsedTime", res));
        }

        Ok(ms)
    }

    /// Returns the raw `CUevent` handle.
    pub fn raw(&self) -> CUevent {
        self.event
    }

    /// Reference to the underlying `CudaDevice`.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            let _ = self.device.bind_to_thread();

            // SAFETY:
            // `self.event` was created by `cuEventCreate` in `CudaEvent::new`,
            // is non-null, and has not been destroyed yet.
            unsafe {
                let lib = cudarc::driver::sys::lib();
                let _res = lib.cuEventDestroy_v2(self.event);
            }

            self.event = std::ptr::null_mut();
        }
    }
}
