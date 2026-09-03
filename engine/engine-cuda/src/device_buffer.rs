use crate::error::CudaError;
use crate::streams::CudaStream;
use cudarc::driver::CudaDevice;
use cudarc::driver::sys::CUdeviceptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// RAII wrapper for VRAM device memory allocated via CUDA driver API `cuMemAlloc`.
#[derive(Debug)]
pub struct DeviceBuffer {
    dptr: CUdeviceptr,
    size: usize,
    device: Arc<CudaDevice>,
}

// SAFETY: DeviceBuffer wraps a CUDA device memory pointer (CUdeviceptr) tied to an Arc<CudaDevice>.
// CUDA device pointers can be safely transferred and accessed across threads following standard Rust borrowing rules.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    /// Allocates `size_bytes` of VRAM device memory on the specified CUDA device.
    pub fn alloc(device: Arc<CudaDevice>, size_bytes: usize) -> Result<Self, CudaError> {
        device.bind_to_thread()?;

        if size_bytes == 0 {
            return Err(CudaError::InvalidSize {
                expected: 1,
                actual: 0,
            });
        }

        let mut dptr: CUdeviceptr = 0;

        // SAFETY:
        // `device.bind_to_thread()` ensured that a valid CUDA context is bound to this thread.
        // `&mut dptr` is a valid pointer on the stack to receive the allocated device pointer.
        // `cuMemAlloc_v2` is called with `size_bytes` bytes.
        let res = unsafe {
            let lib = cudarc::driver::sys::lib();
            lib.cuMemAlloc_v2(&mut dptr, size_bytes)
        };

        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS || dptr == 0 {
            return Err(CudaError::AllocFailed("cuMemAlloc_v2"));
        }

        LIVE_ALLOCATIONS.fetch_add(1, Ordering::SeqCst);

        Ok(Self {
            dptr,
            size: size_bytes,
            device,
        })
    }

    /// Returns the allocated size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the raw CUDA device pointer (`CUdeviceptr`).
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.dptr
    }

    /// Copies data from a host buffer slice to this device memory buffer using the specified stream.
    /// Copies data from a host buffer slice to this device memory buffer using the specified stream
    /// asynchronously without stream synchronization.
    pub fn copy_from_host_async(&self, stream: &CudaStream, src: &[u8]) -> Result<(), CudaError> {
        if src.len() > self.size {
            return Err(CudaError::InvalidSize {
                expected: self.size,
                actual: src.len(),
            });
        }

        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured valid CUDA context on this thread.
        // `self.dptr` points to allocated device memory of at least `self.size` bytes.
        // `src` is a valid host slice of `src.len()` bytes.
        // `stream.raw()` is a valid CUDA stream handle.
        let res = unsafe {
            let lib = cudarc::driver::sys::lib();
            lib.cuMemcpyHtoDAsync_v2(
                self.dptr,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
                stream.raw(),
            )
        };

        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(CudaError::MemcpyFailed("cuMemcpyHtoDAsync_v2", res));
        }

        Ok(())
    }

    /// Copies data from a host buffer slice to this device memory buffer using the specified stream.
    ///
    /// Synchronizes the stream after launching the copy helper.
    pub fn copy_from_host(&self, stream: &CudaStream, src: &[u8]) -> Result<(), CudaError> {
        self.copy_from_host_async(stream, src)?;
        stream.sync()?;
        Ok(())
    }

    /// Copies data from this device memory buffer to a host buffer slice using the specified stream.
    ///
    /// Synchronizes the stream after launching the copy helper.
    pub fn copy_to_host(&self, stream: &CudaStream, dst: &mut [u8]) -> Result<(), CudaError> {
        self.copy_to_host_async(stream, dst)?;
        stream.sync()?;
        Ok(())
    }

    /// Copies data from this device memory buffer to a host buffer slice without synchronizing.
    pub fn copy_to_host_async(&self, stream: &CudaStream, dst: &mut [u8]) -> Result<(), CudaError> {
        if dst.len() > self.size {
            return Err(CudaError::InvalidSize {
                expected: self.size,
                actual: dst.len(),
            });
        }

        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured valid CUDA context on this thread.
        // `self.dptr` points to allocated device memory of at least `self.size` bytes.
        // `dst` is a valid mutable host slice of `dst.len()` bytes.
        // `stream.raw()` is a valid CUDA stream handle.
        let res = unsafe {
            let lib = cudarc::driver::sys::lib();
            lib.cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                self.dptr,
                dst.len(),
                stream.raw(),
            )
        };

        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(CudaError::MemcpyFailed("cuMemcpyDtoHAsync_v2", res));
        }

        Ok(())
    }

    /// Reference to the underlying `CudaDevice`.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Returns the number of currently active `DeviceBuffer` allocations.
    pub fn live_allocations() -> usize {
        LIVE_ALLOCATIONS.load(Ordering::SeqCst)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.dptr != 0 {
            let _ = self.device.bind_to_thread();

            // SAFETY:
            // `self.dptr` was allocated by `cuMemAlloc_v2` in `DeviceBuffer::alloc`,
            // is non-zero, points to valid device memory, and has not been previously freed.
            unsafe {
                let lib = cudarc::driver::sys::lib();
                let _res = lib.cuMemFree_v2(self.dptr);
            }

            LIVE_ALLOCATIONS.fetch_sub(1, Ordering::SeqCst);
            self.dptr = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_live_allocations() {
        assert_eq!(DeviceBuffer::live_allocations(), 0);
    }
}
