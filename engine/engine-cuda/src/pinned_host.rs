use crate::error::CudaError;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// RAII wrapper for page-locked (pinned) host memory allocated via CUDA driver API.
///
/// Ensures memory is aligned to 4096 bytes and calls `cuMemFreeHost` on the original
/// base pointer upon drop.
#[derive(Debug)]
pub struct PinnedHost {
    raw_ptr: *mut std::ffi::c_void,
    aligned_ptr: *mut u8,
    size: usize,
    device: Arc<CudaDevice>,
}

// SAFETY: Pinned host memory is host RAM backed by an Arc<CudaDevice>, which is Send + Sync.
// Exclusive access to data is governed by standard Rust borrowing rules.
unsafe impl Send for PinnedHost {}
unsafe impl Sync for PinnedHost {}

impl PinnedHost {
    /// Required memory alignment in bytes (page size = 4096).
    pub const ALIGNMENT: usize = 4096;

    /// Allocates page-locked host memory of at least `size_bytes`, aligned to 4096 bytes,
    /// on default CUDA device 0.
    pub fn alloc(size_bytes: usize) -> Result<Self, CudaError> {
        let device = CudaDevice::new(0)?;
        Self::alloc_on_device(device, size_bytes)
    }

    /// Allocates page-locked host memory of at least `size_bytes`, aligned to 4096 bytes,
    /// on the specified CUDA device.
    pub fn alloc_on_device(device: Arc<CudaDevice>, size_bytes: usize) -> Result<Self, CudaError> {
        device.bind_to_thread()?;

        let raw_size = size_bytes
            .checked_add(Self::ALIGNMENT)
            .ok_or(CudaError::AllocFailed("allocation size overflow"))?;

        let mut raw_ptr: *mut std::ffi::c_void = std::ptr::null_mut();

        // SAFETY:
        // `device.bind_to_thread()` ensured that a valid CUDA context is bound to this thread.
        // `&mut raw_ptr` is a valid pointer on the stack to receive the allocated base address.
        // `cuMemAllocHost_v2` is called with `raw_size` bytes.
        let res = unsafe {
            let lib = cudarc::driver::sys::lib();
            lib.cuMemAllocHost_v2(&mut raw_ptr, raw_size)
        };

        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS || raw_ptr.is_null() {
            return Err(CudaError::AllocFailed("cuMemAllocHost_v2"));
        }

        let raw_addr = raw_ptr as usize;
        let offset = (Self::ALIGNMENT - (raw_addr % Self::ALIGNMENT)) % Self::ALIGNMENT;

        // SAFETY:
        // `raw_ptr` points to an allocated block of `raw_size = size_bytes + ALIGNMENT` bytes.
        // `offset` is strictly in the range `[0, ALIGNMENT)`, so `offset <= raw_size`.
        // The resulting `aligned_ptr` is 4096-aligned and has at least `size_bytes` usable bytes.
        let aligned_ptr = unsafe { (raw_ptr as *mut u8).add(offset) };

        LIVE_ALLOCATIONS.fetch_add(1, Ordering::SeqCst);

        Ok(Self {
            raw_ptr,
            aligned_ptr,
            size: size_bytes,
            device,
        })
    }

    /// Returns the usable allocated size in bytes.
    pub fn bytes(&self) -> usize {
        self.size
    }

    /// Returns the 4096-aligned host pointer to the memory buffer.
    pub fn as_ptr(&self) -> *mut u8 {
        self.aligned_ptr
    }

    /// Returns a slice view of the pinned host memory.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY:
        // `self.aligned_ptr` is non-null, 4096-aligned, and points to `self.size` valid bytes
        // allocated in host pinned RAM. The memory remains valid for the lifetime of `&self`.
        unsafe { std::slice::from_raw_parts(self.aligned_ptr, self.size) }
    }

    /// Returns a mutable slice view of the pinned host memory.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY:
        // `self.aligned_ptr` is non-null, 4096-aligned, and points to `self.size` valid bytes
        // allocated in host pinned RAM. The memory remains valid for the exclusive lifetime of `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.aligned_ptr, self.size) }
    }

    /// Reference to the underlying `CudaDevice`.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Returns the number of currently active `PinnedHost` allocations.
    pub fn live_allocations() -> usize {
        LIVE_ALLOCATIONS.load(Ordering::SeqCst)
    }
}

impl Drop for PinnedHost {
    fn drop(&mut self) {
        if !self.raw_ptr.is_null() {
            let _ = self.device.bind_to_thread();

            // SAFETY:
            // `self.raw_ptr` was allocated by `cuMemAllocHost_v2` in `alloc_on_device`,
            // is non-null, points to the original base address returned by CUDA driver,
            // and has not been previously freed.
            unsafe {
                let lib = cudarc::driver::sys::lib();
                let _res = lib.cuMemFreeHost(self.raw_ptr);
            }

            LIVE_ALLOCATIONS.fetch_sub(1, Ordering::SeqCst);
            self.raw_ptr = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    #[ignore]
    fn test_pinned_256mb_alloc_write_read_free() {
        let _lock = TEST_MUTEX.lock().expect("mutex lock");
        const MIB: usize = 1024 * 1024;
        const SIZE: usize = 256 * MIB;

        let initial_live = PinnedHost::live_allocations();

        {
            let host_mem =
                PinnedHost::alloc(SIZE).expect("Failed to allocate 256 MiB pinned host memory");
            assert!(host_mem.bytes() >= SIZE);

            let ptr = host_mem.as_ptr();
            assert!(!ptr.is_null());
            assert_eq!(
                (ptr as usize) % 4096,
                0,
                "Host memory pointer must be aligned to 4096 bytes"
            );

            assert_eq!(
                PinnedHost::live_allocations(),
                initial_live + 1,
                "Live allocations count should increment on allocation"
            );

            // Write pattern through as_ptr
            // SAFETY:
            // `ptr` points to at least `SIZE` valid, allocated bytes in host pinned memory.
            unsafe {
                for i in 0..SIZE {
                    let val = ((i.wrapping_mul(7)).wrapping_add(13) & 0xFF) as u8;
                    std::ptr::write(ptr.add(i), val);
                }
            }

            // Read back pattern and assert equality
            // SAFETY:
            // `ptr` points to at least `SIZE` valid bytes initialized in the loop above.
            unsafe {
                for i in 0..SIZE {
                    let expected = ((i.wrapping_mul(7)).wrapping_add(13) & 0xFF) as u8;
                    let actual = std::ptr::read(ptr.add(i));
                    assert_eq!(actual, expected, "Data mismatch at byte index {i}");
                }
            }
        }

        assert_eq!(
            PinnedHost::live_allocations(),
            initial_live,
            "Live allocations count should return to initial after drop"
        );
    }

    #[test]
    #[ignore]
    fn test_pinned_small_alloc() {
        let _lock = TEST_MUTEX.lock().expect("mutex lock");
        let initial_live = PinnedHost::live_allocations();
        {
            let pinned =
                PinnedHost::alloc(1024).expect("Failed to allocate 1024 bytes pinned memory");
            assert_eq!(pinned.bytes(), 1024);
            assert_eq!((pinned.as_ptr() as usize) % 4096, 0);
            assert_eq!(PinnedHost::live_allocations(), initial_live + 1);
        }
        assert_eq!(PinnedHost::live_allocations(), initial_live);
    }

    #[test]
    #[ignore]
    fn test_pinned_slice_accessors() {
        let _lock = TEST_MUTEX.lock().expect("mutex lock");
        let mut pinned = PinnedHost::alloc(4096).expect("Failed to allocate pinned memory");
        let slice_mut = pinned.as_mut_slice();
        slice_mut.fill(42);

        let slice = pinned.as_slice();
        assert_eq!(slice.len(), 4096);
        assert!(slice.iter().all(|&b| b == 42));
    }

    #[test]
    fn test_alignment_offset_math() {
        let compute_offset = |addr: usize| -> usize {
            (PinnedHost::ALIGNMENT - (addr % PinnedHost::ALIGNMENT)) % PinnedHost::ALIGNMENT
        };

        assert_eq!(compute_offset(0), 0);
        assert_eq!(compute_offset(4096), 0);
        assert_eq!(compute_offset(8192), 0);
        assert_eq!(compute_offset(1), 4095);
        assert_eq!(compute_offset(4095), 1);
        assert_eq!(compute_offset(4097), 4095);
        assert_eq!((1 + compute_offset(1)) % 4096, 0);
        assert_eq!((4095 + compute_offset(4095)) % 4096, 0);
        assert_eq!((4097 + compute_offset(4097)) % 4096, 0);
    }
}
