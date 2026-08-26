use cudarc::driver::sys::CUresult;
use thiserror::Error;

/// Errors that can occur during CUDA operations in `engine-cuda`.
#[derive(Error, Debug)]
pub enum CudaError {
    /// CUDA driver error from cudarc.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    /// CUDA host memory allocation failed.
    #[error("CUDA memory allocation failed in '{0}'")]
    AllocFailed(&'static str),

    /// CUDA host memory free failed.
    #[error("CUDA memory free failed in '{0}'")]
    FreeFailed(&'static str),

    /// CUDA stream operation failed.
    #[error("CUDA stream operation failed in '{0}': {1:?}")]
    StreamFailed(&'static str, CUresult),

    /// CUDA memory copy failed.
    #[error("CUDA memory copy operation failed in '{0}': {1:?}")]
    MemcpyFailed(&'static str, CUresult),

    /// CUDA event operation failed.
    #[error("CUDA event operation failed in '{0}': {1:?}")]
    EventFailed(&'static str, CUresult),

    /// CUDA kernel compilation failed.
    #[error("CUDA kernel compilation failed: {0}")]
    KernelCompile(String),

    /// CUDA kernel module load / symbol resolution failed.
    #[error("CUDA kernel load failed in '{0}': {1:?}")]
    KernelLoad(&'static str, CUresult),

    /// CUDA kernel launch failed.
    #[error("CUDA kernel launch failed in '{0}': {1:?}")]
    KernelLaunch(&'static str, CUresult),

    /// CUDA graph operation failed.
    #[error("CUDA graph operation failed in '{0}': {1:?}")]
    GraphFailed(&'static str, CUresult),

    /// Invalid memory transfer or buffer size.
    #[error("Invalid buffer size: expected <= {expected}, got {actual}")]
    InvalidSize { expected: usize, actual: usize },
}
