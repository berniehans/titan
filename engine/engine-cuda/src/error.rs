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
    StreamFailed(&'static str, cudarc::driver::sys::CUresult),
}
