use thiserror::Error;

/// Errors that can occur in `engine-core`.
#[derive(Error, Debug)]
pub enum EngineError {
    /// Error originating from CUDA operations in `engine-cuda`.
    #[error("CUDA error: {0}")]
    Cuda(#[from] engine_cuda::CudaError),

    /// Layer byte length exceeds maximum allowed layer bytes.
    #[error("Layer byte length {actual} exceeds maximum allowed size {expected}")]
    InvalidLayerSize { expected: usize, actual: usize },

    /// CUDA driver error from cudarc.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
}
