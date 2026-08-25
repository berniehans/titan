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

    /// A dequantizer is configured but the layer length is not a multiple of
    /// the Q4_K_M super-block size (144 bytes).
    #[error(
        "Layer byte length {actual} is not a multiple of {block_bytes} required for Q4_K_M dequantization"
    )]
    LayerNotDequantAligned { block_bytes: usize, actual: usize },

    /// CUDA driver error from cudarc.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    /// Error sourced from GGUF/metadata parsing (`engine-io`).
    #[error("GGUF error: {0}")]
    Gguf(#[from] engine_io::GgufError),

    /// A tokenizer pre-tokenization reached an unsupported (non-ASCII) input.
    ///
    /// The ported Qwen2 pre-tokenizer intentionally covers only ASCII; anything
    /// else must be handled by a future full-Unicode pass.
    #[error("tokenizer: unsupported non-ASCII input at byte {0}")]
    NonAsciiInput(usize),

    /// Tokenizer tried to decode an unknown token id (out of vocabulary range).
    #[error("tokenizer: unknown token id {0} for decode")]
    UnknownToken(u32),
}
