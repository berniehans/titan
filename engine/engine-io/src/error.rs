use thiserror::Error;

/// Errors that can occur when parsing or reading GGUF files.
#[derive(Error, Debug)]
pub enum GgufError {
    /// I/O error occurred while reading the file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// CUDA error occurred during pinned memory allocation or transfer.
    #[error("CUDA error: {0}")]
    Cuda(#[from] engine_cuda::CudaError),

    /// File does not start with magic bytes "GGUF".
    #[error("Invalid GGUF magic: expected b\"GGUF\", found {0:?}")]
    InvalidMagic([u8; 4]),

    /// GGUF version is not supported (only v3 is supported).
    #[error("Unsupported GGUF version: expected 3, found {0}")]
    UnsupportedVersion(u32),

    /// String was not valid UTF-8.
    #[error("Invalid UTF-8 string: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    /// Unknown or unsupported GGUF value type ID.
    #[error("Invalid GGUF value type ID: {0}")]
    InvalidValueType(u32),

    /// Unknown or unsupported GGML tensor type ID.
    #[error("Invalid GGML tensor type ID: {0}")]
    InvalidTensorType(u32),

    /// Unexpected end of file while reading field.
    #[error("Unexpected EOF while parsing {0}")]
    UnexpectedEof(&'static str),

    /// Invalid alignment value.
    #[error("Invalid alignment: {0}")]
    InvalidAlignment(u64),

    /// Tensor shape / dimensions are invalid.
    #[error("Invalid tensor shape for '{0}'")]
    InvalidTensorShape(String),

    /// Tensor data bounds exceed file size.
    #[error("Tensor '{name}' out of bounds (offset {offset}, size {size}, file size {file_size})")]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        size: u64,
        file_size: u64,
    },
}
