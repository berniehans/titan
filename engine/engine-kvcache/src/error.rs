use thiserror::Error;

/// Errors that can occur in `engine-kvcache`.
#[derive(Error, Debug)]
pub enum KvCacheError {
    /// The block pool is exhausted; no free physical block remains for a new allocation.
    #[error("KV-cache block pool exhausted: {blocks_used} of {blocks_total} blocks in use")]
    PoolExhausted {
        blocks_used: usize,
        blocks_total: usize,
    },

    /// Invalid argument (e.g. mismatched key/value length, or head dimension).
    #[error("Invalid KV-cache argument: {0}")]
    InvalidArgs(&'static str),
}
