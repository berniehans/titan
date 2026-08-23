pub mod cache;
pub mod error;

pub use cache::{PagedKvCache, PagedKvCacheConfig};
pub use error::KvCacheError;

pub fn version() -> &'static str {
    "0.1.0"
}
