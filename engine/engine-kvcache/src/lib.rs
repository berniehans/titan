pub mod cache;
pub mod cow;
pub mod error;
pub mod radix;

pub use cache::{PagedKvCache, PagedKvCacheConfig};
pub use cow::{CowBlockTable, SharedBlock};
pub use error::KvCacheError;
pub use radix::{PhysicalBlockId, RadixMatchResult, RadixNode, RadixTree};

pub fn version() -> &'static str {
    "0.1.0"
}
