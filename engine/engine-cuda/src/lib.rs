pub mod dequant;
pub mod dequant_q6k;
pub mod device_buffer;
pub mod error;
pub mod event;
pub mod graphs;
pub mod multiformat_gemv;
pub mod norm_rope;
pub mod paged_attention;
pub mod paged_kv;
pub mod pinned_host;
pub mod streams;

pub use dequant::Q4KDequantizer;
pub use dequant_q6k::Q6KDequantizer;
pub use device_buffer::DeviceBuffer;
pub use error::CudaError;
pub use event::CudaEvent;
pub use graphs::{CudaGraph, CudaGraphExec};
pub use multiformat_gemv::{GemvFormat, MultiFormatGEMV};
pub use norm_rope::{MODE_FUSED, MODE_NORM, MODE_ROPE, MODE_SWIGLU, NormRope};
pub use paged_attention::PagedAttention;
pub use paged_kv::{PagedKvGpu, PagedKvLayout};
pub use pinned_host::PinnedHost;
pub use streams::CudaStream;

pub fn version() -> &'static str {
    "0.1.0"
}
