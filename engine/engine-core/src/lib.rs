pub mod dequant;
pub mod error;
pub mod forward_cpu;
pub mod forward_driver;
pub mod layer_double_buffer;
pub mod moe;
pub mod ngram_draft;
pub mod pipeline;
pub mod sampler;
pub mod speculative;
pub mod streaming_forward_driver;
pub mod tokenizer;
pub mod vram_accounting;

pub use dequant::{dequant_q4k_cpu, dequant_q6k_cpu};
pub use error::EngineError;
pub use forward_driver::{ForwardDriver, VramFootprint};
pub use layer_double_buffer::{HostLayerWeights, LayerDoubleBuffer, LayerSlotGpu, LayerTensorSizes};
pub use streaming_forward_driver::StreamingForwardDriver;
pub use moe::{
    BandwidthMeasurement, CpuMoeConfig, ExpertSlotCache, ExpertTensorDesc, GpuProfileInfo,
    HardwareBandwidthProfile, HostExpertBank, LayerCacheStats, MoeBackend, MoeBudgetPlan,
    PrefillDoubleBuffer, RewrittenRouting, balanced_fetch, cpu_expert_swiglu_step,
    cpu_moe_execute_overflow, plan_moe_vram_budget, resolve_backend_recommendation,
    resolve_hybrid_fetch_fraction,
};
pub use ngram_draft::NgramDraftProposer;
pub use pipeline::{Pipeline, PipelineStats};
pub use sampler::{Sampler, SamplerParams};
pub use speculative::{SpeculativeVerificationResult, SpeculativeVerifier};
pub use tokenizer::BpeTokenizer;
pub use vram_accounting::{VramStageBreakdown, compute_static_vram_map};

pub fn version() -> &'static str {
    "0.1.0"
}
