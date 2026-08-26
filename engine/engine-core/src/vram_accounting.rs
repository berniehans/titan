//! VRAM accounting module (Phase 6.9).
//!
//! Provides static estimation and dynamic runtime breakdown of device memory (VRAM)
//! across all engine stages:
//!   - Ping-pong layer staging buffers / uploaded weights
//!   - Resident per-layer paged KV cache pools
//!   - Activation cliffs / reusable scratch device buffers
//!   - Host-device logits transfer buffers

use crate::forward_driver::VramFootprint;
use engine_io::ModelConfig;

/// Per-stage VRAM breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramStageBreakdown {
    /// Memory for uploaded weights / ping-pong buffers.
    pub pingpong_bytes: usize,
    /// Memory for resident per-layer KV pools.
    pub kv_pool_bytes: usize,
    /// Memory for activation scratch buffers across all kernels.
    pub activations_bytes: usize,
    /// Memory for output logits buffer.
    pub logits_bytes: usize,
}

impl VramStageBreakdown {
    /// Computes the total working set in bytes.
    pub fn total_bytes(&self) -> usize {
        self.pingpong_bytes + self.kv_pool_bytes + self.activations_bytes + self.logits_bytes
    }

    /// Computes the fraction of the budget used (0.0 to 1.0).
    pub fn budget_utilization(&self, budget_bytes: usize) -> f64 {
        self.total_bytes() as f64 / budget_bytes.max(1) as f64
    }

    /// Asserts that total working set is <= budget_bytes.
    pub fn assert_within_budget(&self, budget_bytes: usize) -> Result<(), String> {
        let tot = self.total_bytes();
        if tot > budget_bytes {
            Err(format!(
                "VRAM working set {} bytes ({:.2} MB) exceeds budget {} bytes ({:.2} MB)",
                tot,
                tot as f64 / (1024.0 * 1024.0),
                budget_bytes,
                budget_bytes as f64 / (1024.0 * 1024.0),
            ))
        } else {
            Ok(())
        }
    }

    /// Formats a complete human-readable trace table.
    pub fn format_trace(&self, budget_bytes: usize) -> String {
        let tot = self.total_bytes();
        let util = self.budget_utilization(budget_bytes) * 100.0;
        format!(
            "=== VRAM Stage Accounting Breakdown ===\n\
               Stage 1 (Weights/Ping-pong): {:>10} bytes ({:>8.2} MB, {:>5.1}%)\n\
               Stage 2 (Resident KV Pool):   {:>10} bytes ({:>8.2} MB, {:>5.1}%)\n\
               Stage 3 (Scratch Activations):{:>10} bytes ({:>8.2} MB, {:>5.1}%)\n\
               Stage 4 (Logits Transfer):    {:>10} bytes ({:>8.2} MB, {:>5.1}%)\n\
             -------------------------------------------------------\n\
               Total Working Set:          {:>10} bytes ({:>8.2} MB, {:>5.2} GB)\n\
               VRAM Budget Bound:          {:>10} bytes ({:>8.2} MB, {:>5.2} GB)\n\
               Budget Utilization:         {:>7.2}%\n\
             =======================================================",
            self.pingpong_bytes,
            self.pingpong_bytes as f64 / (1024.0 * 1024.0),
            (self.pingpong_bytes as f64 / tot.max(1) as f64) * 100.0,
            self.kv_pool_bytes,
            self.kv_pool_bytes as f64 / (1024.0 * 1024.0),
            (self.kv_pool_bytes as f64 / tot.max(1) as f64) * 100.0,
            self.activations_bytes,
            self.activations_bytes as f64 / (1024.0 * 1024.0),
            (self.activations_bytes as f64 / tot.max(1) as f64) * 100.0,
            self.logits_bytes,
            self.logits_bytes as f64 / (1024.0 * 1024.0),
            (self.logits_bytes as f64 / tot.max(1) as f64) * 100.0,
            tot,
            tot as f64 / (1024.0 * 1024.0),
            tot as f64 / (1024.0 * 1024.0 * 1024.0),
            budget_bytes,
            budget_bytes as f64 / (1024.0 * 1024.0),
            budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            util
        )
    }
}

impl From<VramFootprint> for VramStageBreakdown {
    fn from(f: VramFootprint) -> Self {
        Self {
            pingpong_bytes: f.pingpong_bytes,
            kv_pool_bytes: f.kv_pool_bytes,
            activations_bytes: f.activations_bytes,
            logits_bytes: f.logits_bytes,
        }
    }
}

/// Computes the static VRAM accounting map from model configuration parameters.
pub fn compute_static_vram_map(
    cfg: &ModelConfig,
    max_seq_tokens: usize,
    vocab_size: usize,
    is_streaming_double_buffered: bool,
) -> VramStageBreakdown {
    let h = cfg.hidden_size as usize;
    let hd = cfg.head_dim as usize;
    let nh = cfg.n_head as usize;
    let nkv = cfg.n_head_kv as usize;
    let hff = cfg.intermediate_size as usize;
    let n_layer = cfg.n_layer as usize;

    let q4k_bytes = |elements: usize| -> usize { (elements / 256) * 144 };
    let q_elems = (nh * hd) * h;
    let k_elems = (nkv * hd) * h;
    let o_elems = h * (nh * hd);
    let gate_elems = hff * h;
    let up_elems = hff * h;
    let layer_weights = q4k_bytes(q_elems)
        + q4k_bytes(k_elems)
        + q4k_bytes(o_elems)
        + q4k_bytes(gate_elems)
        + q4k_bytes(up_elems)
        + (h * 4)   // attn_norm
        + (hd * 4)  // q_norm
        + (hd * 4)  // k_norm
        + (h * 4); // ffn_norm

    let pingpong_bytes = if is_streaming_double_buffered {
        2 * layer_weights
    } else {
        n_layer * layer_weights
    };

    let floats_per_token = 2 * (nkv * hd);
    let kv_pool_bytes = n_layer * max_seq_tokens * floats_per_token * std::mem::size_of::<f32>();

    let qdim = nh * hd;
    let kvd = nkv * hd;
    let activations_bytes = (h * 4)
        + (h * 4)
        + (qdim * 4)
        + (kvd * 4)
        + (kvd * 4)
        + (hd * 4)
        + (qdim * 4)
        + (h * 4)
        + (h * 4)
        + (h * 4)
        + (hff * 4)
        + (hff * 4)
        + (hff * 4)
        + (h * 4)
        + (hd * 4)
        + (hff * 4)
        + (std::mem::size_of::<i32>());

    let logits_bytes = vocab_size * std::mem::size_of::<f32>();

    VramStageBreakdown {
        pingpong_bytes,
        kv_pool_bytes,
        activations_bytes,
        logits_bytes,
    }
}
