//! GPU expert slot cache and capped-fetch LRU replacement (Phase 7.3).
//!
//! Implements device-side and host-side routing rewrite for hybrid MoE execution:
//! - Resident hit -> slot ID
//! - Fetched-this-step -> assigned slot ID (evicting LRU)
//! - Overflow miss -> -1 (routed to concurrent CPU executor)
//!
//! Provides the FreeToken `_balanced_fetch` algorithm to prevent PCIe over-fetching.

/// Balanced fetch count: calculates integer fetch count `F ~ frac * misses`
/// choosing between floor and ceil to minimize the longer of the two concurrent
/// execution pipelines (PCIe fetch vs CPU compute).
///
/// Cost formulation:
/// - PCIe time ~ `F * (1 - frac)`
/// - CPU time ~ `(misses - F) * frac`
pub fn balanced_fetch(num_missing: usize, frac_q16: u32) -> usize {
    let q = 1u32 << 16;
    let lo = ((num_missing as u32 * frac_q16) >> 16) as usize;
    let cost = |f: usize| -> u64 {
        let f_u64 = f as u64;
        let m_u64 = num_missing as u64;
        let pcie_cost = f_u64 * (q.saturating_sub(frac_q16)) as u64;
        let cpu_cost = (m_u64.saturating_sub(f_u64)) * frac_q16 as u64;
        pcie_cost.max(cpu_cost)
    };
    if cost(lo) <= cost(lo + 1) {
        lo.min(num_missing)
    } else {
        (lo + 1).min(num_missing)
    }
}

/// Statistics accumulated per layer for telemetry and profiling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerCacheStats {
    /// Total expert requests routed to this layer.
    pub active_requests: usize,
    /// Requests that were hits in the resident GPU slot cache.
    pub resident_hits: usize,
    /// Requests fetched from host memory over PCIe into the slot cache.
    pub pcie_fetched: usize,
    /// Requests that overflowed the fetch cap and executed on CPU.
    pub cpu_overflow: usize,
}

impl LayerCacheStats {
    /// Miss rate before fetch capping: `(pcie_fetched + cpu_overflow) / active_requests`.
    pub fn pre_cap_miss_rate(&self) -> f64 {
        if self.active_requests == 0 {
            0.0
        } else {
            (self.pcie_fetched + self.cpu_overflow) as f64 / self.active_requests as f64
        }
    }

    /// Effective GPU coverage: `(resident_hits + pcie_fetched) / active_requests`.
    pub fn gpu_coverage_rate(&self) -> f64 {
        if self.active_requests == 0 {
            0.0
        } else {
            (self.resident_hits + self.pcie_fetched) as f64 / self.active_requests as f64
        }
    }
}

/// Outcome of rewriting a layer's routed expert IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewrittenRouting {
    /// Rewritten slot IDs aligned with input requests:
    /// - Non-negative values (`>= 0`): GPU slot index
    /// - Negative value (`-1`): CPU overflow (compute on host)
    pub slot_ids: Vec<i32>,
    /// List of expert indices that must be fetched over PCIe this step.
    pub fetched_experts: Vec<usize>,
    /// Destination GPU slot index for each fetched expert.
    pub fetch_target_slots: Vec<usize>,
    /// List of expert indices that must be computed on CPU this step.
    pub cpu_experts: Vec<usize>,
}

/// LRU GPU expert slot cache manager.
#[derive(Debug, Clone)]
pub struct ExpertSlotCache {
    n_layers: usize,
    n_experts_per_layer: usize,
    n_slots_per_layer: usize,
    // [layer * n_slots + slot] -> expert_id or -1
    slot_to_expert: Vec<i32>,
    // [layer * n_experts + expert] -> slot_id or -1
    expert_to_slot: Vec<i32>,
    // [layer * n_slots + slot] -> step timestamp
    slot_last_used: Vec<u64>,
    // Per-layer telemetry
    stats: Vec<LayerCacheStats>,
}

impl ExpertSlotCache {
    /// Creates a new `ExpertSlotCache`.
    pub fn new(n_layers: usize, n_experts_per_layer: usize, n_slots_per_layer: usize) -> Self {
        let total_slots = n_layers * n_slots_per_layer;
        let total_experts = n_layers * n_experts_per_layer;
        Self {
            n_layers,
            n_experts_per_layer,
            n_slots_per_layer,
            slot_to_expert: vec![-1; total_slots],
            expert_to_slot: vec![-1; total_experts],
            slot_last_used: vec![0; total_slots],
            stats: vec![LayerCacheStats::default(); n_layers],
        }
    }

    /// Resets all slot bindings and recency tracking (e.g. at start of new sequence).
    pub fn reset(&mut self) {
        self.slot_to_expert.fill(-1);
        self.expert_to_slot.fill(-1);
        self.slot_last_used.fill(0);
        for s in &mut self.stats {
            *s = LayerCacheStats::default();
        }
    }

    /// Number of GPU slots available per layer.
    pub fn n_slots_per_layer(&self) -> usize {
        self.n_slots_per_layer
    }

    /// Access cumulative statistics for a layer.
    pub fn stats(&self, layer: usize) -> Option<&LayerCacheStats> {
        self.stats.get(layer)
    }

    /// Rewrites requested expert IDs for a single layer decode step.
    ///
    /// - Identifies resident hits -> slot ID
    /// - Calculates balanced fetch count `F` for misses
    /// - Evicts LRU slots to assign `F` fetched experts -> slot ID
    /// - Marks remaining misses as `-1` (CPU overflow)
    pub fn step_layer(
        &mut self,
        layer: usize,
        requested_experts: &[usize],
        fetch_fraction: f64,
        step_id: u64,
    ) -> RewrittenRouting {
        assert!(layer < self.n_layers, "layer out of range");
        let mut slot_ids = vec![-1i32; requested_experts.len()];
        let mut misses = Vec::new();
        let mut fetched_experts = Vec::new();
        let mut fetch_target_slots = Vec::new();
        let mut cpu_experts = Vec::new();

        let layer_slot_base = layer * self.n_slots_per_layer;
        let layer_exp_base = layer * self.n_experts_per_layer;

        // Pass 1: Check hits
        for (i, &exp_id) in requested_experts.iter().enumerate() {
            assert!(exp_id < self.n_experts_per_layer, "expert_id out of range");
            let slot = self.expert_to_slot[layer_exp_base + exp_id];
            if slot >= 0 {
                // Hit
                slot_ids[i] = slot;
                self.slot_last_used[layer_slot_base + slot as usize] = step_id;
                self.stats[layer].resident_hits += 1;
            } else {
                misses.push((i, exp_id));
            }
            self.stats[layer].active_requests += 1;
        }

        // Pass 2: Calculate balanced fetch count
        let frac_q16 = (fetch_fraction.clamp(0.0, 1.0) * 65536.0).round() as u32;
        let max_fetch = balanced_fetch(misses.len(), frac_q16);

        // Pass 3: Assign fetched slots via LRU eviction
        for (idx_in_misses, &(req_idx, exp_id)) in misses.iter().enumerate() {
            if idx_in_misses < max_fetch {
                // Evict LRU slot in this layer
                let mut lru_slot = 0;
                let mut oldest_step = u64::MAX;
                for s in 0..self.n_slots_per_layer {
                    let last = self.slot_last_used[layer_slot_base + s];
                    if last < oldest_step {
                        oldest_step = last;
                        lru_slot = s;
                    }
                }

                // Evict previous tenant
                let prev_tenant = self.slot_to_expert[layer_slot_base + lru_slot];
                if prev_tenant >= 0 {
                    self.expert_to_slot[layer_exp_base + prev_tenant as usize] = -1;
                }

                // Bind new tenant
                self.slot_to_expert[layer_slot_base + lru_slot] = exp_id as i32;
                self.expert_to_slot[layer_exp_base + exp_id] = lru_slot as i32;
                self.slot_last_used[layer_slot_base + lru_slot] = step_id;

                slot_ids[req_idx] = lru_slot as i32;
                fetched_experts.push(exp_id);
                fetch_target_slots.push(lru_slot);
                self.stats[layer].pcie_fetched += 1;
            } else {
                // Overflow -> route to CPU
                slot_ids[req_idx] = -1;
                cpu_experts.push(exp_id);
                self.stats[layer].cpu_overflow += 1;
            }
        }

        RewrittenRouting {
            slot_ids,
            fetched_experts,
            fetch_target_slots,
            cpu_experts,
        }
    }
}
