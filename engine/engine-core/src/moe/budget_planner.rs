//! MoE VRAM budget planner and prefill double buffer (Phase 7.5).
//!
//! Enforces the "MoE-first" VRAM budget policy:
//! 1. Reserve KV cache pages first (guarantees context capacity).
//! 2. Double buffer for prefill streaming (2 expert blocks for async DMA + GEMM overlap).
//! 3. Dynamic floor: floor = 2 * experts_per_layer when prefill overlap feasible, else 1 * experts_per_layer.
//! 4. Greedy expert slots: allocate remaining VRAM to maximize resident GPU expert slots.
//! 5. Hard assertion: total allocated VRAM <= total available VRAM budget.

use crate::error::EngineError;

/// Breakdown and allocation result produced by the MoE budget planner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeBudgetPlan {
    /// Total VRAM budget available (bytes).
    pub total_vram_budget_bytes: usize,
    /// Base non-expert weights (bytes).
    pub static_weights_bytes: usize,
    /// KV cache memory reserved (bytes).
    pub kv_reserved_bytes: usize,
    /// Activation workspace memory (bytes).
    pub activations_bytes: usize,
    /// Prefill streaming double buffer size (bytes).
    pub prefill_double_buffer_bytes: usize,
    /// Memory allocated to resident GPU expert slots (bytes).
    pub expert_slots_bytes: usize,
    /// Number of GPU expert slots allocated per layer.
    pub n_slots_per_layer: usize,
    /// Whether prefill overlap is feasible under this budget.
    pub prefill_overlap_feasible: bool,
    /// Total VRAM allocated across all stages.
    pub total_allocated_bytes: usize,
}

impl MoeBudgetPlan {
    /// Returns remaining unallocated VRAM headroom in bytes.
    pub fn free_headroom_bytes(&self) -> usize {
        self.total_vram_budget_bytes
            .saturating_sub(self.total_allocated_bytes)
    }

    /// Utilization percentage of the VRAM budget (0.0 to 100.0%).
    pub fn utilization_pct(&self) -> f64 {
        if self.total_vram_budget_bytes == 0 {
            0.0
        } else {
            (self.total_allocated_bytes as f64 / self.total_vram_budget_bytes as f64) * 100.0
        }
    }
}

/// Computes the optimal MoE VRAM allocation plan.
pub fn plan_moe_vram_budget(
    total_vram_budget_bytes: usize,
    static_weights_bytes: usize,
    kv_reserved_bytes: usize,
    activations_bytes: usize,
    n_layers: usize,
    n_experts_per_layer: usize,
    expert_slice_size_bytes: usize,
) -> Result<MoeBudgetPlan, EngineError> {
    let fixed_base_bytes = static_weights_bytes
        .checked_add(kv_reserved_bytes)
        .and_then(|s| s.checked_add(activations_bytes))
        .ok_or_else(|| {
            EngineError::Validation("Integer overflow in base VRAM calculation".to_string())
        })?;

    if fixed_base_bytes > total_vram_budget_bytes {
        return Err(EngineError::Validation(format!(
            "Base VRAM requirements ({fixed_base_bytes} bytes) exceed total budget ({total_vram_budget_bytes} bytes)"
        )));
    }

    let remaining_for_moe = total_vram_budget_bytes - fixed_base_bytes;

    // Prefill double buffer requires 2 expert blocks
    let double_buffer_bytes = 2 * expert_slice_size_bytes;
    let prefill_overlap_feasible = remaining_for_moe >= double_buffer_bytes;

    let prefill_bytes = if prefill_overlap_feasible {
        double_buffer_bytes
    } else {
        expert_slice_size_bytes // single buffer fallback
    };

    let remaining_for_slots = remaining_for_moe.saturating_sub(prefill_bytes);

    // Single slot across all layers costs: n_layers * expert_slice_size_bytes
    let layer_slot_cost_bytes = n_layers * expert_slice_size_bytes;

    let n_slots_per_layer = remaining_for_slots
        .checked_div(layer_slot_cost_bytes)
        .map(|slots| slots.min(n_experts_per_layer))
        .unwrap_or(0);

    let expert_slots_bytes = n_slots_per_layer * layer_slot_cost_bytes;
    let total_allocated_bytes = fixed_base_bytes + prefill_bytes + expert_slots_bytes;

    assert!(
        total_allocated_bytes <= total_vram_budget_bytes,
        "Budget invariant violated: allocated {total_allocated_bytes} > budget {total_vram_budget_bytes}"
    );

    Ok(MoeBudgetPlan {
        total_vram_budget_bytes,
        static_weights_bytes,
        kv_reserved_bytes,
        activations_bytes,
        prefill_double_buffer_bytes: prefill_bytes,
        expert_slots_bytes,
        n_slots_per_layer,
        prefill_overlap_feasible,
        total_allocated_bytes,
    })
}

/// Ping-pong double buffer for overlapping host-to-device streaming and compute.
#[derive(Debug)]
pub struct PrefillDoubleBuffer<T> {
    buf_a: T,
    buf_b: T,
    active_is_a: bool,
}

impl<T> PrefillDoubleBuffer<T> {
    /// Wraps two allocated buffers into a ping-pong double buffer.
    pub fn new(buf_a: T, buf_b: T) -> Self {
        Self {
            buf_a,
            buf_b,
            active_is_a: true,
        }
    }

    /// Returns references to (transfer_target, compute_source).
    pub fn ping_pong_pair(&self) -> (&T, &T) {
        if self.active_is_a {
            (&self.buf_b, &self.buf_a)
        } else {
            (&self.buf_a, &self.buf_b)
        }
    }

    /// Returns mutable references to (transfer_target, compute_source).
    pub fn ping_pong_pair_mut(&mut self) -> (&mut T, &mut T) {
        if self.active_is_a {
            (&mut self.buf_b, &mut self.buf_a)
        } else {
            (&mut self.buf_a, &mut self.buf_b)
        }
    }

    /// Swaps the active buffer roles.
    pub fn swap(&mut self) {
        self.active_is_a = !self.active_is_a;
    }
}
