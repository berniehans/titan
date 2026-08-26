//! CPU MoE executor and concurrent hybrid overlap (Phase 7.4).
//!
//! Executes overflow MoE experts on host CPU threads concurrently while PCIe
//! streams missing experts to the GPU slot cache, then merges the partial
//! representations into the final layer activation tensor.

use super::expert_bank::HostExpertBank;
use crate::error::EngineError;
use std::thread;

/// Executes SwiGLU forward pass for a single expert on CPU.
///
/// Computes: `output = down * (silu(gate * x) * (up * x))`
pub fn cpu_expert_swiglu_step(
    x: &[f32],
    w_gate: &[f32], // [intermediate_dim, hidden_dim]
    w_up: &[f32],   // [intermediate_dim, hidden_dim]
    w_down: &[f32], // [hidden_dim, intermediate_dim]
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), hidden_dim);
    let mut gate_out = vec![0.0f32; intermediate_dim];
    let mut up_out = vec![0.0f32; intermediate_dim];

    // 1. gate = w_gate * x, up = w_up * x
    for i in 0..intermediate_dim {
        let row_gate = &w_gate[i * hidden_dim..(i + 1) * hidden_dim];
        let row_up = &w_up[i * hidden_dim..(i + 1) * hidden_dim];
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for j in 0..hidden_dim {
            g += row_gate[j] * x[j];
            u += row_up[j] * x[j];
        }
        gate_out[i] = g;
        up_out[i] = u;
    }

    // 2. SwiGLU: h = silu(gate) * up
    let mut inter = vec![0.0f32; intermediate_dim];
    for i in 0..intermediate_dim {
        let g = gate_out[i];
        let silu = g / (1.0 + (-g).exp());
        inter[i] = silu * up_out[i];
    }

    // 3. out = w_down * inter
    let mut out = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        let row_down = &w_down[i * intermediate_dim..(i + 1) * intermediate_dim];
        let mut sum = 0.0f32;
        for j in 0..intermediate_dim {
            sum += row_down[j] * inter[j];
        }
        out[i] = sum;
    }

    out
}

/// Configuration for CPU MoE execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuMoeConfig {
    pub layer: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
}

/// Computes weighted sum of CPU overflow experts.
pub fn cpu_moe_execute_overflow(
    x: &[f32],
    cpu_expert_ids: &[usize],
    routing_weights: &[f32],
    bank: &HostExpertBank,
    cfg: CpuMoeConfig,
) -> Result<Vec<f32>, EngineError> {
    let mut accumulator = vec![0.0f32; cfg.hidden_dim];
    if cpu_expert_ids.is_empty() {
        return Ok(accumulator);
    }

    for (idx, &exp_id) in cpu_expert_ids.iter().enumerate() {
        let weight = routing_weights.get(idx).copied().unwrap_or(1.0);
        let gate_bytes = bank
            .get_expert_tensor(cfg.layer, exp_id, "gate_ex")
            .ok_or_else(|| EngineError::Validation(format!("missing gate_ex for exp {exp_id}")))?;
        let up_bytes = bank
            .get_expert_tensor(cfg.layer, exp_id, "up_ex")
            .ok_or_else(|| EngineError::Validation(format!("missing up_ex for exp {exp_id}")))?;
        let down_bytes = bank
            .get_expert_tensor(cfg.layer, exp_id, "down_ex")
            .ok_or_else(|| EngineError::Validation(format!("missing down_ex for exp {exp_id}")))?;

        // Cast float slices (assuming f32 storage in bank for CPU execution)
        let w_gate = bytemuck_cast_slice::<f32>(gate_bytes);
        let w_up = bytemuck_cast_slice::<f32>(up_bytes);
        let w_down = bytemuck_cast_slice::<f32>(down_bytes);

        let exp_out = cpu_expert_swiglu_step(
            x,
            w_gate,
            w_up,
            w_down,
            cfg.hidden_dim,
            cfg.intermediate_dim,
        );

        for j in 0..cfg.hidden_dim {
            accumulator[j] += exp_out[j] * weight;
        }
    }

    Ok(accumulator)
}

/// Helper to cast u8 slice to typed float slice.
fn bytemuck_cast_slice<T>(bytes: &[u8]) -> &[T] {
    let elem_size = std::mem::size_of::<T>();
    let count = bytes.len() / elem_size;
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) }
}

/// Simulates hybrid concurrent overlap: runs CPU execution in parallel with a simulated PCIe gather task.
pub fn execute_hybrid_overlapped_step<F>(
    x: &[f32],
    cpu_expert_ids: Vec<usize>,
    routing_weights: Vec<f32>,
    bank: &'static HostExpertBank,
    cfg: CpuMoeConfig,
    pcie_task: F,
) -> Result<Vec<f32>, EngineError>
where
    F: FnOnce() -> Result<(), EngineError> + Send + 'static,
{
    let x_vec = x.to_vec();

    // Spawn CPU worker thread
    let cpu_handle = thread::spawn(move || {
        cpu_moe_execute_overflow(&x_vec, &cpu_expert_ids, &routing_weights, bank, cfg)
    });

    // Run PCIe transfer on caller stream concurrently
    pcie_task()?;

    // Join CPU worker and return merged CPU contributions
    cpu_handle
        .join()
        .map_err(|_| EngineError::Validation("CPU MoE worker thread panicked".to_string()))?
}
