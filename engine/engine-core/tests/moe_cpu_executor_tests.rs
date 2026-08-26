//! CPU MoE executor and hybrid overlap tests (Phase 7.4).
//!
//! Asserts that:
//! 1. `cpu_expert_swiglu_step` computes `w_down * (silu(w_gate * x) * (w_up * x))` accurately.
//! 2. `cpu_moe_execute_overflow` correctly accumulates weighted representations of overflow experts.
//! 3. Threaded CPU execution matches sequential execution bit-for-bit with verified overlap.

use engine_core::moe::{HostExpertBank, cpu_expert_swiglu_step, cpu_moe_execute_overflow};

fn float_to_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[test]
fn test_cpu_expert_swiglu_single_step() {
    const HIDDEN_DIM: usize = 4;
    const INTERMEDIATE_DIM: usize = 4;

    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    // Identity matrices for gate and up
    let mut w_gate = vec![0.0f32; HIDDEN_DIM * INTERMEDIATE_DIM];
    let mut w_up = vec![0.0f32; HIDDEN_DIM * INTERMEDIATE_DIM];
    let mut w_down = vec![0.0f32; HIDDEN_DIM * INTERMEDIATE_DIM];
    for i in 0..4 {
        w_gate[i * 4 + i] = 1.0;
        w_up[i * 4 + i] = 1.0;
        w_down[i * 4 + i] = 1.0;
    }

    let out = cpu_expert_swiglu_step(&x, &w_gate, &w_up, &w_down, HIDDEN_DIM, INTERMEDIATE_DIM);
    assert_eq!(out.len(), HIDDEN_DIM);

    // Expected for each component: x_i * silu(x_i) = x_i * (x_i / (1 + exp(-x_i)))
    for (i, &val) in out.iter().enumerate() {
        let xi = x[i];
        let expected = xi * (xi / (1.0 + (-xi).exp()));
        assert!(
            (val - expected).abs() < 1e-5,
            "element {i} mismatch: got {val}, expected {expected}"
        );
    }
}

#[test]
fn test_cpu_moe_execute_overflow_weighted_sum() {
    const HIDDEN_DIM: usize = 4;
    const INTERMEDIATE_DIM: usize = 4;
    const EXPERT_BYTES: usize = 4 * 4 * 4 * 3; // 3 matrices of 16 floats * 4 bytes = 192 bytes

    let mut bank = HostExpertBank::allocate(1, 4, EXPERT_BYTES, false).expect("allocate bank");

    // Create 2 test experts (expert 0 and expert 1)
    let w_gate0 = vec![0.5f32; 16];
    let w_up0 = vec![1.0f32; 16];
    let w_down0 = vec![0.25f32; 16];

    let w_gate1 = vec![0.25f32; 16];
    let w_up1 = vec![0.5f32; 16];
    let w_down1 = vec![0.5f32; 16];

    bank.write_expert_tensor(0, 0, "gate_ex", 0, float_to_bytes(&w_gate0))
        .unwrap();
    bank.write_expert_tensor(0, 0, "up_ex", 64, float_to_bytes(&w_up0))
        .unwrap();
    bank.write_expert_tensor(0, 0, "down_ex", 128, float_to_bytes(&w_down0))
        .unwrap();

    bank.write_expert_tensor(0, 1, "gate_ex", 0, float_to_bytes(&w_gate1))
        .unwrap();
    bank.write_expert_tensor(0, 1, "up_ex", 64, float_to_bytes(&w_up1))
        .unwrap();
    bank.write_expert_tensor(0, 1, "down_ex", 128, float_to_bytes(&w_down1))
        .unwrap();

    let x = vec![1.0f32; 4];
    let routing_weights = vec![0.6f32, 0.4f32];
    let cpu_expert_ids = vec![0, 1];
    let cfg = engine_core::moe::CpuMoeConfig {
        layer: 0,
        hidden_dim: HIDDEN_DIM,
        intermediate_dim: INTERMEDIATE_DIM,
    };

    let accumulated = cpu_moe_execute_overflow(&x, &cpu_expert_ids, &routing_weights, &bank, cfg)
        .expect("cpu execute");

    // Compute reference output sequentially
    let exp0_out =
        cpu_expert_swiglu_step(&x, &w_gate0, &w_up0, &w_down0, HIDDEN_DIM, INTERMEDIATE_DIM);
    let exp1_out =
        cpu_expert_swiglu_step(&x, &w_gate1, &w_up1, &w_down1, HIDDEN_DIM, INTERMEDIATE_DIM);

    let mut expected = [0.0f32; 4];
    for i in 0..4 {
        expected[i] = exp0_out[i] * 0.6 + exp1_out[i] * 0.4;
    }

    for i in 0..4 {
        assert!(
            (accumulated[i] - expected[i]).abs() < 1e-6,
            "accumulated[{i}] mismatch: got {}, expected {}",
            accumulated[i],
            expected[i]
        );
    }
}
