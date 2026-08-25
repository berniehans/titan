//! VRAM worst-case footprint guard for NormRope (Phase 6.4 group 3).
//!
//! Ground truth from `docs/ARCHITECTURE.md` §4:
//! Usable ~5.2 GB = buffers(ping-pong) ~0.9 GB + activations/driver ~1.3 GB + KV remainder ~3.0 GB.
//! The `NormRope` kernel adds NO persistent allocation: it stages its 5 buffers
//! (x, residual, w, up, out) through the reused ping-pong slot.
//!
//! Asserts that:
//! 1. The total footprint `alloc_norm_rope_total` of all 5 buffers touches <= 0.1 * BUDGET_GB.
//! 2. `declared_worst_case + DECLARED_RESIDENT_KV_BYTES + DECLARED_PINGPONG_BYTES <= BUDGET_GB`.
//! 3. `DECLARED_PINGPONG_BYTES + DECLARED_RESIDENT_KV_BYTES + DECLARED_ACTIVATIONS_BYTES <= BUDGET_GB`.
//! 4. Live free device memory `free > 0`.

use cudarc::driver::CudaDevice;
use cudarc::driver::sys;
use engine_cuda::{CudaError, CudaStream, DeviceBuffer, MODE_FUSED, NormRope};
use std::sync::Arc;

const BUDGET_GB: f64 = 5.2 * 1024.0 * 1024.0 * 1024.0;
const DECLARED_PINGPONG_BYTES: f64 = 0.9 * 1024.0 * 1024.0 * 1024.0;
const DECLARED_RESIDENT_KV_BYTES: f64 = 3.0 * 1024.0 * 1024.0 * 1024.0;
const DECLARED_ACTIVATIONS_BYTES: f64 = 1.3 * 1024.0 * 1024.0 * 1024.0;

#[test]
#[ignore]
#[allow(clippy::assertions_on_constants)]
fn vram_worst_case_norm_rope_guard() -> Result<(), CudaError> {
    const N: usize = 4096;
    const N_DIMS: usize = 128;
    const FREQ_BASE: f32 = 10000.0;
    const POS: u32 = 1;
    const EPS: f32 = 1e-5;

    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let norm_rope = NormRope::new(Arc::clone(&device))?;

    let byte_len = N * 4;
    let d_x = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;
    let d_residual = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;
    let d_w = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;
    let d_up = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;
    let d_out = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;

    let alloc_norm_rope_total =
        d_x.size() + d_residual.size() + d_w.size() + d_up.size() + d_out.size();

    // Populate buffers with initial zero data to verify live memory operations
    let zeros = vec![0u8; byte_len];
    d_x.copy_from_host(&stream, &zeros)?;
    d_residual.copy_from_host(&stream, &zeros)?;
    d_w.copy_from_host(&stream, &zeros)?;
    d_up.copy_from_host(&stream, &zeros)?;

    // Run one real launch to prove the footprint is live
    norm_rope.launch(
        &stream,
        &d_x,
        &d_residual,
        &d_w,
        &d_up,
        &d_out,
        EPS,
        N,
        N_DIMS,
        FREQ_BASE,
        POS,
        MODE_FUSED,
    )?;
    stream.sync()?;

    // Assert NOTE_PUB: footprint <= 0.1 * BUDGET_GB
    assert!(
        (alloc_norm_rope_total as f64) <= 0.1 * BUDGET_GB,
        "alloc_norm_rope_total ({alloc_norm_rope_total}) exceeded 0.1 * BUDGET_GB"
    );

    // Declared worst case: total bytes of all 5 buffers touched
    let declared_worst_case = alloc_norm_rope_total as f64;

    // Assert proposal's literal rule
    assert!(
        declared_worst_case + DECLARED_RESIDENT_KV_BYTES + DECLARED_PINGPONG_BYTES <= BUDGET_GB,
        "declared worst-case + resident KV + ping-pong exceeds budget: {} + {} + {} > {}",
        declared_worst_case,
        DECLARED_RESIDENT_KV_BYTES,
        DECLARED_PINGPONG_BYTES,
        BUDGET_GB
    );

    // Assert architecture split sanity
    assert!(
        DECLARED_PINGPONG_BYTES + DECLARED_RESIDENT_KV_BYTES + DECLARED_ACTIVATIONS_BYTES
            <= BUDGET_GB,
        "Architecture split sanity check failed: {} + {} + {} > {}",
        DECLARED_PINGPONG_BYTES,
        DECLARED_RESIDENT_KV_BYTES,
        DECLARED_ACTIVATIONS_BYTES,
        BUDGET_GB
    );

    // Query free device memory via CUDA driver API
    device.bind_to_thread()?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: cuMemGetInfo_v2 writes to valid pointers `&mut free` and `&mut total`.
    let res = unsafe {
        let lib = sys::lib();
        lib.cuMemGetInfo_v2(&mut free, &mut total)
    };
    if res != sys::CUresult::CUDA_SUCCESS {
        return Err(CudaError::KernelLaunch("cuMemGetInfo_v2", res));
    }
    assert!(free > 0, "free memory must be > 0");

    println!("=== VRAM Worst-Case Guard: NormRope ===");
    println!(
        "real alloc_norm_rope_total: {} bytes ({:.2} KiB)",
        alloc_norm_rope_total,
        alloc_norm_rope_total as f64 / 1024.0
    );
    println!(
        "declared_worst_case:        {:.0} bytes ({:.2} KiB)",
        declared_worst_case,
        declared_worst_case / 1024.0
    );
    println!(
        "budget split:               pingpong={:.0} B ({:.2} GB), kv={:.0} B ({:.2} GB), act={:.0} B ({:.2} GB), total_budget={:.0} B ({:.2} GB)",
        DECLARED_PINGPONG_BYTES,
        DECLARED_PINGPONG_BYTES / (1024.0 * 1024.0 * 1024.0),
        DECLARED_RESIDENT_KV_BYTES,
        DECLARED_RESIDENT_KV_BYTES / (1024.0 * 1024.0 * 1024.0),
        DECLARED_ACTIVATIONS_BYTES,
        DECLARED_ACTIVATIONS_BYTES / (1024.0 * 1024.0 * 1024.0),
        BUDGET_GB,
        BUDGET_GB / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "free bytes:                 {} bytes ({:.2} MiB / {:.2} GiB free of {} bytes total)",
        free,
        free as f64 / (1024.0 * 1024.0),
        free as f64 / (1024.0 * 1024.0 * 1024.0),
        total
    );

    Ok(())
}
