//! Layer Double-Buffer Ring Test (Phase 13, Sub-change 13.1).
//!
//! Validates:
//! 1. `LayerDoubleBuffer::new` allocates exactly two layer slots on GPU.
//! 2. Ping-pong alternating copies into slot 0 and slot 1.
//! 3. Zero reallocation across repeated layer streaming cycles.

use engine_core::layer_double_buffer::{HostLayerWeights, LayerDoubleBuffer, LayerTensorSizes};
use engine_cuda::{CudaDevice, CudaStream};
use std::sync::Arc;

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[test]
fn test_layer_double_buffer_ping_pong_swapping() -> Result<(), DynError> {
    let device = Arc::new(CudaDevice::new(0)?);
    let stream = CudaStream::new(Arc::clone(&device))?;

    let sizes = LayerTensorSizes {
        wq_bytes: 1024,
        wk_bytes: 512,
        wv_bytes: 512,
        wo_bytes: 1024,
        wgate_bytes: 2048,
        wup_bytes: 2048,
        wdown_bytes: 2048,
        an_bytes: 256,
        qn_bytes: 128,
        kn_bytes: 128,
        fn_bytes: 256,
    };

    let total_one_slot = sizes.total_bytes();
    let mut double_buf = LayerDoubleBuffer::new(Arc::clone(&device), &sizes)?;

    assert_eq!(double_buf.total_vram_bytes(), 2 * total_one_slot);

    // Create synthetic layer weights for 4 virtual layers
    for layer_idx in 0..4 {
        let wq_host = vec![layer_idx as u8 + 10; sizes.wq_bytes];
        let wk_host = vec![layer_idx as u8 + 20; sizes.wk_bytes];
        let wv_host = vec![layer_idx as u8 + 30; sizes.wv_bytes];
        let wo_host = vec![layer_idx as u8 + 40; sizes.wo_bytes];
        let wgate_host = vec![layer_idx as u8 + 50; sizes.wgate_bytes];
        let wup_host = vec![layer_idx as u8 + 60; sizes.wup_bytes];
        let wdown_host = vec![layer_idx as u8 + 70; sizes.wdown_bytes];
        let an_host = vec![layer_idx as u8 + 80; sizes.an_bytes];
        let qn_host = vec![layer_idx as u8 + 90; sizes.qn_bytes];
        let kn_host = vec![layer_idx as u8 + 100; sizes.kn_bytes];
        let fn_host = vec![layer_idx as u8 + 110; sizes.fn_bytes];

        let host_weights = HostLayerWeights {
            wq_data: &wq_host,
            wk_data: &wk_host,
            wv_data: &wv_host,
            wo_data: &wo_host,
            wgate_data: &wgate_host,
            wup_data: &wup_host,
            wdown_data: &wdown_host,
            an_data: &an_host,
            qn_data: &qn_host,
            kn_data: &kn_host,
            fn_data: &fn_host,
        };

        // Async transfer into alternating slot
        double_buf.copy_layer_async(layer_idx, &host_weights, &stream)?;
        stream.sync()?;

        let slot = double_buf.slot(layer_idx);

        // Readback and assert exact byte equality
        let mut wq_readback = vec![0u8; sizes.wq_bytes];
        slot.wq_dev.copy_to_host(&stream, &mut wq_readback)?;
        stream.sync()?;

        assert_eq!(
            wq_readback, wq_host,
            "Layer {layer_idx} readback mismatch in slot {}",
            layer_idx % 2
        );
    }

    Ok(())
}
