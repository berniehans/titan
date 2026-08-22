use cudarc::driver::CudaDevice;
use engine_core::{EngineError, Pipeline};
use std::sync::Arc;

#[test]
#[ignore]
fn test_pipeline_layer_ordering() -> Result<(), EngineError> {
    let device = CudaDevice::new(0)?;

    const LAYER_BYTES: usize = 1024 * 1024;
    const NUM_LAYERS: usize = 8;

    let pipeline = Pipeline::new(Arc::clone(&device), LAYER_BYTES)?;

    // 8 layers of 1 MB distinct patterns
    let layers_data: Vec<Vec<u8>> = (0..NUM_LAYERS)
        .map(|i| vec![(i as u8).wrapping_add(1); LAYER_BYTES])
        .collect();

    let layer_refs: Vec<&[u8]> = layers_data.iter().map(|l| l.as_slice()).collect();

    let stats = pipeline.run(&layer_refs)?;

    assert_eq!(stats.layers, 8);

    // Verify slot 0 holds layer 6 (zero-indexed: layer 6 % 2 == 0)
    let mut slot0_out = vec![0u8; LAYER_BYTES];
    pipeline.slots()[0].copy_to_host(pipeline.transfer_stream(), &mut slot0_out)?;
    assert_eq!(slot0_out, layers_data[6], "Slot 0 should contain layer 6");

    // Verify slot 1 holds layer 7 (zero-indexed: layer 7 % 2 == 1)
    let mut slot1_out = vec![0u8; LAYER_BYTES];
    pipeline.slots()[1].copy_to_host(pipeline.transfer_stream(), &mut slot1_out)?;
    assert_eq!(slot1_out, layers_data[7], "Slot 1 should contain layer 7");

    Ok(())
}
