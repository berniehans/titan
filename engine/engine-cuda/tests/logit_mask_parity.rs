use engine_cuda::{CudaDevice, CudaStream, DeviceBuffer, LogitMaskGpu};
use std::sync::Arc;

#[test]
#[ignore]
fn test_gpu_logit_mask_parity() -> Result<(), Box<dyn std::error::Error>> {
    engine_cuda::ensure_cuda_dll_paths();
    let dev = CudaDevice::new(0)?;
    let stream = CudaStream::new(dev.clone())?;
    let mask_gpu = LogitMaskGpu::new(dev.clone())?;

    let vocab_size: usize = 128;
    let bitmask_words = (vocab_size + 31) / 32;

    // Initial logits: 0.0, 1.0, 2.0, ..., 127.0
    let mut host_logits = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        host_logits.push(i as f32);
    }

    // Mask: allow only even tokens (0, 2, 4, 6, ...) -> bits 0, 2, 4, 6 set to 1
    let mut host_mask = vec![0x55555555u32; bitmask_words];

    // Explicitly allow token 1 and disallow token 0 in word 0
    host_mask[0] = 0xAAAAAAAA; // allows 1, 3, 5, 7, ...

    let mut dev_logits = DeviceBuffer::alloc(dev.clone(), vocab_size * 4)?;
    let mut dev_mask = DeviceBuffer::alloc(dev.clone(), bitmask_words * 4)?;

    let logits_bytes: Vec<u8> = host_logits.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mask_bytes: Vec<u8> = host_mask.iter().flat_map(|w| w.to_le_bytes()).collect();

    dev_logits.copy_from_host(&stream, &logits_bytes)?;
    dev_mask.copy_from_host(&stream, &mask_bytes)?;

    // Apply GPU logit mask
    mask_gpu.apply_mask(&stream, &dev_logits, &dev_mask, vocab_size)?;
    stream.sync()?;

    let mut out_bytes = vec![0u8; vocab_size * 4];
    dev_logits.copy_to_host(&stream, &mut out_bytes)?;
    stream.sync()?;

    let mut out_logits = Vec::with_capacity(vocab_size);
    for chunk in out_bytes.chunks_exact(4) {
        out_logits.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }

    // Verify: Odd tokens should retain original value, Even tokens should be -1e30
    for (i, &val) in out_logits.iter().enumerate() {
        if i < 32 {
            if i % 2 == 1 {
                assert_eq!(val, i as f32, "Allowed odd token {} was modified", i);
            } else {
                assert!(val <= -1e29, "Disallowed even token {} was not masked (got {})", i, val);
            }
        }
    }

    Ok(())
}
