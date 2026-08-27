//! Double-buffered layer weight ring for large-model streaming (Phase 13, Sub-change 13.1).
//!
//! Allocates two fixed GPU memory slots (`slot_a`, `slot_b`) to hold transformer layer weights,
//! enabling streaming execution of arbitrarily large models (14B/32B/70B) within a bounded
//! VRAM window (~600 MB) with zero runtime reallocations.

use crate::error::EngineError;
use engine_cuda::{CudaDevice, CudaStream, DeviceBuffer};
use std::sync::Arc;

/// Byte sizes required for each weight tensor in a single transformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerTensorSizes {
    pub wq_bytes: usize,
    pub wk_bytes: usize,
    pub wv_bytes: usize,
    pub wo_bytes: usize,
    pub wgate_bytes: usize,
    pub wup_bytes: usize,
    pub wdown_bytes: usize,
    pub an_bytes: usize,
    pub qn_bytes: usize,
    pub kn_bytes: usize,
    pub fn_bytes: usize,
}

impl LayerTensorSizes {
    /// Sum of all tensor sizes in a single layer slot.
    pub fn total_bytes(&self) -> usize {
        self.wq_bytes
            + self.wk_bytes
            + self.wv_bytes
            + self.wo_bytes
            + self.wgate_bytes
            + self.wup_bytes
            + self.wdown_bytes
            + self.an_bytes
            + self.qn_bytes
            + self.kn_bytes
            + self.fn_bytes
    }
}

/// A preallocated GPU slot holding all weights for one transformer layer.
pub struct LayerSlotGpu {
    pub wq_dev: DeviceBuffer,
    pub wk_dev: DeviceBuffer,
    pub wv_dev: DeviceBuffer,
    pub wo_dev: DeviceBuffer,
    pub wgate_dev: DeviceBuffer,
    pub wup_dev: DeviceBuffer,
    pub wdown_dev: DeviceBuffer,
    pub an_dev: DeviceBuffer,
    pub qn_dev: DeviceBuffer,
    pub kn_dev: DeviceBuffer,
    pub fn_dev: DeviceBuffer,
    pub sizes: LayerTensorSizes,
}

impl LayerSlotGpu {
    /// Allocates a new fixed GPU layer slot.
    pub fn alloc(device: Arc<CudaDevice>, sizes: &LayerTensorSizes) -> Result<Self, EngineError> {
        let wq_dev = DeviceBuffer::alloc(Arc::clone(&device), sizes.wq_bytes)?;
        let wk_dev = DeviceBuffer::alloc(Arc::clone(&device), sizes.wk_bytes)?;
        let wv_dev = DeviceBuffer::alloc(Arc::clone(&device), sizes.wv_bytes)?;
        let wo_dev = DeviceBuffer::alloc(Arc::clone(&device), sizes.wo_bytes)?;
        let wgate_dev = DeviceBuffer::alloc(Arc::clone(&device), sizes.wgate_bytes)?;
        let wup_dev = DeviceBuffer::alloc(Arc::clone(&device), sizes.wup_bytes)?;
        let wdown_dev = DeviceBuffer::alloc(Arc::clone(&device), sizes.wdown_bytes)?;
        let an_dev = DeviceBuffer::alloc(Arc::clone(&device), sizes.an_bytes)?;
        let qn_dev = DeviceBuffer::alloc(Arc::clone(&device), sizes.qn_bytes)?;
        let kn_dev = DeviceBuffer::alloc(Arc::clone(&device), sizes.kn_bytes)?;
        let fn_dev = DeviceBuffer::alloc(Arc::clone(&device), sizes.fn_bytes)?;

        Ok(Self {
            wq_dev,
            wk_dev,
            wv_dev,
            wo_dev,
            wgate_dev,
            wup_dev,
            wdown_dev,
            an_dev,
            qn_dev,
            kn_dev,
            fn_dev,
            sizes: *sizes,
        })
    }
}

/// Host layer weight slices ready for DMA transfer.
pub struct HostLayerWeights<'a> {
    pub wq_data: &'a [u8],
    pub wk_data: &'a [u8],
    pub wv_data: &'a [u8],
    pub wo_data: &'a [u8],
    pub wgate_data: &'a [u8],
    pub wup_data: &'a [u8],
    pub wdown_data: &'a [u8],
    pub an_data: &'a [u8],
    pub qn_data: &'a [u8],
    pub kn_data: &'a [u8],
    pub fn_data: &'a [u8],
}

/// Double-buffered GPU layer weight ring.
pub struct LayerDoubleBuffer {
    pub slot_a: LayerSlotGpu,
    pub slot_b: LayerSlotGpu,
    pub sizes: LayerTensorSizes,
}

impl LayerDoubleBuffer {
    /// Creates a new double-buffer allocating exactly two layer slots on the CUDA device.
    pub fn new(device: Arc<CudaDevice>, sizes: &LayerTensorSizes) -> Result<Self, EngineError> {
        let slot_a = LayerSlotGpu::alloc(Arc::clone(&device), sizes)?;
        let slot_b = LayerSlotGpu::alloc(Arc::clone(&device), sizes)?;

        Ok(Self {
            slot_a,
            slot_b,
            sizes: *sizes,
        })
    }

    /// Returns a reference to the slot with index `idx % 2`.
    pub fn slot(&self, idx: usize) -> &LayerSlotGpu {
        if idx % 2 == 0 {
            &self.slot_a
        } else {
            &self.slot_b
        }
    }

    /// Asynchronously transfers host layer weights into slot `idx % 2` over `stream`.
    pub fn copy_layer_async(
        &mut self,
        idx: usize,
        weights: &HostLayerWeights<'_>,
        stream: &CudaStream,
    ) -> Result<(), EngineError> {
        let target_slot = if idx % 2 == 0 {
            &mut self.slot_a
        } else {
            &mut self.slot_b
        };

        target_slot.wq_dev.copy_from_host(stream, weights.wq_data)?;
        target_slot.wk_dev.copy_from_host(stream, weights.wk_data)?;
        target_slot.wv_dev.copy_from_host(stream, weights.wv_data)?;
        target_slot.wo_dev.copy_from_host(stream, weights.wo_data)?;
        target_slot.wgate_dev.copy_from_host(stream, weights.wgate_data)?;
        target_slot.wup_dev.copy_from_host(stream, weights.wup_data)?;
        target_slot.wdown_dev.copy_from_host(stream, weights.wdown_data)?;
        target_slot.an_dev.copy_from_host(stream, weights.an_data)?;
        target_slot.qn_dev.copy_from_host(stream, weights.qn_data)?;
        target_slot.kn_dev.copy_from_host(stream, weights.kn_data)?;
        target_slot.fn_dev.copy_from_host(stream, weights.fn_data)?;

        Ok(())
    }

    /// Total VRAM bytes occupied by both ping-pong slots.
    pub fn total_vram_bytes(&self) -> usize {
        2 * self.sizes.total_bytes()
    }
}
