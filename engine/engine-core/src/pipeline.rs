use crate::error::EngineError;
use cudarc::driver::CudaDevice;
use engine_cuda::{CudaEvent, CudaStream, DeviceBuffer, Q4KDequantizer};
use std::sync::Arc;
use std::time::Instant;

/// Execution statistics returned by `Pipeline::run`.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineStats {
    /// Total number of layers processed.
    pub layers: usize,
    /// Total wall-clock elapsed time in milliseconds for `Pipeline::run`.
    pub elapsed_ms: f64,
    /// Per-layer GPU elapsed time in milliseconds measured via CUDA events.
    pub layer_elapsed_ms: Vec<f32>,
}

/// Double-buffered CUDA pipeline driver for streaming layer transfers and compute.
pub struct Pipeline {
    device: Arc<CudaDevice>,
    transfer_stream: CudaStream,
    compute_stream: CudaStream,
    slots: [DeviceBuffer; 2],
    copy_done: [CudaEvent; 2],
    compute_done: [CudaEvent; 2],
    max_layer_bytes: usize,
    /// Optional Q4_K dequantizer run in the compute stage. When `None`, the
    /// compute stage keeps its historical stub behavior (old tests stay green).
    dequantizer: Option<Q4KDequantizer>,
    /// Ping-pong output buffers for dequantized floats (only when `dequantizer`
    /// is configured). Each holds `max_layer_bytes / 144 * 256` `f32`.
    dequant_out: Option<[DeviceBuffer; 2]>,
}

/// Bytes per Q4_K_M super-block input.
const Q4K_BLOCK_BYTES: usize = 144;
/// Dequantized floats per Q4_K_M super-block.
const Q4K_FLOATS_PER_BLOCK: usize = 256;

impl Pipeline {
    /// Creates a new `Pipeline` with ping-pong device buffer slots sized to `max_layer_bytes`.
    ///
    /// The compute stage is a stub (no dequantization): it only records the
    /// per-layer compute-done event, preserving the historical pipeline behavior.
    pub fn new(device: Arc<CudaDevice>, max_layer_bytes: usize) -> Result<Self, EngineError> {
        let transfer_stream = CudaStream::new(Arc::clone(&device))?;
        let compute_stream = CudaStream::new(Arc::clone(&device))?;

        let slot0 = DeviceBuffer::alloc(Arc::clone(&device), max_layer_bytes)?;
        let slot1 = DeviceBuffer::alloc(Arc::clone(&device), max_layer_bytes)?;

        let copy_done0 = CudaEvent::new(Arc::clone(&device))?;
        let copy_done1 = CudaEvent::new(Arc::clone(&device))?;

        let compute_done0 = CudaEvent::new(Arc::clone(&device))?;
        let compute_done1 = CudaEvent::new(Arc::clone(&device))?;

        Ok(Self {
            device,
            transfer_stream,
            compute_stream,
            slots: [slot0, slot1],
            copy_done: [copy_done0, copy_done1],
            compute_done: [compute_done0, compute_done1],
            max_layer_bytes,
            dequantizer: None,
            dequant_out: None,
        })
    }

    /// Creates a new `Pipeline` with a Q4_K_M dequantizer running in the
    /// compute stage.
    ///
    /// Compiles and loads the GPU dequant kernel, and allocates ping-pong
    /// output buffers sized to hold the dequantized floats for `max_layer_bytes`
    /// (i.e. `max_layer_bytes / 144 * 256` `f32` per slot). Layers must be
    /// Q4_K_M-aligned (multiples of 144 bytes) when the dequantizer is enabled.
    pub fn with_dequantizer(
        device: Arc<CudaDevice>,
        max_layer_bytes: usize,
    ) -> Result<Self, EngineError> {
        let mut pipeline = Self::new(Arc::clone(&device), max_layer_bytes)?;

        let dequantizer = Q4KDequantizer::new(Arc::clone(&device))?;

        let blocks_per_slot = max_layer_bytes / Q4K_BLOCK_BYTES;
        let out_bytes = blocks_per_slot * Q4K_FLOATS_PER_BLOCK * std::mem::size_of::<f32>();
        let out0 = DeviceBuffer::alloc(Arc::clone(&device), out_bytes)?;
        let out1 = DeviceBuffer::alloc(Arc::clone(&device), out_bytes)?;

        pipeline.dequantizer = Some(dequantizer);
        pipeline.dequant_out = Some([out0, out1]);
        Ok(pipeline)
    }

    /// Runs all provided layers through the double-buffered pipeline.
    ///
    /// Async H2D transfers execute on the transfer stream and stub computes execute on the compute stream.
    /// Overlap is coordinated via CUDA events without CPU busy-waiting or inner-loop stream synchronization.
    pub fn run(&self, layers: &[&[u8]]) -> Result<PipelineStats, EngineError> {
        let start_instant = Instant::now();

        let mut layer_starts = Vec::with_capacity(layers.len());
        let mut layer_ends = Vec::with_capacity(layers.len());

        for (i, &layer) in layers.iter().enumerate() {
            if layer.len() > self.max_layer_bytes {
                return Err(EngineError::InvalidLayerSize {
                    expected: self.max_layer_bytes,
                    actual: layer.len(),
                });
            }
            if self.dequantizer.is_some() && !layer.len().is_multiple_of(Q4K_BLOCK_BYTES) {
                return Err(EngineError::LayerNotDequantAligned {
                    block_bytes: Q4K_BLOCK_BYTES,
                    actual: layer.len(),
                });
            }

            let start_event = CudaEvent::new(Arc::clone(&self.device))?;
            let end_event = CudaEvent::new(Arc::clone(&self.device))?;

            start_event.record(&self.transfer_stream)?;

            let slot_idx = i % 2;

            // Wait for previous compute on this slot (layer i-2) to finish before writing new layer
            if i >= 2 {
                self.compute_done[slot_idx].stream_wait(&self.transfer_stream)?;
            }

            // Async copy layer bytes into slot i mod 2 on the TRANSFER stream
            self.slots[slot_idx].copy_from_host_async(&self.transfer_stream, layer)?;

            // Record copy_done[i mod 2] event on TRANSFER stream
            self.copy_done[slot_idx].record(&self.transfer_stream)?;

            // COMPUTE stream waits on copy_done[i mod 2]
            self.copy_done[slot_idx].stream_wait(&self.compute_stream)?;

            // Compute stage: dequantize the layer on the COMPUTE stream when a
            // dequantizer is configured; otherwise keep the historical stub
            // behavior (record compute-done only).
            if let (Some(dequantizer), Some(out_slots)) = (&self.dequantizer, &self.dequant_out) {
                dequantizer.launch(
                    &self.compute_stream,
                    &self.slots[slot_idx],
                    &out_slots[slot_idx],
                )?;
            }

            // Record compute_done[i mod 2] event on COMPUTE stream
            self.compute_done[slot_idx].record(&self.compute_stream)?;

            end_event.record(&self.compute_stream)?;

            layer_starts.push(start_event);
            layer_ends.push(end_event);
        }

        // Synchronize streams only at the very end
        self.transfer_stream.sync()?;
        self.compute_stream.sync()?;

        let elapsed_ms = start_instant.elapsed().as_secs_f64() * 1000.0;

        let mut layer_elapsed_ms = Vec::with_capacity(layers.len());
        for (start, end) in layer_starts.iter().zip(layer_ends.iter()) {
            layer_elapsed_ms.push(end.elapsed_ms(start)?);
        }

        Ok(PipelineStats {
            layers: layers.len(),
            elapsed_ms,
            layer_elapsed_ms,
        })
    }

    /// Returns a reference to the underlying `CudaDevice`.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Returns a reference to the transfer `CudaStream`.
    pub fn transfer_stream(&self) -> &CudaStream {
        &self.transfer_stream
    }

    /// Returns a reference to the compute `CudaStream`.
    pub fn compute_stream(&self) -> &CudaStream {
        &self.compute_stream
    }

    /// Returns references to the two `DeviceBuffer` ping-pong slots.
    pub fn slots(&self) -> &[DeviceBuffer; 2] {
        &self.slots
    }

    /// Returns a reference to the ping-pong `DeviceBuffer` slot for index `slot_idx`.
    pub fn slot(&self, slot_idx: usize) -> &DeviceBuffer {
        &self.slots[slot_idx % 2]
    }

    /// Returns the maximum layer bytes capacity per slot.
    pub fn max_layer_bytes(&self) -> usize {
        self.max_layer_bytes
    }

    /// Returns whether a Q4_K_M dequantizer is configured for the compute stage.
    pub fn dequantizer_enabled(&self) -> bool {
        self.dequantizer.is_some()
    }

    /// Returns the dequantized-float output buffer for ping-pong slot
    /// `slot_idx`, or `None` when no dequantizer is configured.
    pub fn dequant_out_slot(&self, slot_idx: usize) -> Option<&DeviceBuffer> {
        self.dequant_out.as_ref().map(|out| &out[slot_idx % 2])
    }
}
