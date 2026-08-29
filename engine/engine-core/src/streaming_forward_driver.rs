//! Streaming Forward Driver for Large Model Scaling (Phase 13, Sub-change 13.2).
//!
//! Streams transformer layer weights dynamically over PCIe 4.0 DMA using a dual-stream
//! ping-pong double buffer (`compute_stream`, `transfer_stream`) synchronized with CUDA events.
//! Ensures total GPU memory remains bounded (< 2.0 GB) for arbitrarily large models (14B/32B).

use crate::error::EngineError;
use crate::forward_cpu::{Tensor, embed_lookup};
use crate::forward_driver::{MAX_SPEC_K, bank_tensor, f32_norm, f32_norm_opt, ggml_to_gemv};
use crate::layer_double_buffer::{HostLayerWeights, LayerDoubleBuffer, LayerTensorSizes};
use engine_cuda::{
    BatchedGEMM, CudaDevice, CudaEvent, CudaStream, DeviceBuffer, MODE_NORM, MODE_ROPE,
    MODE_SWIGLU, MultiFormatGEMV, NormRope, PagedAttention, PagedKvGpu, PagedKvLayout,
};
use engine_io::{GgufReader, LoadedPinned, ModelConfig};
use std::sync::Arc;

/// Parsed host layer matrix weights referencing pinned host DRAM memory.
pub struct PinnedLayerWeights<'a> {
    pub wq: Tensor<'a>,
    pub wk: Tensor<'a>,
    pub wv: Tensor<'a>,
    pub wo: Tensor<'a>,
    pub wgate: Tensor<'a>,
    pub wup: Tensor<'a>,
    pub wdown: Tensor<'a>,
}

/// Small resident RMSNorm weight buffers per layer on GPU (< 500 KB total).
pub struct LayerNormsDev {
    pub an_dev: DeviceBuffer,
    pub qn_dev: DeviceBuffer,
    pub kn_dev: DeviceBuffer,
    pub fn_dev: DeviceBuffer,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn alloc_dev(dev: &Arc<CudaDevice>, n_floats: usize) -> Result<DeviceBuffer, EngineError> {
    Ok(DeviceBuffer::alloc(Arc::clone(dev), n_floats * 4)?)
}

fn upload_f32(
    stream: &CudaStream,
    dev: &Arc<CudaDevice>,
    v: &[f32],
) -> Result<DeviceBuffer, EngineError> {
    let b = DeviceBuffer::alloc(Arc::clone(dev), v.len() * 4)?;
    b.copy_from_host(stream, &f32_bytes(v))?;
    Ok(b)
}

fn upload_bytes(
    stream: &CudaStream,
    dev: &Arc<CudaDevice>,
    b: &[u8],
) -> Result<DeviceBuffer, EngineError> {
    let buf = DeviceBuffer::alloc(Arc::clone(dev), b.len())?;
    buf.copy_from_host(stream, b)?;
    Ok(buf)
}

/// Production Layer Streaming Forward Driver for models exceeding physical VRAM.
pub struct StreamingForwardDriver<'a> {
    _device: Arc<CudaDevice>,
    compute_stream: CudaStream,
    transfer_stream: CudaStream,
    events_transfer_done: [CudaEvent; 2],
    events_compute_done: [CudaEvent; 2],
    gemv: MultiFormatGEMV,
    _batched_gemm: BatchedGEMM,
    nr: NormRope,
    pkv: PagedKvGpu,
    pa: PagedAttention,
    layout: PagedKvLayout,
    emb: Tensor<'a>,
    lm_head_weight: Tensor<'a>,
    head_norm: Vec<f32>,
    pinned_layers: Vec<PinnedLayerWeights<'a>>,
    layer_norms: Vec<LayerNormsDev>,
    double_buffer: LayerDoubleBuffer,
    // Per-layer resident KV pools
    layer_kv_pools: Vec<DeviceBuffer>,
    // Reusable single-token decode scratch buffers
    pos_dev: DeviceBuffer,
    zq: DeviceBuffer,
    zk: DeviceBuffer,
    zh: DeviceBuffer,
    _zhd: DeviceBuffer,
    zff: DeviceBuffer,
    bt_dev: DeviceBuffer,
    x_dev: DeviceBuffer,
    input_norm_dev: DeviceBuffer,
    q_dev: DeviceBuffer,
    k_dev: DeviceBuffer,
    v_dev: DeviceBuffer,
    _head_dev: DeviceBuffer,
    attn_dev: DeviceBuffer,
    op_dev: DeviceBuffer,
    h1_dev: DeviceBuffer,
    ffin_dev: DeviceBuffer,
    gate_dev: DeviceBuffer,
    up_dev: DeviceBuffer,
    proj_dev: DeviceBuffer,
    // Speculative working buffers (max_k = 8)
    _spec_x_dev: DeviceBuffer,
    _spec_norm_dev: DeviceBuffer,
    _spec_q_dev: DeviceBuffer,
    _spec_k_dev: DeviceBuffer,
    _spec_v_dev: DeviceBuffer,
    _spec_attn_dev: DeviceBuffer,
    _spec_op_dev: DeviceBuffer,
    _spec_ffin_dev: DeviceBuffer,
    _spec_gate_dev: DeviceBuffer,
    _spec_up_dev: DeviceBuffer,
    _spec_proj_dev: DeviceBuffer,
    _spec_zh: DeviceBuffer,
    _spec_zq: DeviceBuffer,
    _spec_zk: DeviceBuffer,
    _spec_zff: DeviceBuffer,
    // Dims
    h: usize,
    hd: usize,
    nh: usize,
    nkv: usize,
    hff: usize,
    _qdim: usize,
    _kvd: usize,
    n_rot: usize,
    eps: f32,
    base: f32,
    n_layer: usize,
    pub pos: usize,
    has_qk_norm: bool,
}

impl<'a> StreamingForwardDriver<'a> {
    /// Constructs a new StreamingForwardDriver by allocating the ping-pong double buffer and KV pool.
    pub fn new(
        reader: &GgufReader,
        pinned: &'a LoadedPinned,
        cfg: &ModelConfig,
        capacity_tokens: usize,
    ) -> Result<Self, EngineError> {
        let capacity = capacity_tokens.max(1);
        let device = CudaDevice::new(0)?;
        let compute_stream = CudaStream::new_with_priority(Arc::clone(&device), -1)?;
        let transfer_stream = CudaStream::new_with_priority(Arc::clone(&device), 0)?;
        let events_transfer_done = [
            CudaEvent::new(Arc::clone(&device))?,
            CudaEvent::new(Arc::clone(&device))?,
        ];
        let events_compute_done = [
            CudaEvent::new(Arc::clone(&device))?,
            CudaEvent::new(Arc::clone(&device))?,
        ];

        let gemv = MultiFormatGEMV::new(Arc::clone(&device))?;
        let batched_gemm = BatchedGEMM::new(Arc::clone(&device))?;
        let nr = NormRope::new(Arc::clone(&device))?;
        let pkv = PagedKvGpu::new(Arc::clone(&device))?;
        let pa = PagedAttention::new(Arc::clone(&device))?;

        let h = cfg.hidden_size as usize;
        let hd = cfg.head_dim as usize;
        let nh = cfg.n_head as usize;
        let nkv = cfg.n_head_kv as usize;
        let hff = cfg.intermediate_size as usize;
        let eps = cfg.rms_norm_eps;
        let base = cfg.rope_freq_base;
        let n_rot = hd;
        let qdim = nh * hd;
        let kvd = nkv * hd;
        let n_layer = cfg.n_layer as usize;

        let layout = PagedKvLayout {
            n_blocks: 1,
            block_tokens: capacity,
            row_len: kvd,
            data_type: engine_cuda::KvDataType::F32,
        };

        let emb = bank_tensor(reader, pinned, "token_embd.weight")?;
        let lm_head_weight = if reader.get_tensor("output.weight").is_some() {
            bank_tensor(reader, pinned, "output.weight")?
        } else {
            emb
        };
        let head_norm = f32_norm(pinned, "output_norm.weight")?;

        // Parse all layer weights in pinned host DRAM and upload small norm vectors
        let has_qk_norm = pinned.tensor("blk.0.attn_q_norm.weight").is_some();
        let mut pinned_layers = Vec::with_capacity(n_layer);
        let mut layer_norms = Vec::with_capacity(n_layer);
        let mut max_sizes = LayerTensorSizes {
            wq_bytes: 0,
            wk_bytes: 0,
            wv_bytes: 0,
            wo_bytes: 0,
            wgate_bytes: 0,
            wup_bytes: 0,
            wdown_bytes: 0,
        };

        for l in 0..n_layer {
            let wq = bank_tensor(reader, pinned, &format!("blk.{l}.attn_q.weight"))?;
            let wk = bank_tensor(reader, pinned, &format!("blk.{l}.attn_k.weight"))?;
            let wv = bank_tensor(reader, pinned, &format!("blk.{l}.attn_v.weight"))?;
            let wo = bank_tensor(reader, pinned, &format!("blk.{l}.attn_output.weight"))?;
            let wgate = bank_tensor(reader, pinned, &format!("blk.{l}.ffn_gate.weight"))?;
            let wup = bank_tensor(reader, pinned, &format!("blk.{l}.ffn_up.weight"))?;
            let wdown = bank_tensor(reader, pinned, &format!("blk.{l}.ffn_down.weight"))?;

            let an = f32_norm(pinned, &format!("blk.{l}.attn_norm.weight"))?;
            let qn = f32_norm_opt(pinned, &format!("blk.{l}.attn_q_norm.weight"), hd);
            let kn = f32_norm_opt(pinned, &format!("blk.{l}.attn_k_norm.weight"), hd);
            let fn_norm = f32_norm(pinned, &format!("blk.{l}.ffn_norm.weight"))?;

            let an_dev = upload_f32(&compute_stream, &device, &an)?;
            let qn_dev = upload_f32(&compute_stream, &device, &qn)?;
            let kn_dev = upload_f32(&compute_stream, &device, &kn)?;
            let fn_dev = upload_f32(&compute_stream, &device, &fn_norm)?;

            layer_norms.push(LayerNormsDev {
                an_dev,
                qn_dev,
                kn_dev,
                fn_dev,
            });

            max_sizes.wq_bytes = max_sizes.wq_bytes.max(wq.data.len());
            max_sizes.wk_bytes = max_sizes.wk_bytes.max(wk.data.len());
            max_sizes.wv_bytes = max_sizes.wv_bytes.max(wv.data.len());
            max_sizes.wo_bytes = max_sizes.wo_bytes.max(wo.data.len());
            max_sizes.wgate_bytes = max_sizes.wgate_bytes.max(wgate.data.len());
            max_sizes.wup_bytes = max_sizes.wup_bytes.max(wup.data.len());
            max_sizes.wdown_bytes = max_sizes.wdown_bytes.max(wdown.data.len());

            pinned_layers.push(PinnedLayerWeights {
                wq,
                wk,
                wv,
                wo,
                wgate,
                wup,
                wdown,
            });
        }

        // Allocate GPU layer double buffer for matrix weights
        let double_buffer = LayerDoubleBuffer::new(Arc::clone(&device), &max_sizes)?;

        // Allocate resident KV pools for each layer
        let mut layer_kv_pools = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            let pool_dev = upload_bytes(
                &compute_stream,
                &device,
                &vec![0u8; layout.floats_total() * 4],
            )?;
            layer_kv_pools.push(pool_dev);
        }

        // Reusable scratch allocations
        let pos_dev = upload_bytes(&compute_stream, &device, &0u32.to_le_bytes())?;
        let zq = upload_f32(&compute_stream, &device, &vec![0.0f32; qdim])?;
        let zk = upload_f32(&compute_stream, &device, &vec![0.0f32; kvd])?;
        let zh = upload_f32(&compute_stream, &device, &vec![0.0f32; h])?;
        let zhd = upload_f32(&compute_stream, &device, &vec![0.0f32; hd])?;
        let zff = upload_f32(&compute_stream, &device, &vec![0.0f32; hff])?;
        let bt_dev = upload_bytes(&compute_stream, &device, &0u32.to_le_bytes())?;

        let x_dev = alloc_dev(&device, h)?;
        let input_norm_dev = alloc_dev(&device, h)?;
        let q_dev = alloc_dev(&device, qdim)?;
        let k_dev = alloc_dev(&device, kvd)?;
        let v_dev = alloc_dev(&device, kvd)?;
        let head_dev = alloc_dev(&device, hd)?;
        let attn_dev = alloc_dev(&device, qdim)?;
        let op_dev = alloc_dev(&device, h)?;
        let h1_dev = alloc_dev(&device, h)?;
        let ffin_dev = alloc_dev(&device, h)?;
        let gate_dev = alloc_dev(&device, hff)?;
        let up_dev = alloc_dev(&device, hff)?;
        let proj_dev = alloc_dev(&device, hff)?;

        // Speculative preallocated working buffers
        let spec_x_dev = alloc_dev(&device, MAX_SPEC_K * h)?;
        let spec_norm_dev = alloc_dev(&device, MAX_SPEC_K * h)?;
        let spec_q_dev = alloc_dev(&device, MAX_SPEC_K * qdim)?;
        let spec_k_dev = alloc_dev(&device, MAX_SPEC_K * kvd)?;
        let spec_v_dev = alloc_dev(&device, MAX_SPEC_K * kvd)?;
        let spec_attn_dev = alloc_dev(&device, MAX_SPEC_K * qdim)?;
        let spec_op_dev = alloc_dev(&device, MAX_SPEC_K * h)?;
        let spec_ffin_dev = alloc_dev(&device, MAX_SPEC_K * h)?;
        let spec_gate_dev = alloc_dev(&device, MAX_SPEC_K * hff)?;
        let spec_up_dev = alloc_dev(&device, MAX_SPEC_K * hff)?;
        let spec_proj_dev = alloc_dev(&device, MAX_SPEC_K * hff)?;

        let spec_zh = upload_f32(&compute_stream, &device, &vec![0.0f32; MAX_SPEC_K * h])?;
        let spec_zq = upload_f32(&compute_stream, &device, &vec![0.0f32; MAX_SPEC_K * qdim])?;
        let spec_zk = upload_f32(&compute_stream, &device, &vec![0.0f32; MAX_SPEC_K * kvd])?;
        let spec_zff = upload_f32(&compute_stream, &device, &vec![0.0f32; MAX_SPEC_K * hff])?;

        compute_stream.sync()?;
        transfer_stream.sync()?;

        Ok(Self {
            _device: device,
            compute_stream,
            transfer_stream,
            events_transfer_done,
            events_compute_done,
            gemv,
            _batched_gemm: batched_gemm,
            nr,
            pkv,
            pa,
            layout,
            emb,
            lm_head_weight,
            head_norm,
            pinned_layers,
            layer_norms,
            double_buffer,
            layer_kv_pools,
            pos_dev,
            zq,
            zk,
            zh,
            _zhd: zhd,
            zff,
            bt_dev,
            x_dev,
            input_norm_dev,
            q_dev,
            k_dev,
            v_dev,
            _head_dev: head_dev,
            attn_dev,
            op_dev,
            h1_dev,
            ffin_dev,
            gate_dev,
            up_dev,
            proj_dev,
            _spec_x_dev: spec_x_dev,
            _spec_norm_dev: spec_norm_dev,
            _spec_q_dev: spec_q_dev,
            _spec_k_dev: spec_k_dev,
            _spec_v_dev: spec_v_dev,
            _spec_attn_dev: spec_attn_dev,
            _spec_op_dev: spec_op_dev,
            _spec_ffin_dev: spec_ffin_dev,
            _spec_gate_dev: spec_gate_dev,
            _spec_up_dev: spec_up_dev,
            _spec_proj_dev: spec_proj_dev,
            _spec_zh: spec_zh,
            _spec_zq: spec_zq,
            _spec_zk: spec_zk,
            _spec_zff: spec_zff,
            h,
            hd,
            nh,
            nkv,
            hff,
            _qdim: qdim,
            _kvd: kvd,
            n_rot,
            eps,
            base,
            n_layer,
            pos: 0,
            has_qk_norm,
        })
    }

    /// Single-token forward step over streaming layers returning last hidden activation.
    pub fn forward_token(&mut self, token: u32) -> Result<Vec<f32>, EngineError> {
        let p = self.pos;
        if p >= self.layout.block_tokens {
            return Err(engine_cuda::CudaError::InvalidSize {
                expected: self.layout.block_tokens,
                actual: p + 1,
            }
            .into());
        }

        let x_host = embed_lookup(&self.emb, token as usize);
        self.x_dev
            .copy_from_host(&self.compute_stream, &f32_bytes(&x_host))?;
        let pos_bytes = (p as u32).to_le_bytes();
        self.pos_dev
            .copy_from_host(&self.compute_stream, &pos_bytes)?;

        // Prime pipeline: transfer layer 0 into slot 0
        let hw0 = extract_host_weights(&self.pinned_layers[0]);
        self.double_buffer
            .copy_layer_async(0, &hw0, &self.transfer_stream)?;
        self.events_transfer_done[0].record(&self.transfer_stream)?;

        for l in 0..self.n_layer {
            let curr_slot_idx = l % 2;
            let next_slot_idx = (l + 1) % 2;

            // Wait for transfer of current layer into curr_slot_idx before compute
            self.events_transfer_done[curr_slot_idx].stream_wait(&self.compute_stream)?;

            // 1. Asynchronously prefetch layer L+1 on transfer_stream while layer L computes
            if l + 1 < self.n_layer {
                if l > 0 {
                    self.events_compute_done[next_slot_idx].stream_wait(&self.transfer_stream)?;
                }
                let hw_next = extract_host_weights(&self.pinned_layers[l + 1]);
                self.double_buffer
                    .copy_layer_async(next_slot_idx, &hw_next, &self.transfer_stream)?;
                self.events_transfer_done[next_slot_idx].record(&self.transfer_stream)?;
            }

            // 2. Execute layer L forward compute on compute_stream
            let slot = self.double_buffer.slot(curr_slot_idx);
            let pool_dev = &self.layer_kv_pools[l];
            let norms = &self.layer_norms[l];
            let layer_info = &self.pinned_layers[l];

            // a. Input RMSNorm
            self.nr.launch(
                &self.compute_stream,
                &self.x_dev,
                &self.zh,
                &norms.an_dev,
                &self.zh,
                &self.input_norm_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                MODE_NORM,
            )?;

            // b. Q, K, V Projections
            self.gemv.gemv(
                &self.compute_stream,
                ggml_to_gemv(layer_info.wq.ty)?,
                &slot.wq_dev,
                &self.input_norm_dev,
                &self.q_dev,
                layer_info.wq.ne0,
                layer_info.wq.ne1,
            )?;
            self.gemv.gemv(
                &self.compute_stream,
                ggml_to_gemv(layer_info.wk.ty)?,
                &slot.wk_dev,
                &self.input_norm_dev,
                &self.k_dev,
                layer_info.wk.ne0,
                layer_info.wk.ne1,
            )?;
            self.gemv.gemv(
                &self.compute_stream,
                ggml_to_gemv(layer_info.wv.ty)?,
                &slot.wv_dev,
                &self.input_norm_dev,
                &self.v_dev,
                layer_info.wv.ne0,
                layer_info.wv.ne1,
            )?;

            // c. Per-head Q/K RMSNorm + RoPE
            let qk_mode = if self.has_qk_norm {
                MODE_NORM | MODE_ROPE
            } else {
                MODE_ROPE
            };
            self.nr.launch_with_pos_ptr(
                &self.compute_stream,
                &self.q_dev,
                &self.zq,
                &norms.qn_dev,
                &self.zq,
                &self.q_dev,
                self.eps,
                self.hd,
                self.n_rot,
                self.base,
                0,
                qk_mode,
                Some(&self.pos_dev),
            )?;
            self.nr.launch_with_pos_ptr(
                &self.compute_stream,
                &self.k_dev,
                &self.zk,
                &norms.kn_dev,
                &self.zk,
                &self.k_dev,
                self.eps,
                self.hd,
                self.n_rot,
                self.base,
                0,
                qk_mode,
                Some(&self.pos_dev),
            )?;

            // d. Append K, V to resident layer KV pool
            self.pkv.append_kv(
                &self.compute_stream,
                &self.layout,
                pool_dev,
                &self.k_dev,
                &self.v_dev,
                &self.bt_dev,
                p,
                1,
            )?;

            // e. PagedAttention
            self.pa.launch(
                &self.compute_stream,
                &self.q_dev,
                pool_dev,
                &self.bt_dev,
                &self.attn_dev,
                self.nh,
                self.nkv,
                self.hd,
                self.layout.block_tokens,
                p + 1,
                p,
                true,
            )?;

            // f. Output projection + Residual
            self.gemv.gemv(
                &self.compute_stream,
                ggml_to_gemv(layer_info.wo.ty)?,
                &slot.wo_dev,
                &self.attn_dev,
                &self.op_dev,
                layer_info.wo.ne0,
                layer_info.wo.ne1,
            )?;
            self.nr.launch(
                &self.compute_stream,
                &self.op_dev,
                &self.x_dev,
                &self.zh,
                &self.zh,
                &self.h1_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                0,
            )?;

            // g. FFN RMSNorm
            self.nr.launch(
                &self.compute_stream,
                &self.h1_dev,
                &self.zh,
                &norms.fn_dev,
                &self.zh,
                &self.ffin_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                MODE_NORM,
            )?;

            // h. FFN gate & up projections
            self.gemv.gemv(
                &self.compute_stream,
                ggml_to_gemv(layer_info.wgate.ty)?,
                &slot.wgate_dev,
                &self.ffin_dev,
                &self.gate_dev,
                layer_info.wgate.ne0,
                layer_info.wgate.ne1,
            )?;
            self.gemv.gemv(
                &self.compute_stream,
                ggml_to_gemv(layer_info.wup.ty)?,
                &slot.wup_dev,
                &self.ffin_dev,
                &self.up_dev,
                layer_info.wup.ne0,
                layer_info.wup.ne1,
            )?;

            // i. SwiGLU
            self.nr.launch(
                &self.compute_stream,
                &self.gate_dev,
                &self.zff,
                &self.zff,
                &self.up_dev,
                &self.gate_dev,
                self.eps,
                self.hff,
                0,
                self.base,
                0,
                MODE_SWIGLU,
            )?;

            // j. FFN down projection + Residual
            self.gemv.gemv(
                &self.compute_stream,
                ggml_to_gemv(layer_info.wdown.ty)?,
                &slot.wdown_dev,
                &self.gate_dev,
                &self.proj_dev,
                layer_info.wdown.ne0,
                layer_info.wdown.ne1,
            )?;
            self.nr.launch(
                &self.compute_stream,
                &self.proj_dev,
                &self.h1_dev,
                &self.zh,
                &self.zh,
                &self.x_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                0,
            )?;

            // 3. Record compute done for curr_slot_idx
            self.events_compute_done[curr_slot_idx].record(&self.compute_stream)?;
        }

        let mut x_host_out = vec![0u8; self.h * 4];
        self.x_dev
            .copy_to_host(&self.compute_stream, &mut x_host_out)?;
        self.pos += 1;
        self.compute_stream.sync()?;
        self.transfer_stream.sync()?;

        let last_hidden: Vec<f32> = x_host_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(last_hidden)
    }

    /// Single token decode returning next-token logits.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>, EngineError> {
        let last_hidden = self.forward_token(token)?;
        Ok(self.lm_head(&last_hidden))
    }

    /// Prefill sequence of tokens over streaming layer pipeline, computing logits only for the final position.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let mut last_hidden = Vec::new();
        for &tok in tokens {
            last_hidden = self.forward_token(tok)?;
        }
        Ok(self.lm_head(&last_hidden))
    }

    /// Returns the number of transformer layers in this model.
    pub fn n_layers(&self) -> usize {
        self.n_layer
    }

    /// Current token position in resident KV pool.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Returns the active VRAM footprint for weights, KV cache, and scratch buffers.
    pub fn vram_footprint(&self) -> crate::forward_driver::VramFootprint {
        let pingpong_bytes = self.double_buffer.total_vram_bytes();
        let kv_pool_bytes = self.layer_kv_pools.iter().map(|p| p.size()).sum();
        let activations_bytes = self.x_dev.size()
            + self.input_norm_dev.size()
            + self.q_dev.size()
            + self.k_dev.size()
            + self.v_dev.size()
            + self.attn_dev.size()
            + self.op_dev.size()
            + self.h1_dev.size()
            + self.ffin_dev.size()
            + self.gate_dev.size()
            + self.up_dev.size()
            + self.proj_dev.size();

        crate::forward_driver::VramFootprint {
            pingpong_bytes,
            kv_pool_bytes,
            activations_bytes,
            logits_bytes: 0,
        }
    }

    /// Computes final output logits using the embedding weight matrix as tied LM head.
    fn lm_head(&self, x: &[f32]) -> Vec<f32> {
        crate::forward_cpu::logits_from_hidden(&self.lm_head_weight, &self.head_norm, x, self.eps)
    }
}

fn extract_host_weights<'a>(p: &'a PinnedLayerWeights<'a>) -> HostLayerWeights<'a> {
    HostLayerWeights {
        wq_data: p.wq.data,
        wk_data: p.wk.data,
        wv_data: p.wv.data,
        wo_data: p.wo.data,
        wgate_data: p.wgate.data,
        wup_data: p.wup.data,
        wdown_data: p.wdown.data,
    }
}
