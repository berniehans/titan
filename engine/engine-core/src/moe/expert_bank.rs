//! Host expert memory banks and pinned slice allocator (Phase 7.2).
//!
//! Stores routed MoE expert weights in pinned host memory (or pageable fallback)
//! structured as per-(layer, expert) slices for zero-copy PCIe DMA transfer
//! and concurrent CPU MoE execution.

use crate::error::EngineError;
use engine_cuda::PinnedHost;
use std::collections::HashMap;

/// Metadata describing a specific expert tensor within the host bank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertTensorDesc {
    /// Transformer layer index.
    pub layer: usize,
    /// Expert index within this layer (0..num_experts).
    pub expert_id: usize,
    /// Tensor role/name (e.g. "gate_ex", "up_ex", "down_ex").
    pub name: String,
    /// Byte offset within the expert's contiguous slice.
    pub offset_in_expert: usize,
    /// Size of this tensor in bytes.
    pub size_bytes: usize,
}

/// Contiguous host memory buffer holding all expert weights, either pinned or pageable.
enum HostBankStorage {
    /// Pinned host memory (locked RAM) enabling asynchronous PCIe DMA copies.
    Pinned(PinnedHost),
    /// Pageable fallback storage when pinned host memory allocation is unavailable.
    Pageable(Vec<u8>),
}

/// Structured host memory bank for MoE expert weights.
pub struct HostExpertBank {
    storage: HostBankStorage,
    is_pinned: bool,
    n_layers: usize,
    n_experts_per_layer: usize,
    expert_slice_size_bytes: usize,
    total_bytes: usize,
    tensor_map: HashMap<(usize, usize, String), (usize, usize)>, // (layer, expert, name) -> (offset, size)
}

impl HostExpertBank {
    /// Allocates a new `HostExpertBank`.
    ///
    /// Tries allocating pinned host memory first. If pinned allocation fails,
    /// falls back to pageable memory and sets `is_pinned = false`.
    pub fn allocate(
        n_layers: usize,
        n_experts_per_layer: usize,
        expert_slice_size_bytes: usize,
        prefer_pinned: bool,
    ) -> Result<Self, EngineError> {
        let total_experts = n_layers * n_experts_per_layer;
        let total_bytes = total_experts * expert_slice_size_bytes;

        let (storage, is_pinned) = if prefer_pinned {
            match PinnedHost::alloc(total_bytes) {
                Ok(pinned) => (HostBankStorage::Pinned(pinned), true),
                Err(_) => {
                    // Fallback to pageable
                    let pageable = vec![0u8; total_bytes];
                    (HostBankStorage::Pageable(pageable), false)
                }
            }
        } else {
            let pageable = vec![0u8; total_bytes];
            (HostBankStorage::Pageable(pageable), false)
        };

        Ok(Self {
            storage,
            is_pinned,
            n_layers,
            n_experts_per_layer,
            expert_slice_size_bytes,
            total_bytes,
            tensor_map: HashMap::new(),
        })
    }

    /// Whether this host bank is backed by pinned (page-locked) memory.
    pub fn is_pinned(&self) -> bool {
        self.is_pinned
    }

    /// Total size in bytes of the host bank.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Size in bytes of a single expert's contiguous slice.
    pub fn expert_slice_size(&self) -> usize {
        self.expert_slice_size_bytes
    }

    /// Number of transformer layers.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Number of routed experts per layer.
    pub fn n_experts_per_layer(&self) -> usize {
        self.n_experts_per_layer
    }

    /// Computes the linear byte offset for `(layer, expert_id)`.
    fn expert_offset(&self, layer: usize, expert_id: usize) -> Option<usize> {
        if layer >= self.n_layers || expert_id >= self.n_experts_per_layer {
            return None;
        }
        let global_idx = layer * self.n_experts_per_layer + expert_id;
        Some(global_idx * self.expert_slice_size_bytes)
    }

    /// Returns an immutable slice view over the raw bytes of an expert.
    pub fn expert_slice(&self, layer: usize, expert_id: usize) -> Option<&[u8]> {
        let offset = self.expert_offset(layer, expert_id)?;
        let end = offset + self.expert_slice_size_bytes;
        match &self.storage {
            HostBankStorage::Pinned(p) => Some(&p.as_slice()[offset..end]),
            HostBankStorage::Pageable(v) => Some(&v[offset..end]),
        }
    }

    /// Returns a mutable slice view over the raw bytes of an expert.
    pub fn expert_slice_mut(&mut self, layer: usize, expert_id: usize) -> Option<&mut [u8]> {
        let offset = self.expert_offset(layer, expert_id)?;
        let end = offset + self.expert_slice_size_bytes;
        match &mut self.storage {
            HostBankStorage::Pinned(p) => Some(&mut p.as_mut_slice()[offset..end]),
            HostBankStorage::Pageable(v) => Some(&mut v[offset..end]),
        }
    }

    /// Registers and populates a named tensor within `(layer, expert_id)`.
    pub fn write_expert_tensor(
        &mut self,
        layer: usize,
        expert_id: usize,
        tensor_name: &str,
        offset_in_expert: usize,
        data: &[u8],
    ) -> Result<(), EngineError> {
        let expert_slice = self.expert_slice_mut(layer, expert_id).ok_or_else(|| {
            EngineError::Validation(format!("Invalid layer {layer} / expert {expert_id}"))
        })?;

        if offset_in_expert + data.len() > expert_slice.len() {
            return Err(EngineError::Validation(format!(
                "Tensor {} size {} at offset {} exceeds expert slice size {}",
                tensor_name,
                data.len(),
                offset_in_expert,
                expert_slice.len()
            )));
        }

        expert_slice[offset_in_expert..offset_in_expert + data.len()].copy_from_slice(data);
        self.tensor_map.insert(
            (layer, expert_id, tensor_name.to_string()),
            (offset_in_expert, data.len()),
        );
        Ok(())
    }

    /// Reads a registered named tensor within `(layer, expert_id)`.
    pub fn get_expert_tensor(
        &self,
        layer: usize,
        expert_id: usize,
        tensor_name: &str,
    ) -> Option<&[u8]> {
        let (offset_in_expert, size) =
            self.tensor_map
                .get(&(layer, expert_id, tensor_name.to_string()))?;
        let expert_slice = self.expert_slice(layer, expert_id)?;
        Some(&expert_slice[*offset_in_expert..*offset_in_expert + *size])
    }
}
