use crate::types::TensorInfo;
use std::collections::BTreeMap;

/// Classifies whether a tensor belongs to a transformer layer (e.g. "blk.N.*").
///
/// Returns `Some(N)` if the tensor name matches the `blk.N.*` pattern, or `None` otherwise.
pub fn classify_layer(name: &str) -> Option<usize> {
    if let Some(rest) = name.strip_prefix("blk.") {
        let dot_pos = rest.find('.')?;
        let layer_str = &rest[..dot_pos];
        layer_str.parse::<usize>().ok()
    } else {
        None
    }
}

/// Index of model tensors organized by layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayerIndex {
    by_layer: BTreeMap<usize, Vec<TensorInfo>>,
    non_layer: Vec<TensorInfo>,
    all: Vec<TensorInfo>,
}

impl LayerIndex {
    /// Builds a `LayerIndex` from a list of `TensorInfo`.
    pub fn new(tensors: &[TensorInfo]) -> Self {
        let mut by_layer: BTreeMap<usize, Vec<TensorInfo>> = BTreeMap::new();
        let mut non_layer = Vec::new();

        for t in tensors {
            if let Some(layer_idx) = classify_layer(&t.name) {
                by_layer.entry(layer_idx).or_default().push(t.clone());
            } else {
                non_layer.push(t.clone());
            }
        }

        Self {
            by_layer,
            non_layer,
            all: tensors.to_vec(),
        }
    }

    /// List of all detected layer indices in ascending order.
    pub fn layers(&self) -> Vec<usize> {
        self.by_layer.keys().copied().collect()
    }

    /// All tensors in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.all
    }

    /// Tensors belonging to a specific layer index `idx`.
    pub fn by_layer(&self, idx: usize) -> Option<&[TensorInfo]> {
        self.by_layer.get(&idx).map(|v| v.as_slice())
    }

    /// Tensors that do not belong to any `blk.N` layer (e.g. token_embd, output, norm).
    pub fn non_layer_tensors(&self) -> &[TensorInfo] {
        &self.non_layer
    }
}
