use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Instant;

use engine_cuda::PinnedHost;

use crate::error::GgufError;
use crate::reader::GgufReader;

/// In-memory layout and accounting of tensor byte spans within a GGUF model file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedLayout {
    total_size_bytes: u64,
    tensor_spans: HashMap<String, (u64, u64)>,
    layer_spans: BTreeMap<usize, Vec<(u64, u64)>>,
    layer_ranges: BTreeMap<usize, (u64, u64)>,
}

impl LoadedLayout {
    /// Computes the layout and accounting from a parsed GGUF reader.
    pub fn from_reader(reader: &GgufReader) -> Result<Self, GgufError> {
        let tensors = reader.tensor_infos();
        let mut total_size_bytes = 0u64;
        let mut tensor_spans = HashMap::with_capacity(tensors.len());

        for t in tensors {
            let end = t.offset.checked_add(t.size_bytes).ok_or_else(|| {
                GgufError::InvalidTensorShape(format!("Tensor '{}' offset + size overflow", t.name))
            })?;
            if end > total_size_bytes {
                total_size_bytes = end;
            }
            tensor_spans.insert(t.name.clone(), (t.offset, t.size_bytes));
        }

        let mut layer_spans: BTreeMap<usize, Vec<(u64, u64)>> = BTreeMap::new();
        let mut layer_ranges: BTreeMap<usize, (u64, u64)> = BTreeMap::new();

        let layer_index = reader.layer_index();
        for &layer_idx in &layer_index.layers() {
            if let Some(layer_tensors) = layer_index.by_layer(layer_idx) {
                if layer_tensors.is_empty() {
                    continue;
                }
                let mut spans = Vec::with_capacity(layer_tensors.len());
                let mut min_offset = u64::MAX;
                let mut max_end = 0u64;

                for lt in layer_tensors {
                    spans.push((lt.offset, lt.size_bytes));
                    if lt.offset < min_offset {
                        min_offset = lt.offset;
                    }
                    let end = lt.offset.saturating_add(lt.size_bytes);
                    if end > max_end {
                        max_end = end;
                    }
                }

                layer_spans.insert(layer_idx, spans);
                if min_offset <= max_end {
                    let total_layer_size = max_end - min_offset;
                    layer_ranges.insert(layer_idx, (min_offset, total_layer_size));
                }
            }
        }

        Ok(Self {
            total_size_bytes,
            tensor_spans,
            layer_spans,
            layer_ranges,
        })
    }

    /// Total size in bytes of the tensor data area.
    pub fn total_size_bytes(&self) -> u64 {
        self.total_size_bytes
    }

    /// Looks up a tensor span `(offset_into_blob, size_bytes)` by name.
    pub fn tensor_span(&self, name: &str) -> Option<(u64, u64)> {
        self.tensor_spans.get(name).copied()
    }

    /// Tensor spans `[(offset_into_blob, size_bytes), ...]` belonging to a given layer index.
    pub fn layer_spans(&self, layer: usize) -> Option<&[(u64, u64)]> {
        self.layer_spans.get(&layer).map(|v| v.as_slice())
    }

    /// Contiguous start offset and total byte size `(start_offset, total_size)` for a given layer index.
    pub fn layer_range(&self, layer: usize) -> Option<(u64, u64)> {
        self.layer_ranges.get(&layer).copied()
    }
}

/// Owned pinned host buffer holding full GGUF tensor data blob with layout accessors.
#[derive(Debug)]
pub struct LoadedPinned {
    host: PinnedHost,
    layout: LoadedLayout,
    gb_per_second: f64,
}

impl LoadedPinned {
    /// Total usable size in bytes of the loaded pinned memory buffer.
    pub fn total_size_bytes(&self) -> u64 {
        self.layout.total_size_bytes()
    }

    /// Borrow the entire pinned memory buffer as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        self.host.as_slice()
    }

    /// Looks up and borrows the slice of bytes corresponding to the given tensor name.
    pub fn tensor(&self, name: &str) -> Option<&[u8]> {
        let (offset, size) = self.layout.tensor_span(name)?;
        let start = offset as usize;
        let end = start.checked_add(size as usize)?;
        self.host.as_slice().get(start..end)
    }

    /// Looks up and borrows the contiguous slice of bytes for the given layer index.
    pub fn layer(&self, layer: usize) -> Option<&[u8]> {
        let (offset, size) = self.layout.layer_range(layer)?;
        let start = offset as usize;
        let end = start.checked_add(size as usize)?;
        self.host.as_slice().get(start..end)
    }

    /// Reference to the underlying `LoadedLayout`.
    pub fn layout(&self) -> &LoadedLayout {
        &self.layout
    }

    /// Reference to the underlying `PinnedHost`.
    pub fn host(&self) -> &PinnedHost {
        &self.host
    }

    /// Measured transfer throughput in gigabytes per second (GB/s).
    pub fn gb_per_second(&self) -> f64 {
        self.gb_per_second
    }
}

/// Loads the tensor data blob from a GGUF model file into page-locked (pinned) host memory.
///
/// Reads the tensor data blob from disk starting at `reader.tensor_data_offset()`,
/// allocates a `PinnedHost` buffer of the exact layout size, reads the bytes directly into pinned host memory,
/// and logs the transfer throughput metric in GB/s.
pub fn load_to_pinned<P: AsRef<Path>>(
    reader: &GgufReader,
    path: P,
) -> Result<LoadedPinned, GgufError> {
    let layout = LoadedLayout::from_reader(reader)?;
    let data_blob_size = layout.total_size_bytes();

    let start_time = Instant::now();

    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(reader.tensor_data_offset()))?;

    let mut host = PinnedHost::alloc(data_blob_size as usize)?;
    file.read_exact(host.as_mut_slice())?;

    let elapsed = start_time.elapsed();
    let seconds = elapsed.as_secs_f64();
    let gb_per_second = if seconds > 0.0 {
        (data_blob_size as f64) / (seconds * 1_000_000_000.0)
    } else {
        0.0
    };

    tracing::info!(
        bytes = data_blob_size,
        seconds = seconds,
        gb_per_second = gb_per_second,
        "Loaded GGUF tensor data into pinned host memory"
    );

    Ok(LoadedPinned {
        host,
        layout,
        gb_per_second,
    })
}
