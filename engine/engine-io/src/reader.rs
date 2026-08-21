use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use crate::error::GgufError;
use crate::layer::LayerIndex;
use crate::types::{GgmlType, GgufHeader, GgufType, GgufValue, TensorInfo};

/// Reader and parser for GGUF model files.
#[derive(Debug, Clone)]
pub struct GgufReader {
    header: GgufHeader,
    metadata: HashMap<String, GgufValue>,
    tensor_infos: Vec<TensorInfo>,
    layer_index: LayerIndex,
    tensor_data_offset: u64,
    alignment: u64,
}

impl GgufReader {
    /// Opens and parses a GGUF file from the specified path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        // 1. Magic
        let magic_bytes = read_exact_bytes::<_, 4>(&mut reader, "magic")?;
        if &magic_bytes != b"GGUF" {
            return Err(GgufError::InvalidMagic(magic_bytes));
        }

        // 2. Version
        let version = read_u32(&mut reader)?;
        if version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // 3. Tensor count (int64/uint64)
        let tensor_count = read_u64(&mut reader)?;

        // 4. Metadata KV count (int64/uint64)
        let metadata_kv_count = read_u64(&mut reader)?;

        let header = GgufHeader {
            magic: "GGUF".to_string(),
            version,
            tensor_count,
            metadata_kv_count,
        };

        // 5. Metadata KV pairs
        let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut reader)?;
            let val_type_raw = read_u32(&mut reader)?;
            let val_type = GgufType::try_from(val_type_raw)?;
            let val = read_value(&mut reader, val_type)?;
            metadata.insert(key, val);
        }

        // 6. Tensor infos
        let mut tensor_infos = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(&mut reader)?;
            let n_dims = read_u32(&mut reader)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut reader)?);
            }
            let tensor_type_raw = read_u32(&mut reader)?;
            let ggml_type = GgmlType::try_from(tensor_type_raw)?;
            let offset = read_u64(&mut reader)?;
            let size_bytes = compute_tensor_size(&dims, ggml_type, &name)?;

            tensor_infos.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
                size_bytes,
            });
        }

        // 7. Determine tensor data alignment and offset
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32().map(|x| x as u64).or_else(|| v.as_u64()))
            .unwrap_or(32);

        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::InvalidAlignment(alignment));
        }

        let current_pos = reader.stream_position()?;
        let tensor_data_offset = align_up(current_pos, alignment);

        // Validate tensor spans within file bounds
        for t in &tensor_infos {
            let total_end = tensor_data_offset
                .checked_add(t.offset)
                .and_then(|start| start.checked_add(t.size_bytes))
                .ok_or_else(|| GgufError::TensorOutOfBounds {
                    name: t.name.clone(),
                    offset: t.offset,
                    size: t.size_bytes,
                    file_size,
                })?;

            if total_end > file_size {
                return Err(GgufError::TensorOutOfBounds {
                    name: t.name.clone(),
                    offset: t.offset,
                    size: t.size_bytes,
                    file_size,
                });
            }
        }

        let layer_index = LayerIndex::new(&tensor_infos);

        Ok(Self {
            header,
            metadata,
            tensor_infos,
            layer_index,
            tensor_data_offset,
            alignment,
        })
    }

    /// Access the parsed file header.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Access the key-value metadata map.
    pub fn metadata(&self) -> &HashMap<String, GgufValue> {
        &self.metadata
    }

    /// Access the list of tensor descriptions.
    pub fn tensor_infos(&self) -> &[TensorInfo] {
        &self.tensor_infos
    }

    /// Access the layer index for tensor organization.
    pub fn layer_index(&self) -> &LayerIndex {
        &self.layer_index
    }

    /// Byte offset where the tensor data section begins in the file.
    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    /// Tensor data alignment in bytes (default 32).
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Lookup a specific tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_infos.iter().find(|t| t.name == name)
    }
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

fn compute_tensor_size(dims: &[u64], tensor_type: GgmlType, name: &str) -> Result<u64, GgufError> {
    if dims.is_empty() {
        return Ok(0);
    }
    let block_elems = tensor_type.block_elements() as u64;
    let type_size = tensor_type.type_size() as u64;

    let d0 = dims[0];
    if !d0.is_multiple_of(block_elems) {
        return Err(GgufError::InvalidTensorShape(format!(
            "Tensor '{name}': dimension 0 ({d0}) must be a multiple of block elements ({block_elems})"
        )));
    }

    let num_blocks = d0 / block_elems;
    let mut total_bytes = num_blocks.checked_mul(type_size).ok_or_else(|| {
        GgufError::InvalidTensorShape(format!("Tensor '{name}': size overflow"))
    })?;

    for &d in &dims[1..] {
        total_bytes = total_bytes.checked_mul(d).ok_or_else(|| {
            GgufError::InvalidTensorShape(format!("Tensor '{name}': size overflow"))
        })?;
    }

    Ok(total_bytes)
}

fn read_exact_bytes<R: Read, const N: usize>(
    reader: &mut R,
    context: &'static str,
) -> Result<[u8; N], GgufError> {
    let mut buf = [0u8; N];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof(context)
        } else {
            GgufError::Io(e)
        }
    })?;
    Ok(buf)
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8, GgufError> {
    let buf = read_exact_bytes::<R, 1>(reader, "u8")?;
    Ok(buf[0])
}

fn read_i8<R: Read>(reader: &mut R) -> Result<i8, GgufError> {
    let buf = read_exact_bytes::<R, 1>(reader, "i8")?;
    Ok(buf[0] as i8)
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16, GgufError> {
    let buf = read_exact_bytes::<R, 2>(reader, "u16")?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(reader: &mut R) -> Result<i16, GgufError> {
    let buf = read_exact_bytes::<R, 2>(reader, "i16")?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, GgufError> {
    let buf = read_exact_bytes::<R, 4>(reader, "u32")?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32, GgufError> {
    let buf = read_exact_bytes::<R, 4>(reader, "i32")?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32, GgufError> {
    let buf = read_exact_bytes::<R, 4>(reader, "f32")?;
    Ok(f32::from_le_bytes(buf))
}

fn read_bool<R: Read>(reader: &mut R) -> Result<bool, GgufError> {
    let b = read_u8(reader)?;
    Ok(b != 0)
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, GgufError> {
    let buf = read_exact_bytes::<R, 8>(reader, "u64")?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(reader: &mut R) -> Result<i64, GgufError> {
    let buf = read_exact_bytes::<R, 8>(reader, "i64")?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64, GgufError> {
    let buf = read_exact_bytes::<R, 8>(reader, "f64")?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, GgufError> {
    let len = read_u64(reader)?;
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof("string bytes")
        } else {
            GgufError::Io(e)
        }
    })?;
    String::from_utf8(buf).map_err(GgufError::InvalidUtf8)
}

fn read_value<R: Read>(reader: &mut R, value_type: GgufType) -> Result<GgufValue, GgufError> {
    match value_type {
        GgufType::Uint8 => Ok(GgufValue::U8(read_u8(reader)?)),
        GgufType::Int8 => Ok(GgufValue::I8(read_i8(reader)?)),
        GgufType::Uint16 => Ok(GgufValue::U16(read_u16(reader)?)),
        GgufType::Int16 => Ok(GgufValue::I16(read_i16(reader)?)),
        GgufType::Uint32 => Ok(GgufValue::U32(read_u32(reader)?)),
        GgufType::Int32 => Ok(GgufValue::I32(read_i32(reader)?)),
        GgufType::Float32 => Ok(GgufValue::F32(read_f32(reader)?)),
        GgufType::Bool => Ok(GgufValue::Bool(read_bool(reader)?)),
        GgufType::String => Ok(GgufValue::String(read_string(reader)?)),
        GgufType::Uint64 => Ok(GgufValue::U64(read_u64(reader)?)),
        GgufType::Int64 => Ok(GgufValue::I64(read_i64(reader)?)),
        GgufType::Float64 => Ok(GgufValue::F64(read_f64(reader)?)),
        GgufType::Array => {
            let elem_type_raw = read_u32(reader)?;
            let elem_type = GgufType::try_from(elem_type_raw)?;
            let count = read_u64(reader)?;
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(read_value(reader, elem_type)?);
            }
            Ok(GgufValue::Array(items))
        }
    }
}
