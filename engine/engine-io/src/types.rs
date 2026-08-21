use crate::error::GgufError;

/// GGUF file header metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    /// Magic string, expected to be "GGUF".
    pub magic: String,
    /// GGUF format version (currently 3).
    pub version: u32,
    /// Number of tensors contained in the file.
    pub tensor_count: u64,
    /// Number of key-value metadata pairs.
    pub metadata_kv_count: u64,
}

/// GGUF metadata value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            other => Err(GgufError::InvalidValueType(other)),
        }
    }
}

/// Tagged enum for GGUF metadata values.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// Returns the value as `u8` if it matches.
    pub fn as_u8(&self) -> Option<u8> {
        match self {
            Self::U8(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as `i8` if it matches.
    pub fn as_i8(&self) -> Option<i8> {
        match self {
            Self::I8(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as `u16` if it matches.
    pub fn as_u16(&self) -> Option<u16> {
        match self {
            Self::U16(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as `i16` if it matches.
    pub fn as_i16(&self) -> Option<i16> {
        match self {
            Self::I16(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as `u32` if it matches.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as `i32` if it matches.
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::I32(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as `f32` if it matches.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as `bool` if it matches.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as `&str` if it matches.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v.as_str()),
            _ => None,
        }
    }

    /// Returns the value as `u64` if it matches.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as `i64` if it matches.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as `f64` if it matches.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as array slice if it matches.
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Helper to extract an array of strings.
    pub fn as_string_list(&self) -> Option<Vec<&str>> {
        match self {
            Self::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    out.push(item.as_str()?);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

/// GGML tensor types supported by GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl TryFrom<u32> for GgmlType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
            39 => Ok(Self::MXFP4),
            40 => Ok(Self::NVFP4),
            41 => Ok(Self::Q1_0),
            42 => Ok(Self::Q2_0),
            other => Err(GgufError::InvalidTensorType(other)),
        }
    }
}

impl GgmlType {
    /// Number of elements per block (block size in elements).
    pub fn block_elements(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::F64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1
            | Self::IQ4_NL | Self::MXFP4 | Self::NVFP4 | Self::Q1_0 | Self::Q2_0 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K
            | Self::IQ2_XXS | Self::IQ2_XS | Self::IQ3_XXS | Self::IQ1_S
            | Self::IQ3_S | Self::IQ2_S | Self::IQ4_XS | Self::IQ1_M
            | Self::TQ1_0 | Self::TQ2_0 => 256,
        }
    }

    /// Size of one block in bytes.
    pub fn type_size(&self) -> usize {
        match self {
            Self::I8 => 1,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 | Self::F64 => 8,
            Self::Q1_0 => 6,
            Self::Q2_0 => 10,
            Self::MXFP4 => 17,
            Self::Q4_0 | Self::IQ4_NL | Self::NVFP4 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::TQ1_0 => 48,
            Self::IQ1_S => 50,
            Self::IQ1_M => 56,
            Self::IQ2_XXS => 66,
            Self::TQ2_0 => 72,
            Self::IQ2_XS => 74,
            Self::IQ2_S => 82,
            Self::Q2_K => 84,
            Self::IQ3_XXS => 98,
            Self::Q3_K | Self::IQ3_S => 110,
            Self::IQ4_XS => 136,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
        }
    }
}

/// Metadata and layout info for a tensor in the GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Name of the tensor (e.g. "token_embd.weight", "blk.0.attn_q.weight").
    pub name: String,
    /// Dimensions of the tensor [ne0, ne1, ...].
    pub dims: Vec<u64>,
    /// Tensor element data type.
    pub ggml_type: GgmlType,
    /// Offset in bytes from the start of the tensor data area.
    pub offset: u64,
    /// Total byte size of the tensor data.
    pub size_bytes: u64,
}
