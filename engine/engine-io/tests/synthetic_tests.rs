use std::fs::File;
use std::io::Write;
use engine_io::{GgmlType, GgufError, GgufReader, GgufType};

struct SyntheticTensor {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    offset: u64,
    data: Vec<u8>,
}

struct GgufBuilder {
    version: u32,
    magic: [u8; 4],
    kv_entries: Vec<(String, u32, Vec<u8>)>,
    tensors: Vec<SyntheticTensor>,
    alignment: u64,
}

impl GgufBuilder {
    fn new() -> Self {
        Self {
            version: 3,
            magic: *b"GGUF",
            kv_entries: Vec::new(),
            tensors: Vec::new(),
            alignment: 32,
        }
    }

    fn with_magic(mut self, magic: [u8; 4]) -> Self {
        self.magic = magic;
        self
    }

    fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    fn add_scalar_kv(mut self, key: &str, val_type: GgufType, val_bytes: &[u8]) -> Self {
        self.kv_entries.push((key.to_string(), val_type as u32, val_bytes.to_vec()));
        self
    }

    fn add_array_kv(mut self, key: &str, elem_type: GgufType, count: u64, data_bytes: &[u8]) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(elem_type as u32).to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        bytes.extend_from_slice(data_bytes);
        self.kv_entries.push((key.to_string(), GgufType::Array as u32, bytes));
        self
    }

    fn add_tensor(mut self, name: &str, dims: Vec<u64>, ggml_type: GgmlType, data: Vec<u8>) -> Self {
        let offset = self.tensors.iter().map(|t| t.data.len() as u64).sum();
        self.tensors.push(SyntheticTensor {
            name: name.to_string(),
            dims,
            ggml_type: ggml_type as u32,
            offset,
            data,
        });
        self
    }

    fn build_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // 1. Magic
        buf.extend_from_slice(&self.magic);
        // 2. Version
        buf.extend_from_slice(&self.version.to_le_bytes());
        // 3. n_tensors
        let n_tensors = self.tensors.len() as u64;
        buf.extend_from_slice(&n_tensors.to_le_bytes());
        // 4. n_kv
        let n_kv = self.kv_entries.len() as u64;
        buf.extend_from_slice(&n_kv.to_le_bytes());

        // 5. KV entries
        for (key, val_type, val_data) in &self.kv_entries {
            let key_bytes = key.as_bytes();
            buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(key_bytes);
            buf.extend_from_slice(&val_type.to_le_bytes());
            buf.extend_from_slice(val_data);
        }

        // 6. Tensors
        for t in &self.tensors {
            let name_bytes = t.name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&t.ggml_type.to_le_bytes());
            buf.extend_from_slice(&t.offset.to_le_bytes());
        }

        // 7. Align up to alignment
        let current_pos = buf.len() as u64;
        let aligned_pos = (current_pos + self.alignment - 1) & !(self.alignment - 1);
        buf.resize(aligned_pos as usize, 0);

        // 8. Tensor data
        for t in &self.tensors {
            buf.extend_from_slice(&t.data);
        }

        buf
    }

    fn write_to_temp_file(&self, prefix: &str) -> std::path::PathBuf {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!("test_{}_{}.gguf", prefix, std::process::id()));
        let mut file = File::create(&file_path).expect("Create temp file");
        file.write_all(&self.build_bytes()).expect("Write temp file");
        file_path
    }
}

#[test]
fn test_all_scalar_and_array_types() {
    let mut str_bytes = Vec::new();
    let sample_str = "hello_gguf";
    str_bytes.extend_from_slice(&(sample_str.len() as u64).to_le_bytes());
    str_bytes.extend_from_slice(sample_str.as_bytes());

    let mut string_array_data = Vec::new();
    for s in &["token_a", "token_b", "token_c"] {
        string_array_data.extend_from_slice(&(s.len() as u64).to_le_bytes());
        string_array_data.extend_from_slice(s.as_bytes());
    }

    let mut u32_array_data = Vec::new();
    for v in &[10u32, 20u32, 30u32] {
        u32_array_data.extend_from_slice(&v.to_le_bytes());
    }

    let mut f32_array_data = Vec::new();
    for v in &[1.5f32, 2.5f32, 3.5f32] {
        f32_array_data.extend_from_slice(&v.to_le_bytes());
    }

    let pi_f32 = std::f32::consts::PI;
    let e_f64 = std::f64::consts::E;

    let path = GgufBuilder::new()
        .add_scalar_kv("k_u8", GgufType::Uint8, &[42u8])
        .add_scalar_kv("k_i8", GgufType::Int8, &((-42i8).to_le_bytes()))
        .add_scalar_kv("k_u16", GgufType::Uint16, &1000u16.to_le_bytes())
        .add_scalar_kv("k_i16", GgufType::Int16, &(-1000i16).to_le_bytes())
        .add_scalar_kv("k_u32", GgufType::Uint32, &100000u32.to_le_bytes())
        .add_scalar_kv("k_i32", GgufType::Int32, &(-100000i32).to_le_bytes())
        .add_scalar_kv("k_f32", GgufType::Float32, &pi_f32.to_le_bytes())
        .add_scalar_kv("k_bool", GgufType::Bool, &[1u8])
        .add_scalar_kv("k_str", GgufType::String, &str_bytes)
        .add_scalar_kv("k_u64", GgufType::Uint64, &10000000000u64.to_le_bytes())
        .add_scalar_kv("k_i64", GgufType::Int64, &(-10000000000i64).to_le_bytes())
        .add_scalar_kv("k_f64", GgufType::Float64, &e_f64.to_le_bytes())
        .add_array_kv("k_str_arr", GgufType::String, 3, &string_array_data)
        .add_array_kv("k_u32_arr", GgufType::Uint32, 3, &u32_array_data)
        .add_array_kv("k_f32_arr", GgufType::Float32, 3, &f32_array_data)
        .add_tensor("token_embd.weight", vec![4, 2], GgmlType::F32, vec![0u8; 32]) // 4 * 2 * 4 bytes = 32 bytes
        .add_tensor("blk.0.attn_q.weight", vec![4, 2], GgmlType::F32, vec![0u8; 32])
        .write_to_temp_file("scalars_and_arrays");

    let reader = GgufReader::open(&path).expect("Open synthetic file");
    let _ = std::fs::remove_file(&path);

    let meta = reader.metadata();

    assert_eq!(meta.get("k_u8").unwrap().as_u8(), Some(42));
    assert_eq!(meta.get("k_i8").unwrap().as_i8(), Some(-42));
    assert_eq!(meta.get("k_u16").unwrap().as_u16(), Some(1000));
    assert_eq!(meta.get("k_i16").unwrap().as_i16(), Some(-1000));
    assert_eq!(meta.get("k_u32").unwrap().as_u32(), Some(100000));
    assert_eq!(meta.get("k_i32").unwrap().as_i32(), Some(-100000));
    assert!((meta.get("k_f32").unwrap().as_f32().unwrap() - pi_f32).abs() < 1e-6);
    assert_eq!(meta.get("k_bool").unwrap().as_bool(), Some(true));
    assert_eq!(meta.get("k_str").unwrap().as_str(), Some("hello_gguf"));
    assert_eq!(meta.get("k_u64").unwrap().as_u64(), Some(10000000000));
    assert_eq!(meta.get("k_i64").unwrap().as_i64(), Some(-10000000000));
    assert!((meta.get("k_f64").unwrap().as_f64().unwrap() - e_f64).abs() < 1e-12);

    // Arrays
    let str_arr = meta.get("k_str_arr").unwrap().as_string_list().unwrap();
    assert_eq!(str_arr, vec!["token_a", "token_b", "token_c"]);

    let u32_arr = meta.get("k_u32_arr").unwrap().as_array().unwrap();
    assert_eq!(u32_arr.len(), 3);
    assert_eq!(u32_arr[0].as_u32(), Some(10));
    assert_eq!(u32_arr[1].as_u32(), Some(20));
    assert_eq!(u32_arr[2].as_u32(), Some(30));

    let f32_arr = meta.get("k_f32_arr").unwrap().as_array().unwrap();
    assert_eq!(f32_arr.len(), 3);
    assert!((f32_arr[0].as_f32().unwrap() - 1.5).abs() < 1e-5);

    // Tensors & LayerIndex
    assert_eq!(reader.tensor_infos().len(), 2);
    let layer_idx = reader.layer_index();
    assert_eq!(layer_idx.layers(), vec![0]);
    assert_eq!(layer_idx.by_layer(0).unwrap().len(), 1);
    assert_eq!(layer_idx.non_layer_tensors().len(), 1);
    assert_eq!(layer_idx.non_layer_tensors()[0].name, "token_embd.weight");
}

#[test]
fn test_invalid_magic() {
    let path = GgufBuilder::new()
        .with_magic(*b"BADM")
        .write_to_temp_file("bad_magic");

    let result = GgufReader::open(&path);
    let _ = std::fs::remove_file(&path);

    match result {
        Err(GgufError::InvalidMagic(m)) => assert_eq!(&m, b"BADM"),
        other => panic!("Expected InvalidMagic, got {:?}", other),
    }
}

#[test]
fn test_unsupported_version() {
    let path = GgufBuilder::new()
        .with_version(2)
        .write_to_temp_file("v2_file");

    let result = GgufReader::open(&path);
    let _ = std::fs::remove_file(&path);

    match result {
        Err(GgufError::UnsupportedVersion(2)) => {}
        other => panic!("Expected UnsupportedVersion(2), got {:?}", other),
    }
}

#[test]
fn test_invalid_value_type() {
    let mut builder = GgufBuilder::new();
    builder.kv_entries.push(("bad_key".to_string(), 999, vec![0]));
    let path = builder.write_to_temp_file("bad_val_type");

    let result = GgufReader::open(&path);
    let _ = std::fs::remove_file(&path);

    match result {
        Err(GgufError::InvalidValueType(999)) => {}
        other => panic!("Expected InvalidValueType(999), got {:?}", other),
    }
}

#[test]
fn test_invalid_tensor_type() {
    let mut builder = GgufBuilder::new();
    builder.tensors.push(SyntheticTensor {
        name: "t1".to_string(),
        dims: vec![4],
        ggml_type: 9999,
        offset: 0,
        data: vec![0; 16],
    });
    let path = builder.write_to_temp_file("bad_tensor_type");

    let result = GgufReader::open(&path);
    let _ = std::fs::remove_file(&path);

    match result {
        Err(GgufError::InvalidTensorType(9999)) => {}
        other => panic!("Expected InvalidTensorType(9999), got {:?}", other),
    }
}

#[test]
fn test_invalid_alignment() {
    let path = GgufBuilder::new()
        .add_scalar_kv("general.alignment", GgufType::Uint32, &7u32.to_le_bytes())
        .write_to_temp_file("bad_align");

    let result = GgufReader::open(&path);
    let _ = std::fs::remove_file(&path);

    match result {
        Err(GgufError::InvalidAlignment(7)) => {}
        other => panic!("Expected InvalidAlignment(7), got {:?}", other),
    }
}

#[test]
fn test_tensor_out_of_bounds() {
    let mut builder = GgufBuilder::new();
    // Offset far past file end
    builder.tensors.push(SyntheticTensor {
        name: "t_oob".to_string(),
        dims: vec![4, 2],
        ggml_type: GgmlType::F32 as u32,
        offset: 1_000_000,
        data: vec![0; 32],
    });
    let path = builder.write_to_temp_file("tensor_oob");

    let result = GgufReader::open(&path);
    let _ = std::fs::remove_file(&path);

    match result {
        Err(GgufError::TensorOutOfBounds { name, .. }) => assert_eq!(name, "t_oob"),
        other => panic!("Expected TensorOutOfBounds, got {:?}", other),
    }
}

#[test]
fn test_invalid_tensor_dimension_block() {
    let mut builder = GgufBuilder::new();
    // Q4_0 requires dimension 0 to be multiple of 32, but we give 10
    builder.tensors.push(SyntheticTensor {
        name: "t_bad_dim".to_string(),
        dims: vec![10],
        ggml_type: GgmlType::Q4_0 as u32,
        offset: 0,
        data: vec![0; 18],
    });
    let path = builder.write_to_temp_file("bad_dim");

    let result = GgufReader::open(&path);
    let _ = std::fs::remove_file(&path);

    match result {
        Err(GgufError::InvalidTensorShape(_)) => {}
        other => panic!("Expected InvalidTensorShape, got {:?}", other),
    }
}
