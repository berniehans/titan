//! Error-path / negative tests for the GGUF parser and loader.
//!
//! These use a self-contained synthetic-buffer helper (the same pattern as
//! `tests/synthetic_tests.rs`) so none depend on the 400 MiB fixture.

use engine_io::{GgufError, GgufReader, GgufType};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

/// Upper bound for declared string lengths in the parser (mirrors src/reader.rs).
const MAX_STRING_LEN: u64 = 64 * 1024 * 1024; // 64 MiB

/// Writes `bytes` to a temporary `.gguf` file and returns its path.
fn write_temp(prefix: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("errp_{}_{}.gguf", prefix, std::process::id()));
    let mut f = File::create(&path).expect("create temp file");
    f.write_all(bytes).expect("write temp file");
    path
}

/// Builds a GGUF header: magic + version + n_tensors + n_kv.
fn header(version: u32, n_tensors: u64, n_kv: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&version.to_le_bytes());
    b.extend_from_slice(&n_tensors.to_le_bytes());
    b.extend_from_slice(&n_kv.to_le_bytes());
    b
}

/// Appends a length-prefixed GGUF string.
fn push_string(b: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    b.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    b.extend_from_slice(bytes);
}

// ---------------------------------------------------------------------------
// Malformed header
// ---------------------------------------------------------------------------

#[test]
fn error_path_empty_input_is_eof() {
    let p = write_temp("empty", &[]);
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::UnexpectedEof("magic"))),
        "empty file must yield UnexpectedEof on magic, got {res:?}"
    );
}

#[test]
fn error_path_bad_magic_ggux() {
    let mut b = header(3, 0, 0);
    b[0..4].copy_from_slice(b"GGUx");
    let p = write_temp("magic", &b);
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::InvalidMagic(m)) if m == *b"GGUx"),
        "expected InvalidMagic(GGUx), got {res:?}"
    );
}

#[test]
fn error_path_unsupported_version_1() {
    let p = write_temp("v1", &header(1, 0, 0));
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::UnsupportedVersion(1))),
        "expected UnsupportedVersion(1), got {res:?}"
    );
}

#[test]
fn error_path_unsupported_version_2() {
    let p = write_temp("v2", &header(2, 0, 0));
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::UnsupportedVersion(2))),
        "expected UnsupportedVersion(2), got {res:?}"
    );
}

// ---------------------------------------------------------------------------
// Truncation matrix
// ---------------------------------------------------------------------------

#[test]
fn error_path_truncated_at_header_end() {
    // Header claims 1 KV entry but the file ends immediately after the header.
    let p = write_temp("trunchead", &header(3, 0, 1));
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::UnexpectedEof(_))),
        "expected UnexpectedEof reading first KV key, got {res:?}"
    );
}

#[test]
fn error_path_truncated_mid_metadata() {
    // Declares a 10-byte string value but the value bytes are missing.
    let mut b = header(3, 0, 1);
    push_string(&mut b, "key");
    b.extend_from_slice(&(GgufType::String as u32).to_le_bytes());
    b.extend_from_slice(&10u64.to_le_bytes()); // declared value length
    let p = write_temp("truncmeta", &b);
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::UnexpectedEof("string bytes"))),
        "expected UnexpectedEof on truncated string value, got {res:?}"
    );
}

#[test]
fn error_path_truncated_mid_tensor_infos() {
    // One tensor: name present, truncated before n_dims (u32) is read.
    let mut b = header(3, 1, 0);
    push_string(&mut b, "token_embd.weight");
    let p = write_temp("trunctensor", &b);
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::UnexpectedEof(_))),
        "expected UnexpectedEof before tensor dims, got {res:?}"
    );
}

// ---------------------------------------------------------------------------
// Allocation-bomb guards
// ---------------------------------------------------------------------------

#[test]
fn error_path_declared_string_len_overbound_is_bounded() {
    // Declared key length > MAX_STRING_LEN must be rejected without allocating.
    let mut b = header(3, 0, 1);
    b.extend_from_slice(&(MAX_STRING_LEN + 1).to_le_bytes()); // declared key length
    let p = write_temp("strbomb", &b);
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::MetadataTooLarge { .. })),
        "oversized declared string length must be a bounded error, got {res:?}"
    );
}

#[test]
fn error_path_declared_array_len_u64_max_is_bounded() {
    // An Array metadata value declaring u64::MAX items must be rejected before
    // any Vec preallocation; it must never panic.
    let mut b = header(3, 0, 1);
    push_string(&mut b, "key");
    b.extend_from_slice(&(GgufType::Array as u32).to_le_bytes());
    b.extend_from_slice(&(GgufType::Uint32 as u32).to_le_bytes()); // elem type
    b.extend_from_slice(&u64::MAX.to_le_bytes()); // declared count
    let p = write_temp("arrbomb", &b);
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::MetadataTooLarge { .. })),
        "u64::MAX array length must be a bounded error, got {res:?}"
    );
}

#[test]
fn error_path_declared_metadata_kv_count_u64_max_is_bounded() {
    let p = write_temp("kvbomb", &header(3, 0, u64::MAX));
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::MetadataTooLarge { .. })),
        "u64::MAX metadata KV count must be a bounded error, got {res:?}"
    );
}

#[test]
fn error_path_declared_tensor_count_u64_max_is_bounded() {
    let p = write_temp("tbomb", &header(3, u64::MAX, 0));
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::MetadataTooLarge { .. })),
        "u64::MAX tensor count must be a bounded error, got {res:?}"
    );
}

#[test]
fn error_path_declared_n_dims_absurd_is_bounded() {
    // One tensor declaring an absurd dimension count (u32::MAX).
    let mut b = header(3, 1, 0);
    push_string(&mut b, "t");
    b.extend_from_slice(&u32::MAX.to_le_bytes()); // n_dims
    let p = write_temp("ndimsbomb", &b);
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        matches!(res, Err(GgufError::MetadataTooLarge { .. })),
        "absurd n_dims must be a bounded error, got {res:?}"
    );
}

// ---------------------------------------------------------------------------
// Loader mismatch: tensor offsets past EOF
// ---------------------------------------------------------------------------

#[test]
fn error_path_loader_tensor_offsets_past_eof_is_clear_error() {
    // A tensor whose span starts far past EOF must surface as a clear
    // GgufError (TensorOutOfBounds) at the load path, never a panic.
    let mut b = header(3, 1, 0);
    push_string(&mut b, "t_oob");
    b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    b.extend_from_slice(&4u64.to_le_bytes()); // dim 0
    b.extend_from_slice(&(engine_io::GgmlType::F32 as u32).to_le_bytes());
    b.extend_from_slice(&1_000_000u64.to_le_bytes()); // offset far past EOF
    let p = write_temp("loaderoob", &b);
    let res = GgufReader::open(&p);
    let _ = std::fs::remove_file(&p);
    match res {
        Err(GgufError::TensorOutOfBounds { name, .. }) => assert_eq!(name, "t_oob"),
        other => panic!("expected TensorOutOfBounds for t_oob, got {other:?}"),
    }
}
