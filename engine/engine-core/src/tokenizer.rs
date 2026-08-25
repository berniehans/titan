//! BPE tokenizer for the Qwen3/Qwen2 family (`tokenizer.ggml.model=gpt2`,
//! `tokenizer.ggml.pre=qwen2`), ported from llama.cpp `src/llama-vocab.cpp`
//! (`llm_tokenizer_bpe` + `unicode_regex_split`) at pinned commit cb1adf8.
//!
//! Two layers:
//!   1. Pre-tokenization splits ASCII input into "words" following the Qwen2
//!      regex (EcmaScript semantics, leftmost-longest alternation). Non-ASCII
//!      input is rejected with `EngineError::NonAsciiInput` (full-Unicode
//!      pre-tokenization is out of scope for this change; the goldens use
//!      ASCII only).
//!   2. Greedy BPE merges each word bottom-up: token ids are looked up by
//!      text, and bigrams are merged by their merge-file rank (index into
//!      `tokenizer.ggml.merges`).
//!
//! This module intentionally mirrors the token-dripping order and the
//! unfinished-symbol byte fallback of `llm_tokenizer_bpe_session::tokenize`.

use crate as engine_core;
use crate::error::EngineError;
use engine_io::GgufReader;
use engine_io::types::GgufValue;
use std::collections::{BinaryHeap, HashMap};

/// Candidate merge pair for greedy BPE min-heap.
#[derive(Copy, Clone, Eq, PartialEq)]
struct MergeCandidate {
    rank: u32,
    left: usize,
    right: usize,
    size: usize,
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap on rank; tie larger right-left index; tie earlier left index
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| (self.right - self.left).cmp(&(other.right - other.left)))
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Tokenizer for the Qwen3 BPE model family.
pub struct BpeTokenizer {
    /// token id -> raw (escaped) token text as stored in the GGUF.
    tokens: Vec<String>,
    /// escaped token text -> token id.
    text_to_id: HashMap<String, u32>,
    /// merge rule text "A B" -> rank (index into the merges array).
    merge_rank: HashMap<String, u32>,
    /// GPT-2 byte->char mapping: codepoint (as u32) -> original byte.
    byte_map: HashMap<u32, u8>,
    /// Whether the model prepends a BOS token.
    pub add_bos: bool,
    /// End-of-sequence token id.
    pub eos_token_id: u32,
}

impl BpeTokenizer {
    /// Builds a tokenizer from a GGUF reader (reads `tokenizer.ggml.*`).
    pub fn from_reader(reader: &GgufReader) -> Result<Self, engine_core::EngineError> {
        let m = reader.metadata();

        let tokens: Vec<String> = m
            .get("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_array)
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default();

        let mut text_to_id: HashMap<String, u32> = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            text_to_id.insert(t.clone(), i as u32);
        }

        let mut merge_rank: HashMap<String, u32> = HashMap::new();
        if let Some(arr) = m.get("tokenizer.ggml.merges").and_then(GgufValue::as_array) {
            for (rank, v) in arr.iter().enumerate() {
                if let Some(s) = v.as_str() {
                    merge_rank.insert(s.to_string(), rank as u32);
                }
            }
        }

        let add_bos = m
            .get("tokenizer.ggml.add_bos_token")
            .and_then(GgufValue::as_bool)
            .unwrap_or(false);
        let eos_token_id = m
            .get("tokenizer.ggml.eos_token_id")
            .and_then(extract_u32)
            .unwrap_or(0);

        Ok(Self {
            tokens,
            text_to_id,
            merge_rank,
            byte_map: build_gpt2_byte_map(),
            add_bos,
            eos_token_id,
        })
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Encodes ASCII text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, engine_core::EngineError> {
        let words = pre_split(text).map_err(EngineError::NonAsciiInput)?;
        let byte_to_char = build_byte_to_char_table();
        let mut out = Vec::new();
        for word in &words {
            let word_ids = self.bpe_tokenize_word(word, &byte_to_char);
            out.extend(word_ids);
        }
        Ok(out)
    }

    fn bpe_tokenize_word(&self, word: &str, byte_to_char: &[char; 256]) -> Vec<u32> {
        let mut symbols: Vec<String> = word
            .as_bytes()
            .iter()
            .map(|&b| byte_to_char[b as usize].to_string())
            .collect();

        let mut heap = BinaryHeap::new();
        for i in 0..symbols.len().saturating_sub(1) {
            let l = i;
            let r = i + 1;
            let rule = format!("{} {}", symbols[l], symbols[r]);
            if let Some(&rank) = self.merge_rank.get(&rule) {
                let size = symbols[l].len() + symbols[r].len();
                heap.push(MergeCandidate {
                    rank,
                    left: l,
                    right: r,
                    size,
                });
            }
        }

        while let Some(cand) = heap.pop() {
            let l = cand.left;
            let r = cand.right;
            if symbols[l].is_empty() || symbols[r].is_empty() {
                continue;
            }
            let current_size = symbols[l].len() + symbols[r].len();
            if current_size != cand.size {
                continue;
            }
            let rule = format!("{} {}", symbols[l], symbols[r]);
            if self.merge_rank.get(&rule) != Some(&cand.rank) {
                continue;
            }

            let r_text = std::mem::take(&mut symbols[r]);
            symbols[l].push_str(&r_text);

            let mut prev = l as isize - 1;
            while prev >= 0 && symbols[prev as usize].is_empty() {
                prev -= 1;
            }
            if prev >= 0 {
                let p = prev as usize;
                let rule = format!("{} {}", symbols[p], symbols[l]);
                if let Some(&rank) = self.merge_rank.get(&rule) {
                    let size = symbols[p].len() + symbols[l].len();
                    heap.push(MergeCandidate {
                        rank,
                        left: p,
                        right: l,
                        size,
                    });
                }
            }

            let mut next = r + 1;
            while next < symbols.len() && symbols[next].is_empty() {
                next += 1;
            }
            if next < symbols.len() {
                let rule = format!("{} {}", symbols[l], symbols[next]);
                if let Some(&rank) = self.merge_rank.get(&rule) {
                    let size = symbols[l].len() + symbols[next].len();
                    heap.push(MergeCandidate {
                        rank,
                        left: l,
                        right: next,
                        size,
                    });
                }
            }
        }

        let mut out = Vec::new();
        for s in &symbols {
            if s.is_empty() {
                continue;
            }
            if let Some(&id) = self.text_to_id.get(s) {
                out.push(id);
            } else {
                for &b in s.as_bytes() {
                    let piece = byte_to_char[b as usize].to_string();
                    if let Some(&id) = self.text_to_id.get(&piece) {
                        out.push(id);
                    }
                }
            }
        }
        out
    }

    /// Decodes a sequence of token IDs back into a UTF-8 string.
    pub fn decode(&self, ids: &[u32]) -> Result<String, engine_core::EngineError> {
        let mut bytes = Vec::new();
        for &id in ids {
            let tok = self
                .tokens
                .get(id as usize)
                .ok_or(EngineError::UnknownToken(id))?;
            for c in tok.chars() {
                let cp = c as u32;
                if let Some(&b) = self.byte_map.get(&cp) {
                    bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    let s = c.encode_utf8(&mut buf);
                    bytes.extend_from_slice(s.as_bytes());
                }
            }
        }
        String::from_utf8(bytes)
            .map_err(|e| EngineError::Gguf(engine_io::GgufError::InvalidUtf8(e)))
    }
}

/// Splits ASCII text into words using the Qwen2 regex rules.
pub fn pre_split(text: &str) -> Result<Vec<String>, usize> {
    for (i, &b) in text.as_bytes().iter().enumerate() {
        if b >= 0x80 {
            return Err(i);
        }
    }

    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    let mut words = Vec::new();

    let is_letter = |b: u8| b.is_ascii_alphabetic();
    let is_number = |b: u8| b.is_ascii_digit();
    let is_space = |b: u8| matches!(b, b' ' | b'\t' | b'\r' | b'\n' | 0x0B | 0x0C);
    let is_not_newline_letter_number =
        |b: u8| b != b'\r' && b != b'\n' && !is_letter(b) && !is_number(b);
    let is_not_space_letter_number = |b: u8| !is_space(b) && !is_letter(b) && !is_number(b);

    while i < n {
        let mut best_len = 0;
        let mut best_branch = usize::MAX;

        // Branch 1: Contractions
        // (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
        if bytes[i] == b'\'' {
            let rem = &bytes[i..];
            for pat in [b"'s", b"'S", b"'t", b"'T", b"'m", b"'M", b"'d", b"'D"] {
                if rem.starts_with(pat) {
                    let match_len = pat.len();
                    if match_len > best_len || (match_len == best_len && 1 < best_branch) {
                        best_len = match_len;
                        best_branch = 1;
                    }
                }
            }
            for pat in [
                b"'re", b"'rE", b"'Re", b"'RE", b"'ve", b"'vE", b"'Ve", b"'VE", b"'ll", b"'lL",
                b"'Ll", b"'LL",
            ] {
                if rem.starts_with(pat) {
                    let match_len = pat.len();
                    if match_len > best_len || (match_len == best_len && 1 < best_branch) {
                        best_len = match_len;
                        best_branch = 1;
                    }
                }
            }
        }

        // Branch 2: [^\r\n\p{L}\p{N}]?\p{L}+
        if is_not_newline_letter_number(bytes[i]) {
            let mut k = i + 1;
            if k < n && is_letter(bytes[k]) {
                while k < n && is_letter(bytes[k]) {
                    k += 1;
                }
                let match_len = k - i;
                if match_len > best_len || (match_len == best_len && 2 < best_branch) {
                    best_len = match_len;
                    best_branch = 2;
                }
            }
        } else if is_letter(bytes[i]) {
            let mut k = i;
            while k < n && is_letter(bytes[k]) {
                k += 1;
            }
            let match_len = k - i;
            if match_len > best_len || (match_len == best_len && 2 < best_branch) {
                best_len = match_len;
                best_branch = 2;
            }
        }

        // Branch 3: \p{N}
        if is_number(bytes[i]) {
            let match_len = 1;
            if match_len > best_len || (match_len == best_len && 3 < best_branch) {
                best_len = match_len;
                best_branch = 3;
            }
        }

        // Branch 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        let mut k = i;
        if bytes[k] == b' ' {
            k += 1;
        }
        if k < n && is_not_space_letter_number(bytes[k]) {
            while k < n && is_not_space_letter_number(bytes[k]) {
                k += 1;
            }
            while k < n && (bytes[k] == b'\r' || bytes[k] == b'\n') {
                k += 1;
            }
            let match_len = k - i;
            if match_len > best_len || (match_len == best_len && 4 < best_branch) {
                best_len = match_len;
                best_branch = 4;
            }
        }

        // Branch 5: \s*[\r\n]+
        let mut k = i;
        while k < n && is_space(bytes[k]) {
            k += 1;
        }
        while k > i && bytes[k - 1] != b'\r' && bytes[k - 1] != b'\n' {
            k -= 1;
        }
        if k > i {
            let match_len = k - i;
            if match_len > best_len || (match_len == best_len && 5 < best_branch) {
                best_len = match_len;
                best_branch = 5;
            }
        }

        // Branch 6: \s+(?!\S)
        let mut k = i;
        while k < n && is_space(bytes[k]) {
            k += 1;
        }
        if k > i && k == n {
            let match_len = k - i;
            if match_len > best_len || (match_len == best_len && 6 < best_branch) {
                best_len = match_len;
                best_branch = 6;
            }
        }

        // Branch 7: \s
        if is_space(bytes[i]) {
            let match_len = 1;
            if match_len > best_len || (match_len == best_len && 7 < best_branch) {
                best_len = match_len;
            }
        }

        if best_len == 0 {
            best_len = 1;
        }

        words.push(text[i..i + best_len].to_string());
        i += best_len;
    }

    Ok(words)
}

/// Builds the GPT-2 byte mapping: codepoint (as u32) -> original byte.
pub fn build_gpt2_byte_map() -> HashMap<u32, u8> {
    let mut bs: Vec<u32> = Vec::new();
    for b in 0x21..=0x7E {
        bs.push(b);
    }
    for b in 0xA1..=0xAC {
        bs.push(b);
    }
    for b in 0xAE..=0xFF {
        bs.push(b);
    }
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..=255u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut map = HashMap::with_capacity(256);
    for i in 0..bs.len() {
        map.insert(cs[i], bs[i] as u8);
    }
    map
}

/// Builds the byte (0..=255) -> unicode char lookup table.
fn build_byte_to_char_table() -> [char; 256] {
    let mut bs: Vec<u32> = Vec::new();
    for b in 0x21..=0x7E {
        bs.push(b);
    }
    for b in 0xA1..=0xAC {
        bs.push(b);
    }
    for b in 0xAE..=0xFF {
        bs.push(b);
    }
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..=255u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = ['\0'; 256];
    for i in 0..bs.len() {
        table[bs[i] as usize] = char::from_u32(cs[i]).unwrap();
    }
    table
}

fn extract_u32(v: &GgufValue) -> Option<u32> {
    match v {
        GgufValue::U32(x) => Some(*x),
        _ => None,
    }
}
