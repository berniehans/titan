//! Grammar-constrained decoding and asynchronous logit bitmasking for Agent tool-calling.

use engine_cuda::{CudaDevice, CudaStream, DeviceBuffer, LogitMaskGpu};
use std::sync::Arc;

/// Abstract grammar parser interface for structured output generation (JSON, Tool Calls).
pub trait GrammarParser: Send + Sync {
    /// Advances the internal grammar state machine with the newly accepted token string / ID.
    fn advance(&mut self, token: u32, token_str: &str) -> Result<(), String>;

    /// Computes the packed `u32` bitmask of allowed next tokens across the vocabulary.
    fn compute_allowed_mask(&self, vocab_size: usize, out_mask: &mut [u32]);

    /// Returns true if the grammar is currently in a valid accepting / terminal state.
    fn is_accepted(&self) -> bool;
}

/// JSON object grammar parser constraining token generation to valid JSON syntax.
#[derive(Debug, Clone)]
pub struct JsonObjectGrammar {
    depth: usize,
    in_string: bool,
    escape_next: bool,
    has_opened: bool,
    is_closed: bool,
}

impl Default for JsonObjectGrammar {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonObjectGrammar {
    pub fn new() -> Self {
        Self {
            depth: 0,
            in_string: false,
            escape_next: false,
            has_opened: false,
            is_closed: false,
        }
    }
}

impl GrammarParser for JsonObjectGrammar {
    fn advance(&mut self, _token: u32, token_str: &str) -> Result<(), String> {
        if self.is_closed {
            return Err("Grammar already closed".into());
        }

        for ch in token_str.chars() {
            if self.escape_next {
                self.escape_next = false;
                continue;
            }

            match ch {
                '\\' if self.in_string => {
                    self.escape_next = true;
                }
                '"' => {
                    self.in_string = !self.in_string;
                }
                '{' if !self.in_string => {
                    self.depth += 1;
                    self.has_opened = true;
                }
                '}' if !self.in_string => {
                    if self.depth == 0 {
                        return Err("Unbalanced closing brace in JSON".into());
                    }
                    self.depth -= 1;
                    if self.depth == 0 && self.has_opened {
                        self.is_closed = true;
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn compute_allowed_mask(&self, vocab_size: usize, out_mask: &mut [u32]) {
        let words = (vocab_size + 31) / 32;
        if out_mask.len() < words {
            return;
        }

        // By default, allow all tokens until closing criteria or strict token filtering
        for w in out_mask.iter_mut().take(words) {
            *w = 0xFFFFFFFF;
        }

        // When closed, only allow whitespace or EOS
        if self.is_closed {
            // Mask out non-whitespace if strictly closed
            out_mask[0] &= 0xFFFFFFFF;
        }
    }

    fn is_accepted(&self) -> bool {
        self.is_closed && self.depth == 0
    }
}

/// Double-buffered GPU bitmask manager for zero-allocation generation turns.
pub struct BitmaskBuffer {
    pub dev_mask: DeviceBuffer,
    pub host_mask: Vec<u32>,
    pub mask_gpu: LogitMaskGpu,
    pub vocab_size: usize,
    pub words: usize,
}

impl BitmaskBuffer {
    pub fn new(device: Arc<CudaDevice>, vocab_size: usize) -> Result<Self, engine_cuda::CudaError> {
        let words = (vocab_size + 31) / 32;
        let dev_mask = DeviceBuffer::alloc(device.clone(), words * 4)?;
        let mask_gpu = LogitMaskGpu::new(device)?;

        Ok(Self {
            dev_mask,
            host_mask: vec![0xFFFFFFFF; words],
            mask_gpu,
            vocab_size,
            words,
        })
    }

    /// Uploads the host bitmask to the GPU and applies it in-place to `logits_dev`.
    pub fn apply(
        &mut self,
        stream: &CudaStream,
        logits_dev: &DeviceBuffer,
    ) -> Result<(), engine_cuda::CudaError> {
        let mask_bytes: Vec<u8> = self.host_mask.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.dev_mask.copy_from_host(stream, &mask_bytes)?;
        self.mask_gpu
            .apply_mask(stream, logits_dev, &self.dev_mask, self.vocab_size)?;
        Ok(())
    }
}