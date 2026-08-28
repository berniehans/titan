//! Grammar-Guided Constrained Decoding and JSON Schema Validation.
//!
//! Provides deterministic state machine parsers for enforcing 100% syntactically
//! valid JSON generation, OpenAI tool call schemas, and structured outputs.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonParserState {
    Start,
    InObject,
    ExpectKey,
    InKeyString,
    ExpectColon,
    ExpectValue,
    InValueString,
    InNumber,
    InBooleanOrNull,
    ExpectCommaOrClose,
    InArray,
    Complete,
    Error,
}

/// Dynamic stack-based JSON grammar validator for token-by-token logit filtering.
#[derive(Debug, Clone)]
pub struct JsonGrammar {
    pub state: JsonParserState,
    pub stack: Vec<char>,
    pub buffer: String,
    pub depth: usize,
    pub in_escape: bool,
    pub allowed_keys: Option<Vec<String>>,
    pub current_key: String,
    pub current_literal: String,
}

impl Default for JsonGrammar {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGrammar {
    /// Creates a new JSON grammar validator starting from root.
    pub fn new() -> Self {
        Self {
            state: JsonParserState::Start,
            stack: Vec::with_capacity(16),
            buffer: String::with_capacity(256),
            depth: 0,
            in_escape: false,
            allowed_keys: None,
            current_key: String::new(),
            current_literal: String::new(),
        }
    }

    /// Sets explicit allowed JSON keys for schema-guided tool calling.
    pub fn with_keys(mut self, keys: &[&str]) -> Self {
        self.allowed_keys = Some(keys.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Creates a JSON grammar validator starting from an already-opened root object `{`.
    pub fn inside_object() -> Self {
        let mut g = Self::new();
        g.stack.push('}');
        g.depth = 1;
        g.state = JsonParserState::ExpectKey;
        g.buffer.push('{');
        g
    }

    /// Returns true if the accumulated output is complete and valid JSON.
    pub fn is_complete(&self) -> bool {
        self.state == JsonParserState::Complete || (self.depth == 0 && !self.buffer.trim().is_empty())
    }

    /// Returns true if `token_str` is allowed at the current parser state.
    pub fn is_token_valid(&self, token_str: &str) -> bool {
        if token_str.is_empty() {
            return false;
        }

        // Structural progress enforcement
        match self.state {
            JsonParserState::ExpectKey => {
                let trimmed = token_str.trim_start();
                if trimmed.is_empty() {
                    if self.buffer.ends_with(' ') || self.buffer.ends_with('\n') || self.buffer.ends_with('\t') {
                        return false;
                    }
                } else if !trimmed.starts_with('"') && !trimmed.starts_with('}') {
                    return false;
                }
            }
            JsonParserState::ExpectColon => {
                let trimmed = token_str.trim_start();
                if trimmed.is_empty() {
                    if self.buffer.ends_with(' ') || self.buffer.ends_with('\n') || self.buffer.ends_with('\t') {
                        return false;
                    }
                } else if !trimmed.starts_with(':') {
                    return false;
                }
            }
            JsonParserState::ExpectValue => {
                let trimmed = token_str.trim_start();
                if trimmed.is_empty() {
                    if self.buffer.ends_with(' ') || self.buffer.ends_with('\n') || self.buffer.ends_with('\t') {
                        return false;
                    }
                }
            }
            JsonParserState::ExpectCommaOrClose => {
                let trimmed = token_str.trim_start();
                if trimmed.is_empty() {
                    if self.buffer.ends_with(' ') || self.buffer.ends_with('\n') || self.buffer.ends_with('\t') {
                        return false;
                    }
                } else if !trimmed.starts_with(',') && !trimmed.starts_with('}') && !trimmed.starts_with(']') {
                    return false;
                }
            }
            _ => {}
        }

        let mut test_grammar = self.clone();
        for ch in token_str.chars() {
            if !test_grammar.consume_char(ch) {
                return false;
            }
        }
        test_grammar.state != JsonParserState::Error
    }

    /// Advances the grammar parser state by consuming `token_str`.
    pub fn advance(&mut self, token_str: &str) -> bool {
        for ch in token_str.chars() {
            if !self.consume_char(ch) {
                self.state = JsonParserState::Error;
                return false;
            }
        }
        true
    }

    fn consume_char(&mut self, ch: char) -> bool {
        self.buffer.push(ch);

        if self.in_escape {
            self.in_escape = false;
            return matches!(ch, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u' | '0'..='9' | 'a'..='f' | 'A'..='F');
        }

        if ch == '\\' && self.state == JsonParserState::InValueString {
            self.in_escape = true;
            return true;
        }

        match self.state {
            JsonParserState::Start => {
                if ch.is_whitespace() {
                    true
                } else if ch == '{' {
                    self.stack.push('}');
                    self.depth += 1;
                    self.state = JsonParserState::ExpectKey;
                    true
                } else if ch == '[' {
                    self.stack.push(']');
                    self.depth += 1;
                    self.state = JsonParserState::InArray;
                    true
                } else if ch == '"' {
                    self.state = JsonParserState::InValueString;
                    true
                } else if ch.is_ascii_digit() || ch == '-' {
                    self.state = JsonParserState::InNumber;
                    true
                } else if ch == 't' || ch == 'f' || ch == 'n' {
                    self.state = JsonParserState::InBooleanOrNull;
                    true
                } else {
                    false
                }
            }
            JsonParserState::ExpectKey => {
                if ch.is_whitespace() {
                    true
                } else if ch == '"' {
                    self.current_key.clear();
                    self.state = JsonParserState::InKeyString;
                    true
                } else if ch == '}' && self.stack.last() == Some(&'}') {
                    self.stack.pop();
                    self.depth -= 1;
                    self.state = if self.depth == 0 { JsonParserState::Complete } else { JsonParserState::ExpectCommaOrClose };
                    true
                } else {
                    false
                }
            }
            JsonParserState::InKeyString => {
                if ch == '\n' || ch == '\r' || (ch < ' ' && ch != '\t') {
                    return false;
                }
                if ch == '"' {
                    if let Some(ref allowed) = self.allowed_keys {
                        if !allowed.iter().any(|k| k == &self.current_key) {
                            return false;
                        }
                    }
                    self.state = JsonParserState::ExpectColon;
                } else {
                    self.current_key.push(ch);
                    if let Some(ref allowed) = self.allowed_keys {
                        if !allowed.iter().any(|k| k.starts_with(&self.current_key)) {
                            return false;
                        }
                    }
                }
                true
            }
            JsonParserState::ExpectColon => {
                if ch.is_whitespace() {
                    true
                } else if ch == ':' {
                    self.state = JsonParserState::ExpectValue;
                    true
                } else {
                    false
                }
            }
            JsonParserState::ExpectValue => {
                if ch.is_whitespace() {
                    true
                } else if ch == '{' {
                    self.stack.push('}');
                    self.depth += 1;
                    self.state = JsonParserState::ExpectKey;
                    true
                } else if ch == '[' {
                    self.stack.push(']');
                    self.depth += 1;
                    self.state = JsonParserState::InArray;
                    true
                } else if ch == '"' {
                    self.state = JsonParserState::InValueString;
                    true
                } else if ch.is_ascii_digit() || ch == '-' {
                    self.state = JsonParserState::InNumber;
                    true
                } else if ch == 't' || ch == 'f' || ch == 'n' {
                    self.current_literal.clear();
                    self.current_literal.push(ch);
                    self.state = JsonParserState::InBooleanOrNull;
                    true
                } else {
                    false
                }
            }
            JsonParserState::InValueString => {
                if ch == '\n' || ch == '\r' || (ch < ' ' && ch != '\t') {
                    return false;
                }
                if ch == '"' {
                    self.state = JsonParserState::ExpectCommaOrClose;
                }
                true
            }
            JsonParserState::InNumber => {
                if ch.is_ascii_digit() || ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-' {
                    true
                } else if ch == ',' {
                    self.state = if self.stack.last() == Some(&'}') { JsonParserState::ExpectKey } else { JsonParserState::ExpectValue };
                    true
                } else if ch == '}' && self.stack.last() == Some(&'}') {
                    self.stack.pop();
                    self.depth -= 1;
                    self.state = if self.depth == 0 { JsonParserState::Complete } else { JsonParserState::ExpectCommaOrClose };
                    true
                } else if ch == ']' && self.stack.last() == Some(&']') {
                    self.stack.pop();
                    self.depth -= 1;
                    self.state = if self.depth == 0 { JsonParserState::Complete } else { JsonParserState::ExpectCommaOrClose };
                    true
                } else if ch.is_whitespace() {
                    self.state = JsonParserState::ExpectCommaOrClose;
                    true
                } else {
                    false
                }
            }
            JsonParserState::InBooleanOrNull => {
                if ch.is_alphabetic() {
                    self.current_literal.push(ch);
                    let valid_prefix = "true".starts_with(&self.current_literal)
                        || "false".starts_with(&self.current_literal)
                        || "null".starts_with(&self.current_literal);
                    if !valid_prefix {
                        return false;
                    }
                    true
                } else {
                    let is_exact = self.current_literal == "true"
                        || self.current_literal == "false"
                        || self.current_literal == "null";
                    if !is_exact {
                        return false;
                    }
                    if ch == ',' {
                        self.state = if self.stack.last() == Some(&'}') { JsonParserState::ExpectKey } else { JsonParserState::ExpectValue };
                        true
                    } else if (ch == '}' && self.stack.last() == Some(&'}')) || (ch == ']' && self.stack.last() == Some(&']')) {
                        self.stack.pop();
                        self.depth -= 1;
                        self.state = if self.depth == 0 { JsonParserState::Complete } else { JsonParserState::ExpectCommaOrClose };
                        true
                    } else if ch.is_whitespace() {
                        self.state = JsonParserState::ExpectCommaOrClose;
                        true
                    } else {
                        false
                    }
                }
            }
            JsonParserState::ExpectCommaOrClose => {
                if ch.is_whitespace() {
                    true
                } else if ch == ',' {
                    self.state = if self.stack.last() == Some(&'}') { JsonParserState::ExpectKey } else { JsonParserState::ExpectValue };
                    true
                } else if ch == '}' && self.stack.last() == Some(&'}') {
                    self.stack.pop();
                    self.depth -= 1;
                    self.state = if self.depth == 0 { JsonParserState::Complete } else { JsonParserState::ExpectCommaOrClose };
                    true
                } else if ch == ']' && self.stack.last() == Some(&']') {
                    self.stack.pop();
                    self.depth -= 1;
                    self.state = if self.depth == 0 { JsonParserState::Complete } else { JsonParserState::ExpectCommaOrClose };
                    true
                } else {
                    false
                }
            }
            JsonParserState::InArray => {
                if ch.is_whitespace() {
                    true
                } else if ch == ']' && self.stack.last() == Some(&']') {
                    self.stack.pop();
                    self.depth -= 1;
                    self.state = if self.depth == 0 { JsonParserState::Complete } else { JsonParserState::ExpectCommaOrClose };
                    true
                } else {
                    self.state = JsonParserState::ExpectValue;
                    self.consume_char(ch)
                }
            }
            JsonParserState::InObject => {
                self.state = JsonParserState::ExpectKey;
                self.consume_char(ch)
            }
            JsonParserState::Complete => ch.is_whitespace(),
            JsonParserState::Error => false,
        }
    }
}

/// Abstract grammar parser interface for token-by-token validation.
pub trait GrammarParser {
    fn is_accepted(&self) -> bool;
    fn advance(&mut self, token_id: u32, piece: &str) -> Result<(), String>;
}

/// JSON object-specific grammar parser compatible with `GrammarParser`.
#[derive(Debug, Clone, Default)]
pub struct JsonObjectGrammar {
    inner: JsonGrammar,
}

impl JsonObjectGrammar {
    pub fn new() -> Self {
        Self {
            inner: JsonGrammar::new(),
        }
    }
}

impl GrammarParser for JsonObjectGrammar {
    fn is_accepted(&self) -> bool {
        self.inner.is_complete()
    }

    fn advance(&mut self, _token_id: u32, piece: &str) -> Result<(), String> {
        if self.inner.is_complete() {
            if piece.trim().is_empty() {
                return Ok(());
            }
            return Err("Cannot append tokens after JSON completion".to_string());
        }
        if self.inner.advance(piece) && self.inner.state != JsonParserState::Error {
            Ok(())
        } else {
            Err(format!("Syntax error advancing grammar with piece {:?}", piece))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_json_object_transitions() {
        let mut grammar = JsonGrammar::new();
        assert!(grammar.is_token_valid("{\"name\":"));
        grammar.advance("{\"name\":");
        assert!(grammar.is_token_valid("\"get_weather\","));
        grammar.advance("\"get_weather\",");
        assert!(grammar.is_token_valid("\"arguments\":{\"city\":\"Paris\"}}"));
        grammar.advance("\"arguments\":{\"city\":\"Paris\"}}");
        assert!(grammar.is_complete());
    }

    #[test]
    fn test_reject_invalid_json_transitions() {
        let mut grammar = JsonGrammar::new();
        grammar.advance("{\"name\":");
        assert!(!grammar.is_token_valid(":"));
        assert!(!grammar.is_token_valid(","));
    }

    #[test]
    fn test_schema_guided_allowed_keys() {
        let mut grammar = JsonGrammar::inside_object().with_keys(&["city", "weather"]);
        assert!(grammar.is_token_valid("\"city\":"));
        assert!(!grammar.is_token_valid("\"invalid_key\":"));
    }
}
