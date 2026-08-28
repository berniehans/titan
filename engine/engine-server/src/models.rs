/// OpenAI-compatible request/response models for the engine-server.
///
/// The wire contract mirrors the standard OpenAI `/v1/chat/completions`,
/// `/v1/completions`, and `/v1/models` endpoints.
use serde::{Deserialize, Serialize};

/// Chat message in a multi-turn conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role identifier: "system", "user", or "assistant".
    pub role: String,
    /// Message text content.
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Formats a sequence of chat messages into Qwen ChatML prompt format.
pub fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&msg.role);
        prompt.push('\n');
        prompt.push_str(&msg.content);
        prompt.push_str("<|im_end|>\n");
    }
    // Append assistant prompt header to trigger response generation
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn default_max_tokens() -> u32 {
    256
}

/// Body of a `POST /v1/chat/completions` request.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier.
    #[serde(default)]
    pub model: Option<String>,
    /// Array of conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// When true, stream Server-Sent Events (SSE).
    #[serde(default)]
    pub stream: bool,
    /// Sampling temperature in [0.0, 2.0]. 0.0 is greedy argmax.
    pub temperature: Option<f32>,
    /// Top-p nucleus probability mass.
    pub top_p: Option<f32>,
    /// Top-k candidate filtering.
    pub top_k: Option<usize>,
    /// Repetition penalty factor.
    pub repetition_penalty: Option<f32>,
    /// Custom stop token strings.
    pub stop: Option<Vec<String>>,
    /// OpenAI tool definitions for function calling.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// Controls which (if any) tool is called by the model.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Specifies the format that the model must output (e.g. { "type": "json_object" }).
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
}

/// One non-streaming chat choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Non-streaming `POST /v1/chat/completions` response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: CompletionUsage,
}

/// Delta token payload inside streaming SSE chunks.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// One streaming chat chunk choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: DeltaMessage,
    pub finish_reason: Option<String>,
}

/// One SSE payload for a streaming chat completion chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// Body of a legacy `POST /v1/completions` request.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub stop: Option<Vec<String>>,
}

/// One non-streaming completion choice.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: String,
}

/// Token accounting for a completion response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Non-streaming `POST /v1/completions` response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
}

/// One streaming chunk choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// One SSE payload for a generated text token.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

/// Model info object for `GET /v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Response for `GET /v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_formatting() {
        let messages = vec![
            ChatMessage::system("You are a helpful coding assistant."),
            ChatMessage::user("Write hello in Rust"),
        ];
        let prompt = format_chatml(&messages);
        let expected = "<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\nWrite hello in Rust<|im_end|>\n<|im_start|>assistant\n";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_chat_request_deserialization() {
        let json = r#"{
            "model": "qwen3-0.6b",
            "messages": [
                {"role": "user", "content": "hello"}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": true
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.model.as_deref(), Some("qwen3-0.6b"));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "hello");
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.top_p, Some(0.9));
        assert!(req.stream);
    }

    #[test]
    fn test_chat_chunk_serialization() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1720000000,
            model: "qwen3-0.6b".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: DeltaMessage {
                    role: Some("assistant".to_string()),
                    content: Some("Hello".to_string()),
                },
                finish_reason: None,
            }],
        };
        let s = serde_json::to_string(&chunk).expect("serialize");
        assert!(s.contains("\"role\":\"assistant\""));
        assert!(s.contains("\"content\":\"Hello\""));
    }
}
