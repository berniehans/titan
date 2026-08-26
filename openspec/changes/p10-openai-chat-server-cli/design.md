# Design: Phase 10 — OpenAI Chat Completions Server, Streaming SSE & Interactive CLI

## 1. Architecture & Component Interaction

```
[ HTTP Client / Web UI / Cursor ]
                │
                │ HTTP POST /v1/chat/completions
                ▼
      ┌────────────────────┐
      │    Axum Server     │ ──> Parses JSON / Applies ChatML template
      └─────────┬──────────┘
                │ tokio mpsc request channel
                ▼
      ┌────────────────────┐
      │   GPU EngineActor  │ (Dedicated background thread bound to CUDA context)
      │                    │
      │  • ForwardDriver   │ ──> Prefill prompt tokens
      │  • CUDA Graph Exec │ ──> Replay 28-layer graph per token
      │  • Sampler         │ ──> Top-p / Top-k / Temperature / Stop token trim
      └─────────┬──────────┘
                │ mpsc token stream channel
                ▼
      ┌────────────────────┐
      │ SSE Response Body  │ ──> data: {"choices": [{"delta": {"content": "..."}}]}\n\n
      └────────────────────┘
```

---

## 2. Wire Models & ChatML Templating

### Chat Message & Completion Request
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "system", "user", "assistant"
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
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
```

### ChatML Formatting
For Qwen3 models, chat messages are formatted as:
```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```
Special tokens:
- `<|im_start|>`: ID 151644
- `<|im_end|>`: ID 151645
- `<|endoftext|>`: ID 151643

When `<|im_end|>` or any token matching the stop sequences is emitted, generation terminates immediately with `finish_reason: "stop"`.

---

## 3. Production Sampler Algorithm

1. **Repetition Penalty:**
   For all previously generated tokens $t$, apply penalty $\theta$:
   $$\text{logit}[t] = \begin{cases} \text{logit}[t] / \theta & \text{if } \text{logit}[t] > 0 \\ \text{logit}[t] \times \theta & \text{if } \text{logit}[t] \le 0 \end{cases}$$
2. **Temperature Scaling:**
   If $\text{temperature} \le 1e-4$, select $\text{argmax}(\text{logits})$ (Greedy).
   Else, $\text{logits}[i] \leftarrow \text{logits}[i] / \text{temperature}$.
3. **Top-$K$ Filtering:**
   Keep only top $K$ largest logits; set all others to $-\infty$.
4. **Top-$P$ (Nucleus) Filtering:**
   Compute softmax probabilities $p_i = \frac{e^{\text{logit}[i]}}{\sum_j e^{\text{logit}[j]}}$.
   Sort probabilities descending, compute cumulative sum, and mask out items where cumulative sum exceeds $P$ (keeping at least the top 1 token).
5. **Sampling:**
   Sample token from the normalized remaining probability distribution using a pseudo-random RNG.

---

## 4. Interactive CLI (`titan chat`)

The `engine-server` binary will support subcommands:
- `titan serve --model <path> --port 8000 --capacity 2048`
- `titan chat --model <path>`:
  - Initializes `ForwardDriver` and loads the GGUF model into pinned memory.
  - Enters an interactive terminal loop reading user inputs with prompt line editing.
  - Maintains conversation history across multi-turn exchanges.
  - Prefills prompt tokens and streams response tokens directly to `std::io::stdout()` with immediate flushing.
