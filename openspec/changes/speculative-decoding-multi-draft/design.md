# Design: Speculative Decoding Multi-Draft (Llama 3.2 1B -> 3B)

## Architecture Overview
The speculative decoding architecture runs two ForwardDriver instances simultaneously in GPU memory:
1. draft_driver: Llama 3.2 1B Instruct Q4_K_M (807 MB VRAM, 16 layers, 170.4 tok/s).
2. 	arget_driver: Llama 3.2 3B Instruct Q4_K_M (2,020 MB VRAM, 28 layers, 68.9 tok/s).

## Verification Strategy
- **$-Token Proposed Chunk:**  \in \{3, 4, 5\}$ (default =4$).
- **Parallel Target Evaluation:** Input embeddings $[K, 3072]$ are multiplied through batched GEMV kernels with =K$, producing $[K, 128256]$ output logits in a single target GPU pass.
- **Acceptance Probability:** Rejection sampler evaluates $\min(1, P_{\text{target}}(x_i) / P_{\text{draft}}(x_i))$. Under greedy decoding (=0$), token $ is accepted if $\text{argmax}(P_{\text{target}}(x_i)) == x_i$.
- **Fast KV Sync:** If $ tokens are accepted, the target KV-cache advances by $, and the draft model's KV-cache is synchronized to match the target context prefix via radix tree match.
