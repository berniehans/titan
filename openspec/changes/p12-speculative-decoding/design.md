# Design: Phase 12 — Speculative Decoding Engine (Draft Verification & Multi-Token Speculation)

## 1. Speculative Decoding Cycle

```
Current Sequence: [x_0, x_1, ..., x_t]
                  │
                  ▼
  ┌──────────────────────────────┐
  │ Step 1: Draft Proposer       │
  │ Proposes K candidates:       │
  │ [x_{t+1}, x_{t+2}, x_{t+3}]  │
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Step 2: Batched Target Forward Pass (M = K)                │
  │ • Evaluates all K candidates concurrently on GPU           │
  │ • Produces K logit distributions: [p_1, p_2, p_3]          │
  └──────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Step 3: Verification & Rejection Sampling                  │
  │ • Compare target predictions with proposed candidates      │
  │ • Find first mismatch index m <= K                         │
  │ • Accept [x_{t+1}, ..., x_{t+m}], Sample x_{t+m+1}         │
  └──────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Step 4: KV Cache Commit & State Advance                    │
  │ • Commit m + 1 tokens to PagedKvCache                      │
  │ • Advance pos by m + 1 (average 2.5 - 3.5 tokens/step)     │
  └────────────────────────────────────────────────────────────┘
```

---

## 2. Verification & Rejection Sampling Algorithm

For candidate sequence $\hat{x}_1, \dots, \hat{x}_K$:
1. Target model computes logits for positions $t, t+1, \dots, t+K-1$.
2. In greedy mode (temperature = 0):
   - For $i = 1 \dots K$:
     - Let $x_i^* = \operatorname{argmax}(P_{\text{target}}^{(i-1)})$.
     - If $x_i^* == \hat{x}_i$: accept $\hat{x}_i$.
     - Else: accept $x_i^*$ as correction, and stop verification.
   - If all $K$ candidates match: sample extra token $x_{K+1}^* = \operatorname{argmax}(P_{\text{target}}^{(K)})$.
3. In stochastic mode (temperature > 0):
   - Accept candidate $\hat{x}_i$ with probability $\min\left(1, \frac{P_{\text{target}}(\hat{x}_i)}{P_{\text{draft}}(\hat{x}_i)}\right)$.
   - Upon rejection, sample replacement token from normalized positive residue $\max(0, P_{\text{target}} - P_{\text{draft}})$.

---

## 3. Paged KV Cache Rollback & Commit

- `PagedKvCache` allocates block slots sequentially.
- When verifying $K$ candidate tokens, temporary KV representations are appended at positions $t, \dots, t+K-1$.
- If $m < K$ tokens are accepted, the driver advances sequence length by only $m + 1$, allowing subsequent steps to overwrite uncommitted slots $t+m+1 \dots t+K-1$ without memory fragmentation.
