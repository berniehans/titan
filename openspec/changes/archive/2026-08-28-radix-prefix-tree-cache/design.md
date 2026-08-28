# Architecture Design: Radix Tree Prefix Caching (APC)

```
                            RADIX TREE PREFIX CACHING
                            
       Incoming Request: [System Prompt (500) | Tools (1000) | User Turn 2 (30)]
                                      │
                                      ▼
                        RadixTree::match_prefix(tokens)
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        Matched LCP Prefix (1500 tok)          Unmatched Suffix (30 tok)
        [Reused from KV Block Table]           [Forward Pass from pos=1500]
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                      Prefill Time: ~0.5 ms (instead of 25 ms)
                      Next Token Logits @ pos=1530
```

## Data Flow & Integration Points
1. `ForwardDriver` maintains an active `RadixTree`.
2. When `driver.prefill(tokens)` is called:
   - Compute `LCP = radix_tree.match_prefix(tokens)`.
   - If `LCP.matched_tokens > 0` and cached KV blocks match resident position:
     - Advance `driver.pos = LCP.matched_tokens`.
     - Prefill only `&tokens[LCP.matched_tokens..]`.
   - Update `RadixTree` with the new sequence.
