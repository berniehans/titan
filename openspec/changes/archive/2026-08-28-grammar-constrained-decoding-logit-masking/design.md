# Architecture Design: Grammar-Guided Constrained Decoding

```
                           CONSTRAINED SAMPLING PIPELINE
                           
       Raw GPU Logits (152,000 floats in VRAM)
                           │
                           ▼
          [ Grammar State Machine (DFA / JSON Schema) ]
                           │
                           ▼ Allowed Token Bitmask (19 KB in VRAM)
          [ apply_logit_mask_kernel (CUDA JIT) ]
          (Sets disallowed logits to -1e9f in <0.01 ms)
                           │
                           ▼ Masked GPU Logits
          [ GPU Argmax / Softmax Nucleus Sampler ]
                           │
                           ▼
           100% Guaranteed Syntactically Valid JSON Token
```
