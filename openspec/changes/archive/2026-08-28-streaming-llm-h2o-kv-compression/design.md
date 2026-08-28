# Architecture Design: StreamingLLM & H2O KV Compression

```
                          STREAMINGLLM + H2O KV LAYOUT
                          
Logical Tokens: [ 0 .. 3 ] ... [ 4 ................. N-W-1 ] ... [ N-W .. N ]
                ─────────      ────────────────────────────      ───────────
Category:       Attention      Evictable Middle Tokens /         Recent Context
                Sinks (Pinned) Heavy-Hitters                     Rolling Window
                
Physical VRAM:  [ Block 0 ] ──► [ Top-K H2O Heavy Blocks ] ──► [ Recent Blocks ]
                (Always in VRAM)  (Dynamic Reclaimed)          (Rolling Sliding Window)
                
Total VRAM Budget: Constant O(1) (e.g. 512 tokens = 64 MB for 1.5B model)
```
