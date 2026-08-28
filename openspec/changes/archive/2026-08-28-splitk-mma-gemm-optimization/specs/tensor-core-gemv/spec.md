## ADDED Requirements

### Requirement: Split-K Reduction Dimension Partitioning
The GPU execution runtime SHALL partition large matrix reduction dimensions ($K \ge 2048$) across multiple threadblocks ($S_K \in \{2, 4, 8\}$) to maximize Streaming Multiprocessor (SM) occupancy during decode projections.

#### Scenario: Split-K quantized linear projection on 3B model
- **WHEN** executing a linear projection where $K \ge 2048$ (e.g. $d_{\text{hidden}} = 3072$ in Llama 3.2 3B)
- **THEN** computation is partitioned into $S_K$ parallel block slices and reduced into the target accumulator buffer with numerical parity ($\epsilon < 10^{-4}$).

### Requirement: Dynamic Split-K Scaling and Fallback
The runtime SHALL select between standard single-block GEMV and Split-K GEMV based on the ratio of output columns $N$ to reduction length $K$.

#### Scenario: Sub-1B model execution bypass
- **WHEN** evaluating layers with $K \le 1536$ (e.g. Qwen 0.6B or 1.5B)
- **THEN** the single-warp direct reduction kernel is selected to preserve ultra-low dispatch latency without extra reduction workspace overhead.
