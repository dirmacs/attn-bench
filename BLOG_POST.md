# Why Your "Efficient" Attention Is Actually Slower on Apple Silicon

*Comprehensive benchmarking reveals surprising performance characteristics that challenge conventional wisdom*

**Authors**: Baalateja Kataru, Suprabhat Rapolu, Dhruv Sidhu, Shanjeth Gobinath  
**Affiliation**: Dirmacs Labs, DIRMACS

---

We spent weeks building [AttnBench](https://github.com/dirmacs/attn-bench), a statistically rigorous benchmarking framework to test attention mechanisms on Apple Silicon using Swift and MLX. The results challenged nearly everything we assumed about "efficient" attention.

**The Bottom Line**: On Apple Silicon, computing a full N×N attention matrix and masking it is *dramatically faster* than using "true" sparse operations. The overhead isn't small—it's **5.8–7.2×** slower to use gather-based sparsity. And that overhead *increases* with sequence length.

---

## The Motivation

Everyone talks about efficient attention. Sliding window, sparse patterns, linear attention—all promising to break the O(N²) barrier. But here's the problem: **most benchmarks target NVIDIA GPUs**. 

Apple's M-series chips have a fundamentally different architecture:

- **Unified Memory**: CPU and GPU share the same RAM—no PCIe bottleneck
- **AMX Units**: Dedicated matrix multiplication hardware optimized for dense operations
- **Metal**: Apple's GPU compute framework with different optimization priorities

We wanted to know: *Do efficient attention mechanisms actually help on this hardware?*

The answer surprised us.

---

## What We Tested

We implemented **eight attention variants** in Swift/MLX with rigorous statistical methodology:

### Dense Baselines
| Mechanism | Description | KV Heads |
|-----------|-------------|----------|
| **MHA** | Standard multi-head attention | 8 |
| **GQA** | Grouped-query attention | 4 |
| **MQA** | Multi-query attention | 1 |

### "Efficient" Variants
| Mechanism | Complexity | Description |
|-----------|------------|-------------|
| **SWA (Gather)** | O(N·W) | True sparse using gather/slice |
| **MaskedSWA** | O(N²) | Dense matrix + band mask |
| **BlockSparse** | O(N·B) | Block-local + global tokens |
| **LinearAttn** | O(N·D²) | Kernel trick linearization |
| **CausalLinear** | O(N·D²) | Causal via cumulative sums |

### Statistical Rigor

Unlike typical benchmarks that report single measurements, we used:
- **5 independent runs** per configuration
- **20 iterations per run** (100 total measurements)
- **95% confidence intervals** using t-distribution
- **Welch's t-tests** for significance testing
- **Coefficient of variation < 6%** for all mechanisms

---

## The Results

Here's what we measured on an M4 Mac Mini (16GB unified memory):

| Mechanism | N=128 | N=256 | N=512 | N=1024 |
|-----------|-------|-------|-------|--------|
| MHA | 1.04 ms | 1.42 ms | 1.72 ms | **4.38 ms** |
| GQA (kv=4) | **0.87 ms** | 1.37 ms | 1.59 ms | 4.20 ms |
| MQA (kv=1) | 1.12 ms | 1.37 ms | 1.59 ms | 3.97 ms |
| SWA (gather) | 7.08 ms | 12.35 ms | — | — |
| MaskedSWA | 1.24 ms | 1.69 ms | 1.84 ms | 5.10 ms |
| BlockSparse (bs=32) | 1.31 ms | 1.63 ms | 1.71 ms | **2.31 ms** |
| LinearAttn | 1.62 ms | 1.82 ms | 2.01 ms | **2.75 ms** |
| CausalLinear | 2.64 ms | 3.18 ms | 5.94 ms | — |

---

## Finding #1: Gather Operations Are Catastrophically Slow

This was our most striking finding. "True" sliding window attention—where you only compute attention scores within a window using gather/slice operations—was **5.8–7.2× slower** than computing the full matrix and masking it.

```
N=128, Window=64:
  SWA (gather):  7.08 ms
  MaskedSWA:     1.24 ms
  Overhead:      5.7×

N=256, Window=64:
  SWA (gather):  12.35 ms
  MaskedSWA:      1.69 ms
  Overhead:      7.3×  ← Gets WORSE with longer sequences!
```

### Why Does This Happen?

Apple's AMX units are incredibly efficient at dense matrix operations with contiguous memory access. The overhead of:

1. **Slicing** N windows from key/value tensors
2. **Stacking** them into a new tensor  
3. **Running** attention on windowed views

...dramatically exceeds the compute savings from doing less math. The gather operations cause cache misses and memory indirection that the AMX units can't optimize around.

### Visual Confirmation

Our performance heatmap (Figure 5) shows this dramatically: gather-based SWA appears as **uniformly deep red** across all sequence lengths, with speedup values of just 0.11–0.15× relative to MHA. It's a visual outlier—nothing else performs this poorly.

**The Takeaway**: On Apple Silicon, "fake" sparse attention (dense + mask) beats "true" sparse attention (gather/slice) by a massive margin. If your ML framework implements sliding window by masking a full attention matrix, that's actually *optimal* for this hardware.

---

## Finding #2: The Overhead Increases with Sequence Length

Perhaps even more surprising: the gather overhead **grows** as sequences get longer.

| Sequence Length | Gather Overhead |
|-----------------|-----------------|
| N=128 | 5.7× |
| N=256 | 7.3× |

This is counterintuitive—you'd expect sparse methods to become *more* beneficial at longer sequences where O(N²) hurts more. But on Apple Silicon, the opposite happens:

- Longer sequences = more gather operations
- More gathers = more cache misses
- More cache misses = worse performance relative to dense

**Implication**: If gather-based sparse attention is bad at N=128, it's *worse* at N=256, and presumably even worse at longer sequences. This isn't a problem that goes away at scale—it compounds.

---

## Finding #3: Block-Sparse Is the Real Winner

Block-sparse attention divides the sequence into blocks and computes attention within each block, plus global tokens that attend everywhere:

```
┌───┬───┬───┬───┐
│ ■ │   │   │ ■ │  ← Global tokens attend to all
├───┼───┼───┼───┤
│ ■ │ ■ │   │   │  ← Local blocks on diagonal
├───┼───┼───┼───┤
│ ■ │   │ ■ │   │
├───┼───┼───┼───┤
│ ■ │   │   │ ■ │
└───┴───┴───┴───┘
```

At N=1024, block-sparse (with block size 32) achieved:

```
MHA:         4.38 ms
BlockSparse: 2.31 ms
Speedup:     1.90× (p < 0.001)
```

### The Crossover Point

Our Figure 3 clearly shows the crossover trajectory:

| N | BlockSparse vs MHA | Winner |
|---|-------------------|--------|
| 128 | 0.79× | MHA |
| 256 | 0.87× | MHA |
| 512 | 1.01× | Tie |
| 1024 | 1.90× | BlockSparse |

The crossover point is around **N ≈ 400–600**. Below this, MHA wins. Above it, block-sparse dominates.

### Why Block-Sparse Works

Unlike gather-based sparsity, block-sparse attention:

1. **Uses contiguous memory blocks** that fit in L2 cache
2. **Parallelizes naturally** across independent blocks
3. **Avoids gather operations** entirely—just reshape and compute
4. **Scales sublinearly** as sequence length increases

### Block Size Matters

We tested block sizes 32 and 64. Smaller blocks won:

| Sequence | bs=32 | bs=64 | Winner |
|----------|-------|-------|--------|
| N=128 | 1.31 ms | 1.43 ms | bs=32 (+9%) |
| N=256 | 1.63 ms | 1.80 ms | bs=32 (+10%) |
| N=512 | 1.71 ms | 1.81 ms | bs=32 (+6%) |
| N=1024 | 2.31 ms | 2.46 ms | bs=32 (+6%) |

Smaller blocks likely fit better in L1/L2 cache and enable better GPU parallelization across M4's 10-core GPU.

---

## Finding #4: Linear Attention Works (At Long Sequences)

Linear attention rewrites the attention formula to be O(N) instead of O(N²):

```
Standard:  Attention = softmax(QK^T / √d) @ V     → O(N²)
Linear:    Attention ≈ φ(Q) @ (φ(K)^T @ V)        → O(N)
```

The trick is computing `φ(K)^T @ V` first, which gives a D×D matrix instead of N×N.

At N=1024:
```
MHA:        4.38 ms
LinearAttn: 2.75 ms
Speedup:    1.59×
```

### The Crossover Is Later Than Block-Sparse

Linear attention doesn't become faster than MHA until around **N ≈ 768**:

| N | Linear vs MHA | Winner |
|---|---------------|--------|
| 128 | 0.64× | MHA |
| 256 | 0.78× | MHA |
| 512 | 0.86× | MHA |
| 1024 | 1.59× | Linear |

**Trade-off**: Linear attention is an approximation (we use the ELU+1 kernel). For tasks where exact attention matters, this may not be acceptable. But for very long contexts where you need O(N) scaling, it's viable.

---

## Finding #5: Causal Linear Attention Has Growing Overhead

For autoregressive generation (like LLMs), you need causal attention. The linear attention equivalent uses cumulative sums to ensure each position only attends to previous positions.

The problem: **the overhead grows with sequence length**.

| N | LinearAttn | CausalLinear | Overhead |
|---|------------|--------------|----------|
| 128 | 1.62 ms | 2.64 ms | 1.63× |
| 256 | 1.82 ms | 3.18 ms | 1.75× |
| 512 | 2.01 ms | 5.94 ms | 2.96× |

At N=512, causal linear attention is **3× slower** than non-causal. The cumsum operation creates:
- Sequential dependencies that hurt parallelization
- Large intermediate tensors for KV outer products
- Memory bandwidth pressure

**Recommendation**: For causal models on Apple Silicon, consider chunked approaches or stick with block-sparse attention instead of causal linear.

---

## Finding #6: No Single Mechanism Dominates

One of our most important findings: **the optimal mechanism changes with sequence length**.

| Sequence Length | Best Mechanism | Latency |
|-----------------|---------------|---------|
| N ≤ 128 | GQA (kv=4) | 0.87 ms |
| N ≈ 256 | MaskedSWA (w=128) | 1.34 ms |
| N ≈ 512 | MQA (kv=1) | 1.59 ms |
| N ≥ 1024 | BlockSparse (bs=32) | 2.31 ms |

This suggests production systems should implement **adaptive attention dispatch**—selecting the mechanism based on input sequence length at runtime.

---

## Finding #7: The L2 Cache Boundary

We observed a dramatic scaling change at N=512→1024:

| Transition | Latency Ratio | Expected (O(N²)) |
|------------|---------------|------------------|
| N=128→256 | 1.37× | 4× |
| N=256→512 | 1.21× | 4× |
| N=512→1024 | **2.55×** | 4× |

There's a sharp "elbow" in MHA scaling. Why?

For N=1024 with 8 heads: the attention matrix is 1024 × 1024 × 4 bytes ≈ **4 MB**. Apple M4's L2 cache is approximately **4 MB per performance core cluster**.

At N=1024, the attention matrix exceeds L2 cache, causing a transition from compute-bound to memory-bound execution. This validates the importance of cache-aware implementations like block-sparse.

---

## Finding #8: The MQA Paradox

Here's something counterintuitive: **MQA is slower than MHA at short sequences**.

| N | MHA | MQA | MQA speedup |
|---|-----|-----|-------------|
| 128 | 1.04 ms | 1.12 ms | **0.93×** (7% slower!) |
| 256 | 1.42 ms | 1.37 ms | 1.03× |
| 512 | 1.72 ms | 1.59 ms | 1.08× |
| 1024 | 4.38 ms | 3.97 ms | 1.10× |

MQA has only 1 KV head instead of 8, so it should be faster everywhere, right? Wrong.

**Explanation**: At short sequences, the broadcasting overhead for sharing a single KV head across 8 query heads exceeds the memory savings. Apple Silicon's unified memory doesn't penalize memory access as heavily as discrete GPU memory, reducing MQA's traditional advantage.

**When to use MQA**: For KV cache memory savings (87.5% reduction), especially in inference with large batch sizes. Accept the short-sequence latency overhead.

---

## Practical Recommendations

Based on our comprehensive benchmarking:

### ✅ For Short Sequences (N ≤ 128)
**Use GQA** with 4 KV heads. It provides 20% speedup over MHA with 50% memory reduction.

### ✅ For Medium Sequences (N ≤ 512)
**Use standard MHA or MQA**. Differences are marginal, so choose based on memory requirements.

### ✅ For Long Sequences (N > 512)
**Use Block-Sparse** with block size 32. Nearly 2× speedup at N=1024, scaling better as sequences grow.

### ✅ For Very Long Contexts
**Consider Linear Attention** if the approximation is acceptable for your task.

### ❌ Never Use Gather-Based Sparse Attention
On Apple Silicon, always use masked implementations instead. The gather overhead is prohibitive and gets worse at longer sequences.

### ❌ Avoid Causal Linear Attention
The cumsum overhead grows to 3× at N=512. Use block-sparse with causal masking instead.

### 🔄 Consider Adaptive Dispatch
Implement runtime selection of attention mechanism based on sequence length for optimal performance across all inputs.

---

## The Code

Here's the core insight for block-sparse attention:

```swift
// Reshape to blocks: [B, H, numBlocks, blockSize, Dh]
let qBlocks = q.reshaped([b, h, numBlocks, blockSize, dh])
let kBlocks = k.reshaped([b, h, numBlocks, blockSize, dh])
let vBlocks = v.reshaped([b, h, numBlocks, blockSize, dh])

// Block-local attention - no gather needed!
let scores = matmul(qBlocks, kBlocks.transposed()) * scale
let probs = softmax(scores, axis: -1)
let localOut = matmul(probs, vBlocks)
```

And linear attention's key insight:

```swift
// Feature map ensures non-negative attention weights
let phiQ = featureMap(q)  // ELU(x) + 1
let phiK = featureMap(k)

// Compute K^T @ V first: gives [D, D] instead of [N, N]
let kv = matmul(phiK.transposed(), v)  // [B, H, Dh, Dh]

// Then Q @ (K^T @ V): still O(N)
let output = matmul(phiQ, kv)  // [B, H, N, Dh]
```

---

## Why This Matters

As Apple Silicon becomes more prevalent for ML inference—especially on-device deployment—understanding its unique performance characteristics is crucial.

**The key insight**: Common wisdom from CUDA-land doesn't apply:

1. **Memory indirection is expensive** — Unified memory is great, but gather operations still kill performance
2. **AMX loves dense ops** — The matrix units are so efficient that sparse savings rarely compensate for overhead
3. **Cache efficiency matters** — Block-sparse wins partly because attention blocks fit in L2
4. **No silver bullet** — The optimal mechanism depends on sequence length

---

## Visual Analysis

Our seven publication-quality figures reveal patterns that reinforce these findings:

- **Figure 1 (Latency Scaling)**: All curves cluster at N=128, then dramatically diverge—SWA shoots up while block-sparse stays flat
- **Figure 2 (Gather vs Masked)**: The bar height difference is immediately striking—a 6× visual gap
- **Figure 3 (Block-Sparse Speedup)**: Clear crossover trajectory from red (slower) to green (faster) region
- **Figure 5 (Heatmap)**: SWA is uniformly deep red—a dramatic visual outlier against the gradient pattern of other mechanisms
- **Figure 7 (Scaling Analysis)**: MHA's slope is 1.73 (sub-quadratic), while block-sparse is 0.89 (near-linear)

---

## What's Next

We're planning to:
- **Extend to M1/M2/M3** to see how findings generalize across Apple Silicon generations
- **Implement Flash Attention-style tiling** in MLX
- **Benchmark training** (backward pass) performance
- **Explore longer sequences** with memory-efficient implementations
- **Test real models** end-to-end, not just attention in isolation

---

## Conclusion

If you're deploying transformer models on Apple Silicon:

1. **Don't assume "efficient" means faster** — Test on your actual hardware
2. **Block-sparse is your friend** for long sequences
3. **Masked > Gather** for any windowed attention pattern
4. **Consider adaptive dispatch** based on input characteristics

The full implementation, data, and paper are available at [AttnBench](https://github.com/dirmacs/attn-bench). Star the repo if you found this useful!

---

*This research was conducted at **Dirmacs Labs**, the R&D division of DIRMACS focused on exploring cutting-edge technologies from hardware-aware algorithm design to high-level ML systems.*

---

**Configuration used:**
- Hardware: Apple M4 Mac Mini, 16GB unified memory
- OS: macOS Tahoe 26.2
- Batch size: 1
- Model dimension: 512
- Attention heads: 8
- Head dimension: 64
- Statistical rigor: 5 runs × 20 iterations = 100 measurements per config

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [Longformer](https://arxiv.org/abs/2004.05150) — Beltagy et al., 2020
- [BigBird](https://arxiv.org/abs/2007.14062) — Zaheer et al., 2020
- [Linear Transformers](https://arxiv.org/abs/2006.16236) — Katharopoulos et al., 2020
- [GQA](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [MLX](https://github.com/ml-explore/mlx-swift) — Apple ML Research