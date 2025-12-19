# AttnBench

Benchmarking dense and sparse attention mechanisms using Swift and [MLX](https://github.com/ml-explore/mlx-swift) on Apple Silicon.

## Overview

AttnBench provides comprehensive benchmarks comparing different attention variants used in transformer models, from standard dense attention to novel sparse/efficient mechanisms. This is valuable for understanding the performance characteristics of Apple Silicon (M1/M2/M3/M4) when running different attention patterns.

### Dense Attention (O(N²) complexity)

- **MHA (Multi-Head Attention)**: Standard attention where each head has its own K and V projections
- **GQA (Grouped Query Attention)**: K and V are shared across groups of query heads
- **MQA (Multi-Query Attention)**: All query heads share a single K and V head

### Sparse/Efficient Attention (Sub-quadratic complexity)

- **Sliding Window Attention (SWA)**: O(N·W) complexity - only attends to a local window around each position
- **Block-Sparse Attention**: Divides the attention matrix into blocks, computing only local + global patterns
- **Linear Attention**: O(N) complexity via the kernel trick: Q(K^TV) instead of (QK^T)V
- **Causal Linear Attention**: O(N) causal attention using cumulative sums

## Why This Matters

### Research Value

Most "sliding window" implementations in high-level frameworks are **fake** - they just mask the full N×N matrix (still O(N²) compute). This benchmark implements **true** efficient attention using tensor slicing and gather operations.

Key questions this benchmark helps answer:

1. **At what sequence length does SWA become faster than MHA on M-series chips?**
2. **Do M4 AMX units prefer dense matmuls or batched small matmuls?**
3. **What's the overhead of linear attention vs the quadratic savings crossover?**
4. **How well does MLX handle gather/slice operations vs dense compute?**

### Novel Implementations

| Mechanism | Complexity | MLX Challenge |
|-----------|------------|---------------|
| Sliding Window | O(N·W) | Tests `gather`/`slice` performance vs dense on AMX |
| Block-Sparse | O(N·B + N·G) | Tests batched small matmuls vs one giant matmul |
| Linear Attention | O(N·D²) | Tests numerical stability with feature maps |
| Causal Linear | O(N·D²) | Tests `cumsum` scan operation efficiency |

## Requirements

- macOS 13.3 or later
- Apple Silicon (M1/M2/M3/M4)
- Xcode (required for Metal shader compilation)
- CMake and Ninja (for building)

## Installation

### Install Build Tools

```bash
brew install cmake ninja
```

### Install Xcode

Xcode is required to compile Metal shaders. Install it from the Mac App Store, then:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
```

## Building

```bash
mkdir build && cd build
cmake .. -G Ninja
ninja
```

## Running

```bash
./build/AttnBench
```

### Sample Output

```
name,b,n,dModel,heads,kvHeads,iters,msPerIter
MHA,1,128,1024,16,16,50,1.4981
GQA(kv=4),1,128,1024,16,4,50,1.1278
MQA(kv=1),1,128,1024,16,1,50,1.3023
SWA(w=32),1,128,1024,16,16,50,2.1456
SWA(w=64),1,128,1024,16,16,50,2.8934
BlockSparse(bs=32),1,128,1024,16,16,50,1.7823
LinearAttn,1,128,1024,16,16,50,0.9234
CausalLinearAttn,1,128,1024,16,16,50,1.4521
...
```

### Output Columns

| Column | Description |
|--------|-------------|
| `name` | Attention mechanism type |
| `b` | Batch size |
| `n` | Sequence length |
| `dModel` | Model dimension |
| `heads` | Number of query heads |
| `kvHeads` | Number of key/value heads |
| `iters` | Number of iterations |
| `msPerIter` | Milliseconds per iteration |

## Attention Mechanisms Explained

### Sliding Window Attention (SWA)

Instead of computing the full N×N attention matrix:
```
Standard: scores[i, j] for all i, j ∈ [0, N)  → O(N²)
```

SWA only computes scores within a window:
```
SWA: scores[i, j] for j ∈ [i - W/2, i + W/2]  → O(N·W)
```

**Implementation**: Uses padding + slicing to construct overlapping windows, avoiding the full matrix computation. This tests whether MLX's gather operations are efficient enough on Apple Silicon.

### Block-Sparse Attention

Divides the sequence into blocks and computes:
1. **Local attention**: Each block attends only to itself
2. **Global attention**: First G tokens attend to/from all positions (sink tokens)

```
┌───┬───┬───┬───┐
│ ■ │   │   │ ■ │  ← Global tokens attend everywhere
├───┼───┼───┼───┤
│ ■ │ ■ │   │   │  ← Local block attention
├───┼───┼───┼───┤
│ ■ │   │ ■ │   │
├───┼───┼───┼───┤
│ ■ │   │   │ ■ │
└───┴───┴───┴───┘
```

**Implementation**: Reshapes into blocks `[B, H, N/Block, Block, D]` and performs block-wise matmuls. Tests whether M-series chips prefer fewer large kernels or many small ones.

### Linear Attention

Rewrites softmax attention using kernel feature maps:
```
Standard: Attn = softmax(QK^T / √d) @ V          → O(N²)
Linear:   Attn ≈ φ(Q) @ (φ(K)^T @ V) / norm     → O(N·D²)
```

The key insight: computing `φ(K)^T @ V` first gives [D, D] instead of [N, N].

**Feature map**: Uses ELU(x) + 1 to ensure non-negative values (required for valid attention weights).

### Causal Linear Attention

For autoregressive (causal) attention, each position only attends to past positions:
```
output[i] = Σⱼ≤ᵢ (φ(k[j]) ⊗ v[j]) @ φ(q[i])
```

**Implementation**: Uses `cumsum` to accumulate KV outer products efficiently. Tests MLX's scan operation performance.

## Running Tests

Tests are run via `xcodebuild` (not CMake):

```bash
xcodebuild test -scheme AttnBench-Package -destination 'platform=OS X'
```

## Project Structure

```
attn-bench/
├── CMakeLists.txt              # CMake build configuration
├── Package.swift               # Swift Package Manager manifest
├── LICENSE                     # MIT License
├── Sources/
│   ├── AttnBench/
│   │   └── AttnBench.swift     # Main executable entry point
│   └── AttnBenchLib/
│       └── Attention.swift     # Core attention implementations
│           ├── Dense: MHA, GQA, MQA
│           ├── SlidingWindowAttention
│           ├── BlockSparseAttention
│           ├── LinearAttention
│           └── CausalLinearAttention
└── Tests/
    └── AttnBenchTests/
        └── AttnBenchTests.swift # Unit tests (30+ tests)
```

## Configuration

Modify benchmark parameters in `Sources/AttnBench/AttnBench.swift`:

```swift
let b = 1                              // Batch size
let dModel = 1024                      // Model dimension
let heads = 16                         // Number of attention heads
let seqs = [128, 256, 512, 1024]       // Sequence lengths to test
let iters = 50                         // Iterations per benchmark
let warmup = 10                        // Warmup iterations

// Sparse attention parameters
let windowSizes = [32, 64, 128]        // Sliding window sizes
let blockSizes = [32, 64]              // Block sizes for block-sparse
```

## Interpreting Results

### Expected Patterns

1. **Short sequences (N < 256)**: Dense attention (MHA) is usually fastest due to M-series AMX efficiency with large matrices
2. **Long sequences (N > 512)**: Sparse methods should start winning
3. **Linear attention**: Constant-ish time regardless of N, but with D² overhead
4. **Sliding window**: Should show clear O(N·W) scaling

### Key Metrics to Watch

- **Crossover point**: Where SWA becomes faster than MHA
- **Block size sensitivity**: Optimal block size for block-sparse
- **Linear attention overhead**: Fixed cost that determines when it's worthwhile

## References

- [Sliding Window Attention](https://www.abhik.xyz/concepts/attention/sliding-window-attention)
- [Block-Sparse Attention (MIT-HAN Lab)](https://github.com/mit-han-lab/Block-Sparse-Attention)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- [Linear Transformers (Katharopoulos et al.)](https://arxiv.org/abs/2006.16236)

## License

MIT

## Acknowledgments

- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's machine learning framework for Apple Silicon
- Inspired by research on efficient attention mechanisms for long-context LLMs