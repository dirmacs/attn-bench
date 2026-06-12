# AttnBench

<div align="center">

**Benchmarking Sparse and Efficient Attention Mechanisms on Apple Silicon**

*A research project by [Dirmacs Labs](https://dirmacs.com), DIRMACS*

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![MLX](https://img.shields.io/badge/MLX-Swift-blue.svg)](https://github.com/ml-explore/mlx-swift)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-black.svg)](https://www.apple.com/mac/)

</div>

---

## Authors

**Baalateja Kataru** · **Suprabhat Rapolu** · **Dhruv Sidhu** · **Shanjeth Gobinath**

---

## Overview

AttnBench is a comprehensive, statistically rigorous benchmarking framework for evaluating attention mechanisms on Apple Silicon. Implemented in Swift using [MLX](https://github.com/ml-explore/mlx-swift), it challenges common assumptions about "efficient" attention and reveals surprising performance characteristics unique to Apple's unified memory architecture.

### 🔬 What We Discovered

Our benchmarks reveal that **conventional wisdom about sparse attention doesn't apply to Apple Silicon**:

| Finding | Result | Implication |
|---------|--------|-------------|
| 🚫 **Gather overhead is prohibitive** | 5.8–7.2× slower than masked dense | Don't use gather-based sparse kernels |
| ✅ **Block-sparse excels at scale** | 1.9× speedup at N=1024 | Use for long sequences (>512 tokens) |
| ✅ **Linear attention is viable** | 1.6× speedup at N=1024 | Good for very long contexts |
| 📈 **Overhead grows with sequence length** | 5.7× → 7.3× as N increases | The problem gets worse, not better |
| 🎯 **No single winner** | Optimal mechanism varies by N | Use adaptive selection |

### 📊 Visual Analysis Highlights

Our seven publication-quality figures reveal striking patterns:

- **Heatmap pattern**: Gather-based SWA appears as **uniformly deep red** (0.11–0.15× speedup) across all sequence lengths—a dramatic visual outlier
- **Crossover trajectory**: Block-sparse transitions from slower-than-MHA (red) at N=128 to nearly 2× faster (bright green) at N=1024
- **L2 cache boundary**: MHA scaling shows a sharp "elbow" at N=512→1024, coinciding with attention matrices exceeding ~4MB L2 cache

---

## Key Results

### The Gather Overhead Problem

On Apple Silicon, "true" sparse attention using gather/slice operations is **dramatically slower** than simply computing the full attention matrix and masking it:

```
┌─────────────────────────────────────────────────────────┐
│  Gather-based SWA vs Masked SWA                         │
├─────────────────────────────────────────────────────────┤
│  N=128:  SWA(gather) = 7.08ms  vs  MaskedSWA = 1.24ms  │
│          Overhead: 5.7×                                 │
│                                                         │
│  N=256:  SWA(gather) = 12.35ms vs  MaskedSWA = 1.69ms  │
│          Overhead: 7.3× (INCREASES with sequence!)      │
└─────────────────────────────────────────────────────────┘
```

**Why?** Apple's AMX matrix units achieve near-peak throughput for contiguous memory accesses. The overhead of gather operations (non-contiguous memory access, cache misses) exceeds any computational savings from computing fewer elements.

### Block-Sparse: The Real Winner

Block-sparse attention provides substantial speedups for long sequences:

| Sequence Length | MHA | BlockSparse (bs=32) | Speedup | p-value |
|-----------------|-----|---------------------|---------|---------|
| N=128 | 1.04 ms | 1.31 ms | 0.79× | <0.01 |
| N=256 | 1.42 ms | 1.63 ms | 0.87× | 0.02 |
| N=512 | 1.72 ms | 1.71 ms | **1.01×** | 0.89 |
| N=1024 | 4.38 ms | 2.31 ms | **1.90×** | <0.001 |

**Crossover point**: N ≈ 400–600 (clearly visible in our figures)

### Optimal Mechanism by Sequence Length

```
┌────────────────────────────────────────────────────────────────┐
│  Sequence Length    │  Best Mechanism      │  Latency          │
├─────────────────────┼──────────────────────┼───────────────────┤
│  N ≤ 128            │  GQA (kv=4)          │  0.87 ± 0.04 ms   │
│  128 < N ≤ 256      │  MaskedSWA (w=128)   │  1.34 ± 0.07 ms   │
│  256 < N ≤ 512      │  MQA (kv=1)          │  1.59 ± 0.08 ms   │
│  N > 512            │  BlockSparse (bs=32) │  2.31 ± 0.12 ms   │
└────────────────────────────────────────────────────────────────┘
```

---

## Attention Mechanisms

### Dense Attention (O(N²) complexity)

| Mechanism | Description | KV Heads | Best Use Case |
|-----------|-------------|----------|---------------|
| **MHA** | Multi-Head Attention (baseline) | h | General purpose |
| **GQA** | Grouped Query Attention | h/g | Short sequences, memory-constrained |
| **MQA** | Multi-Query Attention | 1 | KV cache reduction (87.5% savings) |

### Sparse/Efficient Attention

| Mechanism | Complexity | Description | Recommendation |
|-----------|------------|-------------|----------------|
| **SWA (Gather)** | O(N·W) | True sparse via gather/slice | ❌ Avoid on Apple Silicon |
| **MaskedSWA** | O(N²) | Dense + band mask | ✅ Use instead of gather |
| **BlockSparse** | O(N·B) | Block-local + global tokens | ✅ Best for N > 512 |
| **LinearAttn** | O(N·D²) | Kernel trick linearization | ✅ Very long contexts |
| **CausalLinear** | O(N·D²) | Causal via cumsum | ⚠️ Growing overhead with N |

---

## Statistical Methodology

AttnBench employs rigorous statistical methodology that goes beyond single-run measurements:

```
For each configuration:
  For run in 1..5:           # 5 independent runs
    Warmup: 5 iterations     # Excluded from timing
    Measure: 20 iterations   # Individual timing per iteration
    
  Total: 100 measurements per configuration
  Report: Mean ± 95% CI (t-distribution)
  Significance: Welch's t-test
  Reproducibility: CV < 6% for all mechanisms
```

### Why This Matters

- **Thermal throttling**: Single runs are affected by CPU/GPU temperature
- **System variability**: Background processes affect timing
- **Confidence intervals**: Quantify measurement reliability
- **Significance testing**: Distinguish real effects from noise

---

## Requirements

- **macOS** 13.3+ (tested on macOS Tahoe 26.2)
- **Apple Silicon** (M1/M2/M3/M4)
- **Xcode** (required for Metal shader compilation)
- **CMake** and **Ninja** (for building)
- **Python 3.10+** (for analysis)
- **Typst** (optional, for compiling the paper)

### Hardware Tested

| Parameter | Value |
|-----------|-------|
| Hardware | Apple M4 Mac Mini |
| Memory | 16 GB Unified |
| CPU Cores | 10 (4P + 6E) |
| GPU Cores | 10 |
| OS | macOS Tahoe 26.2 |

---

## Quick Start

### 1. Install Dependencies

```bash
# Build tools
brew install cmake ninja

# Xcode Metal toolchain
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

# Python dependencies (for analysis)
pip install -r analysis/requirements.txt

# Typst (optional, for paper compilation)
brew install typst
```

### 2. Build

```bash
cmake -B build -G Ninja
cmake --build build
```

### 3. Run Benchmarks

```bash
./build/AttnBench > data/benchmark_results.csv
```

### 4. Generate Figures

```bash
python analysis/analyze_benchmarks.py \
  --input data/benchmark_results.csv \
  --output figures/
```

### 5. Compile Paper (Optional)

```bash
typst compile paper/paper.typ paper/paper.pdf
```

---

## Project Structure

```
attn-bench/
├── Sources/
│   ├── AttnBench/
│   │   └── AttnBench.swift      # Benchmark driver
│   └── AttnBenchLib/
│       └── Attention.swift      # All attention implementations
├── Tests/
│   └── AttnBenchTests/          # 33 unit tests
├── analysis/
│   ├── analyze_benchmarks.py    # Statistical analysis + figures
│   └── requirements.txt         # Python dependencies
├── data/
│   └── benchmark_results.csv    # Raw benchmark output
├── docs/
│   ├── BLOG_POST.md             # Technical blog post
│   ├── CONTRIBUTING.md          # Contribution guidelines
│   └── LINKEDIN_POST.md         # Social media content
├── figures/                     # Generated visualizations
│   ├── fig1_latency_scaling.pdf
│   ├── fig2_gather_vs_masked.pdf
│   ├── fig3_blocksparse_speedup.pdf
│   ├── fig4_linear_attention.pdf
│   ├── fig5_heatmap.pdf
│   ├── fig6_dense_variants.pdf
│   ├── fig7_scaling_analysis.pdf
│   └── statistical_summary.txt
├── paper/
│   ├── paper.typ                # Research paper (Typst)
│   ├── paper.pdf                # Compiled paper
│   └── references.bib           # Bibliography
├── CMakeLists.txt               # CMake build
├── Package.swift                # Swift Package Manager
└── LICENSE                      # MIT License
```

---

## Generated Outputs

### Figures

| Figure | Description |
|--------|-------------|
| `fig1_latency_scaling.pdf` | Latency vs. sequence length for all mechanisms |
| `fig2_gather_vs_masked.pdf` | Gather overhead analysis (the 5.8–7.2× slowdown) |
| `fig3_blocksparse_speedup.pdf` | Block-sparse crossover analysis |
| `fig4_linear_attention.pdf` | Linear vs quadratic scaling comparison |
| `fig5_heatmap.pdf` | Performance heatmap (mechanisms × sequence lengths) |
| `fig6_dense_variants.pdf` | MHA vs GQA vs MQA comparison |
| `fig7_scaling_analysis.pdf` | Log-log complexity analysis |

### Data Files

| File | Description |
|------|-------------|
| `statistical_summary.txt` | Full statistical report with key findings |
| `table_results.tex` | LaTeX table for papers |
| `benchmark_stats.json` | Machine-readable statistics |

---

## Practical Recommendations

### For Inference on Apple Silicon

| Scenario | Recommendation |
|----------|----------------|
| Short contexts (N ≤ 128) | Use **GQA** with 4 KV heads for 20% speedup |
| Medium contexts (128 < N ≤ 512) | Standard **MHA** or **MQA**; differences are marginal |
| Long contexts (N > 512) | Use **BlockSparse** (bs=32) for up to 90% speedup |
| Very long contexts (N >> 1024) | Consider **LinearAttn** if approximation is acceptable |
| Sliding window needed | **Always use masked implementation**, never gather-based |
| Memory-constrained | Use **MQA** for 87.5% KV cache reduction |

### For Framework Developers

1. **Default to masked-dense implementations** — Gather-based sparse kernels should be opt-in, not default
2. **Implement adaptive dispatch** — Select mechanism based on sequence length at runtime
3. **Tune block sizes per chip** — Our bs=32 > bs=64 finding may differ on M1/M2/M3
4. **Profile cumsum operations** — Causal linear attention overhead grows with sequence length

### For Model Architects

1. **GQA over MQA for short-context applications** — GQA provides consistent speedups; MQA has short-sequence overhead
2. **Consider hybrid architectures** — Dense attention in early layers, sparse in later layers
3. **Window size is free in masked SWA** — Choose based on model quality, not performance

---

## Running Tests

```bash
xcodebuild test -scheme AttnBench-Package -destination 'platform=OS X'
```

All 33 tests verify:
- Shape correctness for all mechanisms
- Finite outputs (no NaN/Inf)
- Mechanism naming conventions
- Statistical utility functions

---

## Citation

If you use AttnBench in your research, please cite:

```bibtex
@misc{attnbench2024,
  title={AttnBench: Benchmarking Sparse and Efficient Attention 
         Mechanisms on Apple Silicon},
  author={Kataru, Baalateja and Rapolu, Suprabhat and 
          Sidhu, Dhruv and Gobinath, Shanjeth},
  year={2024},
  institution={Dirmacs Labs, DIRMACS},
  howpublished={\url{https://github.com/dirmacs/attn-bench}}
}
```

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [Longformer](https://arxiv.org/abs/2004.05150) — Beltagy et al., 2020
- [BigBird](https://arxiv.org/abs/2007.14062) — Zaheer et al., 2020
- [Linear Transformers](https://arxiv.org/abs/2006.16236) — Katharopoulos et al., 2020
- [GQA](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [MLX](https://github.com/ml-explore/mlx) — Apple ML Research

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [MLX Swift](https://github.com/ml-explore/mlx-swift) — Apple's ML framework for Apple Silicon
- **Dirmacs Labs** for supporting this research initiative

---

<div align="center">

**[Dirmacs Labs](https://dirmacs.com)** — Exploring cutting-edge technologies from hardware to ML systems

</div>