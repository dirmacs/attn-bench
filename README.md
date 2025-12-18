# AttnBench

Benchmarking different attention mechanisms (MHA, GQA, MQA) using Swift and [MLX](https://github.com/ml-explore/mlx-swift) on Apple Silicon.

## Overview

AttnBench compares the performance of three attention variants commonly used in transformer models:

- **MHA (Multi-Head Attention)**: Standard attention where each head has its own K and V projections
- **GQA (Grouped Query Attention)**: K and V are shared across groups of query heads (e.g., 4 KV heads for 16 query heads)
- **MQA (Multi-Query Attention)**: All query heads share a single K and V head

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
MHA,1,256,1024,16,16,50,1.8218
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
└── Tests/
    └── AttnBenchTests/
        └── AttnBenchTests.swift # Unit tests
```

## Configuration

You can modify the benchmark parameters in `AttnBench.swift`:

```swift
let b = 1              // Batch size
let dModel = 1024      // Model dimension
let heads = 16         // Number of attention heads
let seqs = [128, 256, 512, 1024]  // Sequence lengths to test
let iters = 50         // Iterations per benchmark
let warmup = 10        // Warmup iterations
```

## How It Works

1. **Linear Projections**: Each attention block uses learned linear projections for Q, K, V, and output
2. **Scaled Dot-Product Attention (SDPA)**: Computes attention scores, applies softmax, and aggregates values
3. **Benchmarking**: Runs warmup iterations, then times the forward pass over multiple iterations

## Why Use GQA/MQA?

GQA and MQA reduce memory bandwidth requirements by sharing K/V projections across query heads:

- **MHA**: O(n² × h) memory for KV cache
- **GQA**: O(n² × g) memory, where g < h
- **MQA**: O(n²) memory (single KV head)

This makes GQA/MQA particularly beneficial for inference with long sequences or large batch sizes.

## License

MIT

## Acknowledgments

- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's machine learning framework for Apple Silicon