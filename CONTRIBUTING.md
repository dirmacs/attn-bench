# Contributing to AttnBench

Thank you for your interest in contributing to AttnBench! This document provides guidelines and instructions for contributing.

## About Dirmacs Labs

AttnBench is a research project by **Dirmacs Labs**, the R&D division of DIRMACS focused on exploring cutting-edge technologies from hardware-aware algorithm design to high-level ML systems.

## Ways to Contribute

### 1. Report Issues

- **Bug reports**: If you encounter bugs, please open an issue with:
  - Hardware details (M1/M2/M3/M4, memory size)
  - macOS version
  - Steps to reproduce
  - Expected vs actual behavior
  - Error messages or logs

- **Feature requests**: Describe the feature and its use case

### 2. Add Benchmarks

We welcome contributions that extend our benchmark coverage:

- **New attention mechanisms**: Implement additional variants (e.g., Flash Attention, Reformer)
- **New hardware**: Run benchmarks on different Apple Silicon variants (M1 Pro/Max/Ultra, M2, M3)
- **New configurations**: Test different model dimensions, head counts, batch sizes

### 3. Improve Analysis

- Enhance the Python analysis pipeline
- Add new visualizations
- Improve statistical methodology

### 4. Documentation

- Fix typos or clarify explanations
- Add examples or tutorials
- Translate documentation

## Development Setup

### Prerequisites

```bash
# Install build tools
brew install cmake ninja

# Install Xcode command line tools
xcode-select --install

# Set up Xcode for Metal compilation
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

# Install Python dependencies
pip install -r analysis/requirements.txt

# Install Typst (for paper compilation)
brew install typst
```

### Building

```bash
cmake -B build -G Ninja
cmake --build build
```

### Running Tests

```bash
xcodebuild test -scheme AttnBench-Package -destination 'platform=OS X'
```

## Code Style

### Swift

- Follow Swift API Design Guidelines
- Use meaningful variable names
- Add documentation comments for public APIs
- Keep functions focused and concise

```swift
/// Computes block-sparse attention with local and global patterns.
/// - Parameters:
///   - q: Query tensor [B, H, N, Dh]
///   - k: Key tensor [B, H, N, Dh]
///   - v: Value tensor [B, H, N, Dh]
///   - blockSize: Size of local attention blocks
///   - numGlobalTokens: Number of tokens with global attention
/// - Returns: Attention output [B, H, N, Dh]
public func blockSparseSDPA(...) -> MLXArray {
    // Implementation
}
```

### Python

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to functions

```python
def compute_speedup(
    baseline: BenchmarkStats,
    comparison: BenchmarkStats
) -> tuple[float, float, float]:
    """
    Compute speedup ratio with confidence intervals.
    
    Args:
        baseline: Statistics for the baseline mechanism
        comparison: Statistics for the comparison mechanism
    
    Returns:
        Tuple of (speedup, ci_lower, ci_upper)
    """
```

## Submitting Changes

### Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** with clear, atomic commits
4. **Run tests** to ensure nothing is broken
5. **Update documentation** if needed
6. **Submit a pull request** with:
   - Clear description of changes
   - Link to related issues
   - Screenshots/results if applicable

### Commit Messages

Use clear, descriptive commit messages:

```
Add Reformer attention implementation

- Implement LSH attention mechanism
- Add unit tests for shape correctness
- Update README with new mechanism
```

### Adding New Attention Mechanisms

When adding a new attention mechanism:

1. **Implement** in `Sources/AttnBenchLib/Attention.swift`:
   ```swift
   public class YourAttention: AttentionBlock {
       public var name: String { "YourAttn" }
       
       public func forward(x: MLXArray) -> MLXArray {
           // Implementation
       }
   }
   ```

2. **Add tests** in `Tests/AttnBenchTests/`:
   ```swift
   func testYourAttentionShape() throws {
       let attn = YourAttention(dModel: 64, heads: 4, ...)
       let x = MLXRandom.normal([1, 16, 64])
       let y = attn.forward(x: x)
       XCTAssertEqual(y.shape, x.shape)
   }
   ```

3. **Add to benchmark** in `Sources/AttnBench/AttnBench.swift`

4. **Update analysis** if needed in `analysis/analyze_benchmarks.py`

5. **Document** in README.md

## Benchmark Contribution Guidelines

If you're contributing benchmark results from different hardware:

1. **Provide full hardware details**:
   - Chip model (e.g., M2 Pro, M3 Max)
   - Total memory
   - macOS version
   - Any thermal/performance settings

2. **Use the standard configuration**:
   - 5 independent runs
   - 20 iterations per run
   - 5 warmup iterations
   - Batch size 1
   - Model dimension 512
   - 8 attention heads

3. **Submit raw data** along with analysis

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the technical merits
- Help newcomers learn

## Questions?

- Open an issue for questions
- Tag maintainers for urgent matters

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AttnBench!

**Dirmacs Labs** — Exploring cutting-edge technologies from hardware to ML systems