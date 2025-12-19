import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Utilities

/// Returns the current time in seconds with nanosecond precision.
@inline(__always)
public func nowSeconds() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000.0
}

/// Forces evaluation of lazy MLX arrays.
public func forceEval(_ arrays: [MLXArray]) {
    eval(arrays)
}

// MARK: - Benchmark Data

/// A row of benchmark results.
public struct BenchRow {
    public let name: String
    public let b: Int
    public let n: Int
    public let dModel: Int
    public let heads: Int
    public let kvHeads: Int
    public let iters: Int
    public let msPerIter: Double

    public init(name: String, b: Int, n: Int, dModel: Int, heads: Int, kvHeads: Int, iters: Int, msPerIter: Double) {
        self.name = name
        self.b = b
        self.n = n
        self.dModel = dModel
        self.heads = heads
        self.kvHeads = kvHeads
        self.iters = iters
        self.msPerIter = msPerIter
    }
}

/// Prints benchmark rows as CSV to stdout.
public func printCSV(_ rows: [BenchRow]) {
    print("name,b,n,dModel,heads,kvHeads,iters,msPerIter")
    for r in rows {
        print("\(r.name),\(r.b),\(r.n),\(r.dModel),\(r.heads),\(r.kvHeads),\(r.iters),\(String(format: "%.4f", r.msPerIter))")
    }
}

// MARK: - Linear Layer

/// A simple linear (fully-connected) layer.
public struct Linear {
    public let w: MLXArray
    public let b: MLXArray

    public init(_ inDim: Int, _ outDim: Int, seed: UInt64) {
        let key = MLXRandom.key(seed)
        // Xavier/He-like initialization: N(0, 1/sqrt(inDim))
        let scale = Float(1.0 / sqrt(Double(inDim)))
        self.w = MLXRandom.normal([inDim, outDim], key: key) * scale
        self.b = MLXArray.zeros([outDim], dtype: .float32)
        forceEval([w, b])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, N, In], w: [In, Out] => [B, N, Out]
        return matmul(x, w) + b
    }
}

// MARK: - Attention Protocol

/// Protocol for attention blocks.
public protocol AttentionBlock {
    var name: String { get }
    func forward(x: MLXArray) -> MLXArray
}

// MARK: - Scaled Dot-Product Attention

/// Scaled dot-product attention for already-projected Q, K, V.
///
/// Shapes:
/// - q: [..., N, Dh]
/// - k: [..., N, Dh]
/// - v: [..., N, Dh]
///
/// Returns: [..., N, Dh]
public func sdpa(q: MLXArray, k: MLXArray, v: MLXArray, dh: Int) -> MLXArray {
    // Transpose K for attention score computation
    let kt = k.transposed(axes: Array(0..<(k.ndim - 2)) + [k.ndim - 1, k.ndim - 2])
    let scale = Float(1.0 / sqrt(Double(dh)))
    let scores = matmul(q, kt) * scale

    // Softmax over last axis
    let probs = softmax(scores, axis: scores.ndim - 1)
    return matmul(probs, v)
}

// MARK: - Multi-Head Attention (MHA)

/// Standard Multi-Head Attention.
///
/// Each head has its own K and V projections.
public struct MHA: AttentionBlock {
    public let name: String
    public let b: Int
    public let n: Int
    public let dModel: Int
    public let heads: Int
    public let dh: Int
    public let wq: Linear
    public let wk: Linear
    public let wv: Linear
    public let wo: Linear

    public init(b: Int, n: Int, dModel: Int, heads: Int, seed: UInt64) {
        precondition(dModel % heads == 0, "dModel must be divisible by heads")
        self.name = "MHA"
        self.b = b
        self.n = n
        self.dModel = dModel
        self.heads = heads
        self.dh = dModel / heads
        self.wq = Linear(dModel, dModel, seed: seed &+ 1)
        self.wk = Linear(dModel, dModel, seed: seed &+ 2)
        self.wv = Linear(dModel, dModel, seed: seed &+ 3)
        self.wo = Linear(dModel, dModel, seed: seed &+ 4)
    }

    public func forward(x: MLXArray) -> MLXArray {
        // x: [B, N, D]
        var q = wq(x)
        var k = wk(x)
        var v = wv(x)

        // [B, N, H*Dh] -> [B, H, N, Dh]
        q = q.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        k = k.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        v = v.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])

        // SDPA on [B, H, N, Dh]
        let out = sdpa(q: q, k: k, v: v, dh: dh)

        // [B, H, N, Dh] -> [B, N, H*Dh] -> [B, N, D]
        let merged = out.transposed(axes: [0, 2, 1, 3]).reshaped([b, n, dModel])
        return wo(merged)
    }
}

// MARK: - Grouped Query Attention (GQA)

/// Grouped Query Attention.
///
/// K and V are shared across groups of query heads, reducing memory bandwidth.
public struct GQA: AttentionBlock {
    public let name: String
    public let b: Int
    public let n: Int
    public let dModel: Int
    public let heads: Int
    public let kvHeads: Int
    public let group: Int
    public let dh: Int
    public let wq: Linear
    public let wk: Linear
    public let wv: Linear
    public let wo: Linear

    public init(b: Int, n: Int, dModel: Int, heads: Int, kvHeads: Int, seed: UInt64) {
        precondition(dModel % heads == 0, "dModel must be divisible by heads")
        precondition(heads % kvHeads == 0, "heads must be divisible by kvHeads")
        self.name = "GQA"
        self.b = b
        self.n = n
        self.dModel = dModel
        self.heads = heads
        self.kvHeads = kvHeads
        self.group = heads / kvHeads
        self.dh = dModel / heads

        self.wq = Linear(dModel, heads * dh, seed: seed &+ 10)
        self.wk = Linear(dModel, kvHeads * dh, seed: seed &+ 11)
        self.wv = Linear(dModel, kvHeads * dh, seed: seed &+ 12)
        self.wo = Linear(dModel, dModel, seed: seed &+ 13)
    }

    public func forward(x: MLXArray) -> MLXArray {
        // Project
        var q = wq(x) // [B, N, H*Dh]
        var k = wk(x) // [B, N, KV*Dh]
        let v = wv(x) // [B, N, KV*Dh]

        // Reshape to heads
        // q: [B, H, N, Dh]
        q = q.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        // k, v: [B, KV, N, Dh]
        k = k.reshaped([b, n, kvHeads, dh]).transposed(axes: [0, 2, 1, 3])

        // Group Q heads: [B, KV, G, N, Dh]
        let qg = q.reshaped([b, kvHeads, group, n, dh])

        // Broadcast K, V across group dimension by inserting size-1 dim
        let kb = k.reshaped([b, kvHeads, 1, n, dh])
        let vb = v.reshaped([b, kvHeads, 1, n, dh])

        // SDPA over last dims: results [B, KV, G, N, Dh]
        let outg = sdpa(q: qg, k: kb, v: vb, dh: dh)

        // Merge back: [B, H, N, Dh] -> [B, N, D] -> out proj
        let out = outg.reshaped([b, heads, n, dh])
        let merged = out.transposed(axes: [0, 2, 1, 3]).reshaped([b, n, dModel])
        return wo(merged)
    }
}

// MARK: - Multi-Query Attention (MQA)

/// Multi-Query Attention.
///
/// All query heads share a single K and V head. This is GQA with kvHeads=1.
public struct MQA: AttentionBlock {
    let inner: GQA
    public var name: String { "MQA" }

    public init(b: Int, n: Int, dModel: Int, heads: Int, seed: UInt64) {
        self.inner = GQA(b: b, n: n, dModel: dModel, heads: heads, kvHeads: 1, seed: seed)
    }

    public func forward(x: MLXArray) -> MLXArray {
        inner.forward(x: x)
    }
}

// MARK: - Benchmark Harness

// MARK: - Sliding Window Attention (SWA)

/// True Sliding Window Attention with O(N·W) complexity.
///
/// Instead of computing the full N×N attention matrix and masking,
/// this implementation only computes attention scores within a local window
/// around each query position, reducing complexity from O(N²) to O(N·W).
///
/// This tests MLX's gather/slice performance vs dense compute on Apple Silicon.
public struct SlidingWindowAttention: AttentionBlock {
    public let name: String
    public let b: Int
    public let n: Int
    public let dModel: Int
    public let heads: Int
    public let windowSize: Int
    public let dh: Int
    public let wq: Linear
    public let wk: Linear
    public let wv: Linear
    public let wo: Linear

    public init(b: Int, n: Int, dModel: Int, heads: Int, windowSize: Int, seed: UInt64) {
        precondition(dModel % heads == 0, "dModel must be divisible by heads")
        precondition(windowSize > 0, "windowSize must be positive")
        self.name = "SWA(w=\(windowSize))"
        self.b = b
        self.n = n
        self.dModel = dModel
        self.heads = heads
        self.windowSize = min(windowSize, n)  // Cap window at sequence length
        self.dh = dModel / heads
        self.wq = Linear(dModel, dModel, seed: seed &+ 1)
        self.wk = Linear(dModel, dModel, seed: seed &+ 2)
        self.wv = Linear(dModel, dModel, seed: seed &+ 3)
        self.wo = Linear(dModel, dModel, seed: seed &+ 4)
    }

    public func forward(x: MLXArray) -> MLXArray {
        // x: [B, N, D]
        var q = wq(x)
        var k = wk(x)
        var v = wv(x)

        // Reshape to [B, H, N, Dh]
        q = q.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        k = k.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        v = v.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])

        // True sliding window: construct windowed K and V using padding + slicing
        // This avoids computing the full N×N matrix
        let out = slidingWindowSDPA(q: q, k: k, v: v, windowSize: windowSize, dh: dh)

        // [B, H, N, Dh] -> [B, N, H*Dh] -> [B, N, D]
        let merged = out.transposed(axes: [0, 2, 1, 3]).reshaped([b, n, dModel])
        return wo(merged)
    }
}

/// Sliding window scaled dot-product attention.
///
/// For each query position i, only computes attention to positions [i-W/2, i+W/2].
/// Uses padding and slicing to construct overlapping windows efficiently.
///
/// - Parameters:
///   - q: Query tensor [B, H, N, Dh]
///   - k: Key tensor [B, H, N, Dh]
///   - v: Value tensor [B, H, N, Dh]
///   - windowSize: Size of the attention window
///   - dh: Head dimension
/// - Returns: Output tensor [B, H, N, Dh]
public func slidingWindowSDPA(q: MLXArray, k: MLXArray, v: MLXArray, windowSize: Int, dh: Int) -> MLXArray {
    let n = q.dim(2)
    let halfWindow = windowSize / 2
    let scale = Float(1.0 / sqrt(Double(dh)))

    // Pad K and V on the sequence dimension for boundary handling
    // Pad shape: [B, H, N + windowSize - 1, Dh]
    let padWidth = [[0, 0], [0, 0], [halfWindow, windowSize - halfWindow - 1], [0, 0]]
    let kPadded = padded(k, widths: padWidth, mode: .constant, value: MLXArray(Float(-1e9)))
    let vPadded = padded(v, widths: padWidth, mode: .constant, value: MLXArray(Float(0)))

    // Build windowed views by gathering slices for each position
    // For position i, we want k[i:i+windowSize] and v[i:i+windowSize]
    // Collect windows: result shape [B, H, N, W, Dh]
    var windowedK: [MLXArray] = []
    var windowedV: [MLXArray] = []

    for i in 0..<n {
        // Slice [B, H, windowSize, Dh]
        let kSlice = kPadded[0..., 0..., i..<(i + windowSize), 0...]
        let vSlice = vPadded[0..., 0..., i..<(i + windowSize), 0...]
        windowedK.append(kSlice)
        windowedV.append(vSlice)
    }

    // Stack to get [N, B, H, W, Dh] then transpose to [B, H, N, W, Dh]
    let kWindows = stacked(windowedK, axis: 0).transposed(axes: [1, 2, 0, 3, 4])
    let vWindows = stacked(windowedV, axis: 0).transposed(axes: [1, 2, 0, 3, 4])

    // Compute attention scores: q @ kWindows^T
    // q: [B, H, N, 1, Dh], kWindows: [B, H, N, W, Dh]
    // scores: [B, H, N, 1, W]
    let qExpanded = q.reshaped([q.dim(0), q.dim(1), n, 1, dh])
    let kWindowsT = kWindows.transposed(axes: [0, 1, 2, 4, 3])  // [B, H, N, Dh, W]
    let scores = matmul(qExpanded, kWindowsT) * scale  // [B, H, N, 1, W]

    // Softmax over window dimension
    let probs = softmax(scores, axis: -1)

    // Weighted sum: probs @ vWindows
    // probs: [B, H, N, 1, W], vWindows: [B, H, N, W, Dh]
    // result: [B, H, N, 1, Dh]
    let attended = matmul(probs, vWindows)

    // Remove the extra dimension: [B, H, N, Dh]
    return attended.squeezed(axis: 3)
}

// MARK: - Block-Sparse Attention

/// Block-Sparse Attention with configurable block patterns.
///
/// Divides the N×N attention matrix into blocks and only computes specific blocks:
/// - Local blocks: diagonal blocks for local context
/// - Global tokens: first few tokens attend to/from all positions
///
/// This tests how well MLX handles batched small matmuls vs one giant matmul.
/// Finding the crossover point where block-sparse becomes faster than dense
/// on M-series chips is valuable benchmarking data.
public struct BlockSparseAttention: AttentionBlock {
    public let name: String
    public let b: Int
    public let n: Int
    public let dModel: Int
    public let heads: Int
    public let blockSize: Int
    public let globalTokens: Int
    public let dh: Int
    public let wq: Linear
    public let wk: Linear
    public let wv: Linear
    public let wo: Linear

    public init(b: Int, n: Int, dModel: Int, heads: Int, blockSize: Int, globalTokens: Int = 1, seed: UInt64) {
        precondition(dModel % heads == 0, "dModel must be divisible by heads")
        precondition(n % blockSize == 0, "n must be divisible by blockSize")
        self.name = "BlockSparse(bs=\(blockSize),g=\(globalTokens))"
        self.b = b
        self.n = n
        self.dModel = dModel
        self.heads = heads
        self.blockSize = blockSize
        self.globalTokens = min(globalTokens, n)
        self.dh = dModel / heads
        self.wq = Linear(dModel, dModel, seed: seed &+ 1)
        self.wk = Linear(dModel, dModel, seed: seed &+ 2)
        self.wv = Linear(dModel, dModel, seed: seed &+ 3)
        self.wo = Linear(dModel, dModel, seed: seed &+ 4)
    }

    public func forward(x: MLXArray) -> MLXArray {
        // x: [B, N, D]
        var q = wq(x)
        var k = wk(x)
        var v = wv(x)

        // Reshape to [B, H, N, Dh]
        q = q.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        k = k.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        v = v.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])

        let out = blockSparseSDPA(q: q, k: k, v: v, blockSize: blockSize, globalTokens: globalTokens, dh: dh)

        // [B, H, N, Dh] -> [B, N, H*Dh] -> [B, N, D]
        let merged = out.transposed(axes: [0, 2, 1, 3]).reshaped([b, n, dModel])
        return wo(merged)
    }
}

/// Block-sparse scaled dot-product attention.
///
/// Computes attention in a block-sparse pattern:
/// 1. Local block attention: each block attends only to itself
/// 2. Global tokens: first `g` tokens attend to all positions
///
/// - Parameters:
///   - q: Query tensor [B, H, N, Dh]
///   - k: Key tensor [B, H, N, Dh]
///   - v: Value tensor [B, H, N, Dh]
///   - blockSize: Size of each local block
///   - globalTokens: Number of global tokens
///   - dh: Head dimension
/// - Returns: Output tensor [B, H, N, Dh]
public func blockSparseSDPA(q: MLXArray, k: MLXArray, v: MLXArray, blockSize: Int, globalTokens: Int, dh: Int) -> MLXArray {
    let batchSize = q.dim(0)
    let numHeads = q.dim(1)
    let n = q.dim(2)
    let numBlocks = n / blockSize
    let scale = Float(1.0 / sqrt(Double(dh)))

    // Part 1: Local block attention
    // Reshape to blocks: [B, H, numBlocks, blockSize, Dh]
    let qBlocks = q.reshaped([batchSize, numHeads, numBlocks, blockSize, dh])
    let kBlocks = k.reshaped([batchSize, numHeads, numBlocks, blockSize, dh])
    let vBlocks = v.reshaped([batchSize, numHeads, numBlocks, blockSize, dh])

    // Compute block-local attention: [B, H, numBlocks, blockSize, blockSize]
    let kBlocksT = kBlocks.transposed(axes: [0, 1, 2, 4, 3])
    let localScores = matmul(qBlocks, kBlocksT) * scale
    let localProbs = softmax(localScores, axis: -1)
    var localOut = matmul(localProbs, vBlocks)  // [B, H, numBlocks, blockSize, Dh]

    // Reshape back: [B, H, N, Dh]
    localOut = localOut.reshaped([batchSize, numHeads, n, dh])

    // Part 2: Global token attention (if enabled)
    if globalTokens > 0 {
        // Global queries: first `globalTokens` positions attend to ALL keys
        let qGlobal = q[0..., 0..., 0..<globalTokens, 0...]  // [B, H, g, Dh]

        // Full attention for global tokens
        let kT = k.transposed(axes: [0, 1, 3, 2])  // [B, H, Dh, N]
        let globalScores = matmul(qGlobal, kT) * scale  // [B, H, g, N]
        let globalProbs = softmax(globalScores, axis: -1)
        let globalOut = matmul(globalProbs, v)  // [B, H, g, Dh]

        // Also: ALL positions attend to global keys (sink tokens)
        let kGlobal = k[0..., 0..., 0..<globalTokens, 0...]  // [B, H, g, Dh]
        let vGlobal = v[0..., 0..., 0..<globalTokens, 0...]  // [B, H, g, Dh]
        let kGlobalT = kGlobal.transposed(axes: [0, 1, 3, 2])  // [B, H, Dh, g]
        let sinkScores = matmul(q, kGlobalT) * scale  // [B, H, N, g]
        let sinkProbs = softmax(sinkScores, axis: -1)
        let sinkOut = matmul(sinkProbs, vGlobal)  // [B, H, N, Dh]

        // Combine: average local and sink attention (simplified combination)
        // In practice, you'd use a learned gating mechanism
        var combined = (localOut + sinkOut) * 0.5

        // Replace global token outputs with their full attention results
        // Build indices for scatter-like update
        let nonGlobalOut = combined[0..., 0..., globalTokens..., 0...]
        combined = concatenated([globalOut, nonGlobalOut], axis: 2)

        return combined
    }

    return localOut
}

// MARK: - Linear Attention

/// Linear Attention with O(N) complexity.
///
/// Rewrites attention as Q(K^T V) instead of (QK^T)V, changing complexity from O(N²) to O(N).
/// Uses feature maps (ELU+1) to approximate the softmax kernel.
///
/// This is the basis of efficient architectures like RWKV and Linear Transformers.
/// Benchmarking the accuracy/speed trade-off vs standard attention is valuable research.
public struct LinearAttention: AttentionBlock {
    public let name: String
    public let b: Int
    public let n: Int
    public let dModel: Int
    public let heads: Int
    public let dh: Int
    public let wq: Linear
    public let wk: Linear
    public let wv: Linear
    public let wo: Linear
    public let eps: Float

    public init(b: Int, n: Int, dModel: Int, heads: Int, seed: UInt64, eps: Float = 1e-6) {
        precondition(dModel % heads == 0, "dModel must be divisible by heads")
        self.name = "LinearAttn"
        self.b = b
        self.n = n
        self.dModel = dModel
        self.heads = heads
        self.dh = dModel / heads
        self.eps = eps
        self.wq = Linear(dModel, dModel, seed: seed &+ 1)
        self.wk = Linear(dModel, dModel, seed: seed &+ 2)
        self.wv = Linear(dModel, dModel, seed: seed &+ 3)
        self.wo = Linear(dModel, dModel, seed: seed &+ 4)
    }

    public func forward(x: MLXArray) -> MLXArray {
        // x: [B, N, D]
        var q = wq(x)
        var k = wk(x)
        var v = wv(x)

        // Reshape to [B, H, N, Dh]
        q = q.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        k = k.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        v = v.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])

        let out = linearSDPA(q: q, k: k, v: v, dh: dh, eps: eps)

        // [B, H, N, Dh] -> [B, N, H*Dh] -> [B, N, D]
        let merged = out.transposed(axes: [0, 2, 1, 3]).reshaped([b, n, dModel])
        return wo(merged)
    }
}

/// Linear scaled dot-product attention using kernel feature maps.
///
/// Instead of computing the full N×N attention matrix:
///   Attn(Q, K, V) = softmax(QK^T / sqrt(d)) @ V    -- O(N²)
///
/// We use a feature map φ to approximate:
///   Attn(Q, K, V) ≈ φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)    -- O(N)
///
/// The key insight: computing (φ(K)^T @ V) first gives [Dh, Dh] instead of [N, N].
///
/// We use ELU+1 as the feature map: φ(x) = ELU(x) + 1
/// This ensures non-negativity (required for valid attention weights).
///
/// - Parameters:
///   - q: Query tensor [B, H, N, Dh]
///   - k: Key tensor [B, H, N, Dh]
///   - v: Value tensor [B, H, N, Dh]
///   - dh: Head dimension
///   - eps: Small epsilon for numerical stability
/// - Returns: Output tensor [B, H, N, Dh]
public func linearSDPA(q: MLXArray, k: MLXArray, v: MLXArray, dh: Int, eps: Float) -> MLXArray {
    // Feature map: ELU(x) + 1 ensures non-negative values
    // ELU(x) = x if x > 0, else exp(x) - 1
    // ELU(x) + 1 = x + 1 if x > 0, else exp(x)
    func featureMap(_ x: MLXArray) -> MLXArray {
        // ELU + 1: ensures positivity for valid attention distribution
        let positive = maximum(x, MLXArray(Float(0)))
        let negative = minimum(x, MLXArray(Float(0)))
        return positive + MLXArray(Float(1)) + exp(negative) - MLXArray(Float(1))
    }

    // Apply feature maps
    let phiQ = featureMap(q)  // [B, H, N, Dh]
    let phiK = featureMap(k)  // [B, H, N, Dh]

    // Compute K^T @ V first: [B, H, Dh, N] @ [B, H, N, Dh] = [B, H, Dh, Dh]
    let phiKT = phiK.transposed(axes: [0, 1, 3, 2])  // [B, H, Dh, N]
    let kv = matmul(phiKT, v)  // [B, H, Dh, Dh]

    // Compute Q @ (K^T @ V): [B, H, N, Dh] @ [B, H, Dh, Dh] = [B, H, N, Dh]
    let numerator = matmul(phiQ, kv)

    // Normalization: sum of attention weights per query
    // K^T @ 1 (sum over sequence): [B, H, Dh, N] @ [B, H, N, 1] = [B, H, Dh, 1]
    let ones = MLXArray.ones([phiK.dim(0), phiK.dim(1), phiK.dim(2), 1], dtype: phiK.dtype)
    let kSum = matmul(phiKT, ones)  // [B, H, Dh, 1]

    // Q @ (K^T @ 1): [B, H, N, Dh] @ [B, H, Dh, 1] = [B, H, N, 1]
    let denominator = matmul(phiQ, kSum) + eps  // Add eps for stability

    return numerator / denominator
}

// MARK: - Causal Linear Attention

/// Causal Linear Attention using cumulative sums for autoregressive generation.
///
/// Implements causal (unidirectional) linear attention using cumsum operations.
/// This is critical for language modeling where each position can only attend to past positions.
///
/// Complexity: O(N) time and O(D²) memory per head.
public struct CausalLinearAttention: AttentionBlock {
    public let name: String
    public let b: Int
    public let n: Int
    public let dModel: Int
    public let heads: Int
    public let dh: Int
    public let wq: Linear
    public let wk: Linear
    public let wv: Linear
    public let wo: Linear
    public let eps: Float

    public init(b: Int, n: Int, dModel: Int, heads: Int, seed: UInt64, eps: Float = 1e-6) {
        precondition(dModel % heads == 0, "dModel must be divisible by heads")
        self.name = "CausalLinearAttn"
        self.b = b
        self.n = n
        self.dModel = dModel
        self.heads = heads
        self.dh = dModel / heads
        self.eps = eps
        self.wq = Linear(dModel, dModel, seed: seed &+ 1)
        self.wk = Linear(dModel, dModel, seed: seed &+ 2)
        self.wv = Linear(dModel, dModel, seed: seed &+ 3)
        self.wo = Linear(dModel, dModel, seed: seed &+ 4)
    }

    public func forward(x: MLXArray) -> MLXArray {
        // x: [B, N, D]
        var q = wq(x)
        var k = wk(x)
        var v = wv(x)

        // Reshape to [B, H, N, Dh]
        q = q.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        k = k.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        v = v.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])

        let out = causalLinearSDPA(q: q, k: k, v: v, dh: dh, eps: eps)

        // [B, H, N, Dh] -> [B, N, H*Dh] -> [B, N, D]
        let merged = out.transposed(axes: [0, 2, 1, 3]).reshaped([b, n, dModel])
        return wo(merged)
    }
}

/// Causal linear attention using cumulative sums.
///
/// For causal attention, we need: output[i] = sum(φ(k[j]) ⊗ v[j] for j <= i) @ φ(q[i])
/// This can be computed efficiently using cumsum over the KV outer products.
///
/// - Parameters:
///   - q: Query tensor [B, H, N, Dh]
///   - k: Key tensor [B, H, N, Dh]
///   - v: Value tensor [B, H, N, Dh]
///   - dh: Head dimension
///   - eps: Small epsilon for numerical stability
/// - Returns: Output tensor [B, H, N, Dh]
public func causalLinearSDPA(q: MLXArray, k: MLXArray, v: MLXArray, dh: Int, eps: Float) -> MLXArray {
    // Feature map: ELU(x) + 1
    func featureMap(_ x: MLXArray) -> MLXArray {
        let positive = maximum(x, MLXArray(Float(0)))
        let negative = minimum(x, MLXArray(Float(0)))
        return positive + MLXArray(Float(1)) + exp(negative) - MLXArray(Float(1))
    }

    let phiQ = featureMap(q)  // [B, H, N, Dh]
    let phiK = featureMap(k)  // [B, H, N, Dh]

    // Compute outer product of K and V at each position
    // kv[i] = k[i] ⊗ v[i] : [B, H, N, Dh, Dh]
    let phiKExpanded = phiK[.ellipsis, .newAxis]  // [B, H, N, Dh, 1]
    let vExpanded = v[0..., 0..., 0..., .newAxis, 0...]  // [B, H, N, 1, Dh]
    let kv = phiKExpanded * vExpanded  // [B, H, N, Dh, Dh] via broadcast

    // Cumulative sum along sequence dimension for causal masking
    // cumKV[i] = sum(kv[j] for j <= i)
    let cumKV = kv.cumsum(axis: 2)  // [B, H, N, Dh, Dh]

    // Compute output: phiQ @ cumKV
    // phiQ: [B, H, N, 1, Dh], cumKV: [B, H, N, Dh, Dh]
    let phiQExpanded = phiQ[0..., 0..., 0..., .newAxis, 0...]  // [B, H, N, 1, Dh]
    let numerator = matmul(phiQExpanded, cumKV).squeezed(axis: 3)  // [B, H, N, Dh]

    // Normalization: cumulative sum of K features
    let cumK = phiK.cumsum(axis: 2)  // [B, H, N, Dh]
    let denominator = (phiQ * cumK).sum(axis: -1, keepDims: true) + eps  // [B, H, N, 1]

    return numerator / denominator
}

// MARK: - Benchmark Harness

/// Benchmarks an attention block and returns the average time in milliseconds per iteration.
///
/// - Parameters:
///   - block: The attention block to benchmark
///   - b: Batch size
///   - n: Sequence length
///   - dModel: Model dimension
///   - iters: Number of timed iterations
///   - warmup: Number of warmup iterations
///   - seed: Random seed for input generation
/// - Returns: Average milliseconds per iteration
public func bench(block: AttentionBlock, b: Int, n: Int, dModel: Int, iters: Int, warmup: Int, seed: UInt64) -> Double {
    let key = MLXRandom.key(seed)
    let x = MLXRandom.normal([b, n, dModel], key: key)
    forceEval([x])

    // Warmup
    for _ in 0..<warmup {
        let y = block.forward(x: x)
        forceEval([y])
    }

    // Timed iterations
    let t0 = nowSeconds()
    for _ in 0..<iters {
        let y = block.forward(x: x)
        forceEval([y])
    }
    let t1 = nowSeconds()

    return (t1 - t0) * 1000.0 / Double(iters)
}
