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
public struct BenchRow: Sendable {
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
public struct Linear: Sendable {
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
public protocol AttentionBlock: Sendable {
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
