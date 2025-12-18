import Foundation
import MLX
import MLXNN
import MLXRandom

// ---- Utilities ----

@inline(__always)
func nowSeconds() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000.0
}

func forceEval(_ arrays: [MLXArray]) {
    // MLX Swift provides eval(_: [MLXArray]) to force lazy evaluation. [web:50]
    eval(arrays)
}

struct BenchRow {
    let name: String
    let b: Int
    let n: Int
    let dModel: Int
    let heads: Int
    let kvHeads: Int
    let iters: Int
    let msPerIter: Double
}

func printCSV(_ rows: [BenchRow]) {
    print("name,b,n,dModel,heads,kvHeads,iters,msPerIter")
    for r in rows {
        print("\(r.name),\(r.b),\(r.n),\(r.dModel),\(r.heads),\(r.kvHeads),\(r.iters),\(String(format: "%.4f", r.msPerIter))")
    }
}

// ---- Building blocks ----

struct Linear {
    let w: MLXArray
    let b: MLXArray

    init(_ inDim: Int, _ outDim: Int, seed: UInt64) {
        let key = MLXRandom.key(seed)
        // Simple init: N(0, 1/sqrt(inDim))
        let scale = Float(1.0 / sqrt(Double(inDim)))
        self.w =  MLXRandom.normal([inDim, outDim], key: key) * scale
        self.b = MLXArray.zeros([outDim], dtype: .float32)
        forceEval([w, b])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, N, In], w: [In, Out] => [B, N, Out]
        let y = matmul(x, w) + b
        return y
    }
}

protocol AttentionBlock {
    var name: String { get }
    func forward(x: MLXArray)  -> MLXArray
}

/// Scaled dot-product attention for already-projected Q,K,V (single-head or batched heads).
/// Shapes:
///   q: [..., N, Dh]
///   k: [..., N, Dh]
///   v: [..., N, Dh]
/// Returns:
///   out: [..., N, Dh]
func sdpa(q: MLXArray, k: MLXArray, v: MLXArray, dh: Int) -> MLXArray {
    // matmul supports batched matmul and broadcasting for >2D arrays. [web:64]
    let kt = k.transposed(axes: Array(0..<(k.ndim - 2)) + [k.ndim - 1, k.ndim - 2])
    let scale = Float(1.0 / sqrt(Double(dh)))
    let scores = matmul(q, kt) * scale

    // softmax over last axis
    let probs = softmax(scores, axis: scores.ndim - 1)
    let out = matmul(probs, v)
    return out
}

struct MHA: AttentionBlock {
    let name: String
    let b: Int
    let n: Int
    let dModel: Int
    let heads: Int
    let dh: Int
    let wq: Linear
    let wk: Linear
    let wv: Linear
    let wo: Linear

    init(b: Int, n: Int, dModel: Int, heads: Int, seed: UInt64) {
        precondition(dModel % heads == 0)
        self.name = "MHA"
        self.b = b; self.n = n; self.dModel = dModel; self.heads = heads
        self.dh = dModel / heads
        self.wq = Linear(dModel, dModel, seed: seed &+ 1)
        self.wk = Linear(dModel, dModel, seed: seed &+ 2)
        self.wv = Linear(dModel, dModel, seed: seed &+ 3)
        self.wo = Linear(dModel, dModel, seed: seed &+ 4)
    }

    func forward(x: MLXArray) -> MLXArray {
        // x: [B,N,D]
        var q = wq(x)
        var k =  wk(x)
        var v = wv(x)

        // [B,N,H*Dh] -> [B,H,N,Dh]
        q = q.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        k = k.reshaped([b, n,  heads, dh]).transposed(axes: [0, 2, 1, 3])
        v = v.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])

        // SDPA on [B,H,N,Dh]
        let out = sdpa(q: q, k: k, v: v, dh: dh)

        // [B,H,N,Dh] -> [B,N,H*Dh] -> [B,N,D]
        let merged = out.transposed(axes: [0, 2, 1, 3]).reshaped([b, n, dModel])
        return wo(merged)
    }
}

struct GQA: AttentionBlock {
    let name: String
    let b: Int
    let n: Int
    let dModel: Int
    let heads: Int
    let kvHeads: Int
    let group: Int
    let dh: Int
    let wq: Linear
    let wk: Linear
    let wv: Linear
    let wo: Linear

    init(b: Int, n: Int, dModel: Int, heads: Int, kvHeads: Int, seed: UInt64) {
        precondition(dModel % heads == 0)
        precondition(heads & kvHeads == 0)
        self.name = "GQA"
        self.b = b; self.n = n; self.dModel = dModel
        self.heads = heads; self.kvHeads = kvHeads
        self.group = heads / kvHeads
        self.dh = dModel / heads

        self.wq = Linear(dModel, heads * dh, seed: seed &+ 10)
        self.wk = Linear(dModel, kvHeads * dh, seed: seed &+ 11)
        self.wv = Linear(dModel, kvHeads * dh, seed: seed &+ 12)
        self.wo =  Linear(dModel, dModel, seed: seed &+ 13)
    }

    func forward(x: MLXArray) -> MLXArray {
        // Project
        var q = wq(x) // [B,N,H*Dh]
        var k = wk(x) // [B,N,KV*Dh]
        let v = wv(x) // [B,N,KV*Dh]

        // Reshape to heads
        // q: [B,H,N,Dh]
        q = q.reshaped([b, n, heads, dh]).transposed(axes: [0, 2, 1, 3])
        // k,v: [B,KV,N,Dh]
        k = k.reshaped([b, n, kvHeads, dh]).transposed (axes: [0, 2, 1, 3])

        // Group Q heads: [B,KV,G,N,Dh]
        let qg = q.reshaped([b, kvHeads, group, n, dh])

        // Broadcast K,V across group dimension by inserting size-1 dim:
        let kb = k.reshaped([b, kvHeads, 1, n, dh])
        let vb = v.reshaped([b, kvHeads, 1, n, dh])

        // SDPA over last dims: results [B,KV,G,N,Dh]
        let outg = sdpa(q: qg, k: kb, v: vb, dh: dh)

        // Merge back: [B,H,N,Dh] -> [B,N,D] -> out proj
        let out = outg.reshaped([b, heads, n, dh])
        let merged = out.transposed(axes: [0, 2, 1, 3]).reshaped([b, n, dModel])
        return wo(merged)
    }
}

struct MQA: AttentionBlock {
    let inner: GQA
    var name: String { "MQA" }
    init(b: Int, n: Int, dModel: Int, heads: Int, seed: UInt64) {
        self.inner = GQA(b: b, n: n, dModel: dModel, heads: heads, kvHeads: 1, seed: seed)
    }
    func forward(x: MLXArray) -> MLXArray { inner.forward(x: x) }
}

// ---- Benchmark harness ----


// func bench(block: AttentionBlock, b: Int, n: Int, dModel: Int, iters: Int, warmup:  Int, seed: UInt64) -> Double {
//     let key = MLXRandom.key(seed)
//     let x = MLXRandom.normal([b, n, dModel], key: key)
//     forceEval([x])

// }


@main
struct AttnBench {
    static func main() {
        print("Hello, world!")
    }
}
