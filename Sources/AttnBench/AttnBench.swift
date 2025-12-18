import Foundation
import MLX
import MLXNN
import MLXRandom

// ---- Utilities ----

@inline(__always)
func nowSeconds() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000.0
}

func forceEval(_, arrays: [MLXArray]) {
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

func printCSV(_, rows: [BenchRow]) {
    print("name,b")
}


@main
struct AttnBench {
    static func main() {
        print("Hello, world!")
    }
}
