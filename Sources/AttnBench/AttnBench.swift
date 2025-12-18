import AttnBenchLib

@main
struct AttnBenchMain {
    static func main() {
        // Configuration
        let b = 1              // Batch size
        let dModel = 1024      // Model dimension
        let heads = 16         // Number of attention heads
        let seqs = [128, 256, 512, 1024]  // Sequence lengths to test
        let iters = 50         // Iterations per benchmark
        let warmup = 10        // Warmup iterations

        var rows: [BenchRow] = []

        for n in seqs {
            let mha = MHA(b: b, n: n, dModel: dModel, heads: heads, seed: 1)
            let gqa = GQA(b: b, n: n, dModel: dModel, heads: heads, kvHeads: 4, seed: 2)
            let mqa = MQA(b: b, n: n, dModel: dModel, heads: heads, seed: 3)

            let blocks: [(String, AttentionBlock, Int)] = [
                ("MHA", mha, heads),
                ("GQA(kv=4)", gqa, 4),
                ("MQA(kv=1)", mqa, 1)
            ]

            for (name, block, kvh) in blocks {
                let ms = bench(
                    block: block,
                    b: b,
                    n: n,
                    dModel: dModel,
                    iters: iters,
                    warmup: warmup,
                    seed: UInt64(100 + n)
                )
                rows.append(BenchRow(
                    name: name,
                    b: b,
                    n: n,
                    dModel: dModel,
                    heads: heads,
                    kvHeads: kvh,
                    iters: iters,
                    msPerIter: ms
                ))
            }
        }

        printCSV(rows)
    }
}
