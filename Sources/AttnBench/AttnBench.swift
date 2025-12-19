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

        // Sliding window sizes to test (powers of 2)
        let windowSizes = [32, 64, 128]

        // Block sizes for block-sparse attention
        let blockSizes = [32, 64]

        var rows: [BenchRow] = []

        for n in seqs {
            // ===== Dense Attention Baselines =====
            let mha = MHA(b: b, n: n, dModel: dModel, heads: heads, seed: 1)
            let gqa = GQA(b: b, n: n, dModel: dModel, heads: heads, kvHeads: 4, seed: 2)
            let mqa = MQA(b: b, n: n, dModel: dModel, heads: heads, seed: 3)

            let denseBlocks: [(String, AttentionBlock, Int)] = [
                ("MHA", mha, heads),
                ("GQA(kv=4)", gqa, 4),
                ("MQA(kv=1)", mqa, 1)
            ]

            for (name, block, kvh) in denseBlocks {
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

            // ===== Sliding Window Attention =====
            // Only test window sizes that are <= sequence length
            for w in windowSizes where w <= n {
                let swa = SlidingWindowAttention(b: b, n: n, dModel: dModel, heads: heads, windowSize: w, seed: 4)
                let ms = bench(
                    block: swa,
                    b: b,
                    n: n,
                    dModel: dModel,
                    iters: iters,
                    warmup: warmup,
                    seed: UInt64(200 + n)
                )
                rows.append(BenchRow(
                    name: "SWA(w=\(w))",
                    b: b,
                    n: n,
                    dModel: dModel,
                    heads: heads,
                    kvHeads: heads,  // SWA uses full heads like MHA
                    iters: iters,
                    msPerIter: ms
                ))
            }

            // ===== Block-Sparse Attention =====
            // Only test block sizes that divide the sequence length
            for bs in blockSizes where n % bs == 0 {
                let blockSparse = BlockSparseAttention(
                    b: b, n: n, dModel: dModel, heads: heads,
                    blockSize: bs, globalTokens: 1, seed: 5
                )
                let ms = bench(
                    block: blockSparse,
                    b: b,
                    n: n,
                    dModel: dModel,
                    iters: iters,
                    warmup: warmup,
                    seed: UInt64(300 + n)
                )
                rows.append(BenchRow(
                    name: "BlockSparse(bs=\(bs))",
                    b: b,
                    n: n,
                    dModel: dModel,
                    heads: heads,
                    kvHeads: heads,
                    iters: iters,
                    msPerIter: ms
                ))
            }

            // ===== Linear Attention =====
            let linearAttn = LinearAttention(b: b, n: n, dModel: dModel, heads: heads, seed: 6)
            let linearMs = bench(
                block: linearAttn,
                b: b,
                n: n,
                dModel: dModel,
                iters: iters,
                warmup: warmup,
                seed: UInt64(400 + n)
            )
            rows.append(BenchRow(
                name: "LinearAttn",
                b: b,
                n: n,
                dModel: dModel,
                heads: heads,
                kvHeads: heads,
                iters: iters,
                msPerIter: linearMs
            ))

            // ===== Causal Linear Attention =====
            let causalLinear = CausalLinearAttention(b: b, n: n, dModel: dModel, heads: heads, seed: 7)
            let causalMs = bench(
                block: causalLinear,
                b: b,
                n: n,
                dModel: dModel,
                iters: iters,
                warmup: warmup,
                seed: UInt64(500 + n)
            )
            rows.append(BenchRow(
                name: "CausalLinearAttn",
                b: b,
                n: n,
                dModel: dModel,
                heads: heads,
                kvHeads: heads,
                iters: iters,
                msPerIter: causalMs
            ))
        }

        printCSV(rows)

        // Print summary
        print("\n// ===== BENCHMARK SUMMARY =====")
        print("// Dense Attention (O(N²)):")
        print("//   - MHA: Standard multi-head attention")
        print("//   - GQA: Grouped-query attention (fewer KV heads)")
        print("//   - MQA: Multi-query attention (single KV head)")
        print("//")
        print("// Sparse/Efficient Attention:")
        print("//   - SWA: Sliding Window Attention O(N·W) - true slicing, not masked")
        print("//   - BlockSparse: Block-local + global tokens attention")
        print("//   - LinearAttn: O(N) via kernel trick Q(K^TV)")
        print("//   - CausalLinearAttn: O(N) causal via cumsum")
        print("//")
        print("// Key findings to look for:")
        print("//   1. At what N does SWA become faster than MHA?")
        print("//   2. Do M4 AMX units prefer dense or batched small matmuls?")
        print("//   3. Linear attention overhead vs quadratic savings crossover")
    }
}
