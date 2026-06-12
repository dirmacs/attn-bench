import AttnBenchLib

@main
struct AttnBenchMain {
    static func main() {
        // ===========================================================================
        // CONFIGURATION - Statistically Rigorous Benchmarking
        // ===========================================================================

        // Basic parameters
        let b = 1              // Batch size
        let dModel = 512       // Model dimension
        let heads = 8          // Number of attention heads
        let seqs = [128, 256, 512, 1024]  // Sequence lengths to test

        // Statistical rigor parameters
        let numRuns = 5        // Number of independent runs per configuration
        let itersPerRun = 20   // Iterations per run
        let warmup = 5         // Warmup iterations per run

        // Window sizes for sliding window attention
        let windowSizes = [32, 64, 128]

        // Block sizes for block-sparse attention
        let blockSizes = [32, 64]

        // Memory safety threshold (~2GB for attention matrices)
        let maxAttentionMemoryBytes = 2 * 1024 * 1024 * 1024
        func isMemorySafe(b: Int, heads: Int, n: Int) -> Bool {
            let attentionMatrixBytes = b * heads * n * n * 4
            return attentionMatrixBytes < maxAttentionMemoryBytes
        }

        // ===========================================================================
        // BENCHMARK EXECUTION
        // ===========================================================================

        var allResults: [BenchResult] = []
        var aggregatedStats: [(AggregatedStats, Int, Int, Int, Int)] = []  // (stats, b, dModel, heads, kvHeads)

        print("// ===========================================================================")
        print("// ATTNBENCH - Statistically Rigorous Attention Mechanism Benchmarking")
        print("// ===========================================================================")
        print("// Configuration:")
        print("//   Batch Size: \(b)")
        print("//   Model Dimension: \(dModel)")
        print("//   Heads: \(heads)")
        print("//   Sequence Lengths: \(seqs)")
        print("//   Independent Runs: \(numRuns)")
        print("//   Iterations per Run: \(itersPerRun)")
        print("//   Warmup: \(warmup)")
        print("// ===========================================================================")
        print("")

        for n in seqs {
            guard isMemorySafe(b: b, heads: heads, n: n) else {
                print("// Skipping n=\(n): would exceed memory limit")
                continue
            }

            print("// Benchmarking sequence length N=\(n)...")

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
                print("//   Running \(name)...")
                let results = benchMultipleRuns(
                    block: block,
                    name: name,
                    b: b,
                    n: n,
                    dModel: dModel,
                    heads: heads,
                    kvHeads: kvh,
                    runs: numRuns,
                    itersPerRun: itersPerRun,
                    warmup: warmup,
                    baseSeed: UInt64(100 + n)
                )
                allResults.append(contentsOf: results)

                let stats = AggregatedStats(results: results)
                aggregatedStats.append((stats, b, dModel, heads, kvh))
            }

            // ===== Sliding Window Attention =====
            for w in windowSizes where w <= n && w <= 128 {
                // True SWA (gather-based) - only for small N due to memory
                if n <= 256 {
                    print("//   Running SWA(w=\(w))...")
                    let swa = SlidingWindowAttention(b: b, n: n, dModel: dModel, heads: heads, windowSize: w, seed: 4)
                    let results = benchMultipleRuns(
                        block: swa,
                        name: "SWA(w=\(w))",
                        b: b,
                        n: n,
                        dModel: dModel,
                        heads: heads,
                        kvHeads: heads,
                        runs: numRuns,
                        itersPerRun: itersPerRun,
                        warmup: warmup,
                        baseSeed: UInt64(200 + n + w)
                    )
                    allResults.append(contentsOf: results)

                    let stats = AggregatedStats(results: results)
                    aggregatedStats.append((stats, b, dModel, heads, heads))
                }

                // Masked SWA (dense+mask) - always safe
                print("//   Running MaskedSWA(w=\(w))...")
                let maskedSwa = MaskedSlidingWindowAttention(b: b, n: n, dModel: dModel, heads: heads, windowSize: w, seed: 4)
                let maskedResults = benchMultipleRuns(
                    block: maskedSwa,
                    name: "MaskedSWA(w=\(w))",
                    b: b,
                    n: n,
                    dModel: dModel,
                    heads: heads,
                    kvHeads: heads,
                    runs: numRuns,
                    itersPerRun: itersPerRun,
                    warmup: warmup,
                    baseSeed: UInt64(250 + n + w)
                )
                allResults.append(contentsOf: maskedResults)

                let maskedStats = AggregatedStats(results: maskedResults)
                aggregatedStats.append((maskedStats, b, dModel, heads, heads))
            }

            // ===== Block-Sparse Attention =====
            for bs in blockSizes where n % bs == 0 {
                print("//   Running BlockSparse(bs=\(bs))...")
                let blockSparse = BlockSparseAttention(
                    b: b, n: n, dModel: dModel, heads: heads,
                    blockSize: bs, globalTokens: 1, seed: 5
                )
                let results = benchMultipleRuns(
                    block: blockSparse,
                    name: "BlockSparse(bs=\(bs))",
                    b: b,
                    n: n,
                    dModel: dModel,
                    heads: heads,
                    kvHeads: heads,
                    runs: numRuns,
                    itersPerRun: itersPerRun,
                    warmup: warmup,
                    baseSeed: UInt64(300 + n + bs)
                )
                allResults.append(contentsOf: results)

                let stats = AggregatedStats(results: results)
                aggregatedStats.append((stats, b, dModel, heads, heads))
            }

            // ===== Linear Attention =====
            print("//   Running LinearAttn...")
            let linearAttn = LinearAttention(b: b, n: n, dModel: dModel, heads: heads, seed: 6)
            let linearResults = benchMultipleRuns(
                block: linearAttn,
                name: "LinearAttn",
                b: b,
                n: n,
                dModel: dModel,
                heads: heads,
                kvHeads: heads,
                runs: numRuns,
                itersPerRun: itersPerRun,
                warmup: warmup,
                baseSeed: UInt64(400 + n)
            )
            allResults.append(contentsOf: linearResults)

            let linearStats = AggregatedStats(results: linearResults)
            aggregatedStats.append((linearStats, b, dModel, heads, heads))

            // ===== Causal Linear Attention =====
            // Skip for large N due to cumsum overhead creating large intermediates
            if n <= 512 {
                print("//   Running CausalLinearAttn...")
                let causalLinear = CausalLinearAttention(b: b, n: n, dModel: dModel, heads: heads, seed: 7)
                let causalResults = benchMultipleRuns(
                    block: causalLinear,
                    name: "CausalLinearAttn",
                    b: b,
                    n: n,
                    dModel: dModel,
                    heads: heads,
                    kvHeads: heads,
                    runs: numRuns,
                    itersPerRun: itersPerRun,
                    warmup: warmup,
                    baseSeed: UInt64(500 + n)
                )
                allResults.append(contentsOf: causalResults)

                let causalStats = AggregatedStats(results: causalResults)
                aggregatedStats.append((causalStats, b, dModel, heads, heads))
            }
        }

        // ===========================================================================
        // OUTPUT - Extended CSV with per-iteration timing
        // ===========================================================================

        print("")
        print("// ===========================================================================")
        print("// EXTENDED RESULTS (per-iteration timing)")
        print("// ===========================================================================")
        printExtendedCSVHeader()
        printExtendedCSV(allResults)

        // ===========================================================================
        // OUTPUT - Aggregated Statistics CSV
        // ===========================================================================

        print("")
        print("// ===========================================================================")
        print("// AGGREGATED STATISTICS")
        print("// ===========================================================================")
        printAggregatedCSVHeader()
        for (stats, b, dModel, heads, kvHeads) in aggregatedStats {
            printAggregatedCSV(stats, b: b, dModel: dModel, heads: heads, kvHeads: kvHeads)
        }

        // ===========================================================================
        // SUMMARY - Key Findings
        // ===========================================================================

        print("")
        print("// ===========================================================================")
        print("// BENCHMARK SUMMARY")
        print("// ===========================================================================")
        print("// Configuration:")
        print("//   b=\(b), dModel=\(dModel), heads=\(heads)")
        print("//   Independent runs: \(numRuns)")
        print("//   Iterations per run: \(itersPerRun)")
        print("//   Total measurements per config: \(numRuns * itersPerRun)")
        print("//")
        print("// Attention Mechanisms Tested:")
        print("//   Dense: MHA, GQA(kv=4), MQA(kv=1)")
        print("//   Sliding Window: SWA (gather), MaskedSWA (dense+mask)")
        print("//   Block-Sparse: bs=32, bs=64")
        print("//   Linear: LinearAttn, CausalLinearAttn")
        print("//")
        print("// Statistical Notes:")
        print("//   - Each mechanism was run \(numRuns) independent times")
        print("//   - Each run includes \(warmup) warmup iterations (excluded from timing)")
        print("//   - 95% CIs computed using t-distribution")
        print("//   - Per-iteration data available for detailed analysis")
        print("//")
        print("// Key Findings (preliminary):")

        // Find best mechanism at each sequence length from aggregated stats
        var bestBySeq: [Int: (String, Double, Double)] = [:]  // seq -> (name, mean, ci95)
        for (stats, _, _, _, _) in aggregatedStats {
            if let current = bestBySeq[stats.n] {
                if stats.grandMean < current.1 {
                    bestBySeq[stats.n] = (stats.name, stats.grandMean, stats.ci95)
                }
            } else {
                bestBySeq[stats.n] = (stats.name, stats.grandMean, stats.ci95)
            }
        }

        for n in seqs.sorted() {
            if let (name, mean, ci95) = bestBySeq[n] {
                print("//   N=\(n): Best = \(name) (\(String(format: "%.2f", mean)) ± \(String(format: "%.2f", ci95)) ms)")
            }
        }

        print("//")
        print("// For full statistical analysis, run:")
        print("//   python analysis/analyze_benchmarks.py --input <csv_file> --output figures/")
        print("// ===========================================================================")
    }
}
