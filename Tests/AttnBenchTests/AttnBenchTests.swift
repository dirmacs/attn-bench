import Testing
import MLX
import MLXRandom

@testable import AttnBenchLib

@Suite("Linear Layer Tests")
struct LinearTests {
    @Test("Linear layer produces correct output shape")
    func testLinearShape() {
        let linear = Linear(64, 128, seed: 42)
        let x = MLXRandom.normal([2, 10, 64], key: MLXRandom.key(1))
        let y = linear(x)

        #expect(y.shape == [2, 10, 128])
    }

    @Test("Linear layer weights have correct shape")
    func testLinearWeights() {
        let linear = Linear(32, 64, seed: 42)

        #expect(linear.w.shape == [32, 64])
        #expect(linear.b.shape == [64])
    }
}

@Suite("Attention Output Shape Tests")
struct AttentionShapeTests {
    @Test("MHA produces correct output shape")
    func testMHAShape() {
        let b = 2
        let n = 16
        let dModel = 64
        let heads = 4

        let mha = MHA(b: b, n: n, dModel: dModel, heads: heads, seed: 42)
        let x = MLXRandom.normal([b, n, dModel], key: MLXRandom.key(1))
        eval(x)

        let y = mha.forward(x: x)
        eval(y)

        #expect(y.shape == [b, n, dModel])
    }

    @Test("GQA produces correct output shape")
    func testGQAShape() {
        let b = 2
        let n = 16
        let dModel = 64
        let heads = 8
        let kvHeads = 2

        let gqa = GQA(b: b, n: n, dModel: dModel, heads: heads, kvHeads: kvHeads, seed: 42)
        let x = MLXRandom.normal([b, n, dModel], key: MLXRandom.key(1))
        eval(x)

        let y = gqa.forward(x: x)
        eval(y)

        #expect(y.shape == [b, n, dModel])
    }

    @Test("MQA produces correct output shape")
    func testMQAShape() {
        let b = 2
        let n = 16
        let dModel = 64
        let heads = 4

        let mqa = MQA(b: b, n: n, dModel: dModel, heads: heads, seed: 42)
        let x = MLXRandom.normal([b, n, dModel], key: MLXRandom.key(1))
        eval(x)

        let y = mqa.forward(x: x)
        eval(y)

        #expect(y.shape == [b, n, dModel])
    }
}

@Suite("SDPA Tests")
struct SDPATests {
    @Test("SDPA produces correct output shape")
    func testSDPAShape() {
        let b = 2
        let h = 4
        let n = 16
        let dh = 32

        let q = MLXRandom.normal([b, h, n, dh], key: MLXRandom.key(1))
        let k = MLXRandom.normal([b, h, n, dh], key: MLXRandom.key(2))
        let v = MLXRandom.normal([b, h, n, dh], key: MLXRandom.key(3))
        eval([q, k, v])

        let out = sdpa(q: q, k: k, v: v, dh: dh)
        eval(out)

        #expect(out.shape == [b, h, n, dh])
    }

    @Test("SDPA output values are finite")
    func testSDPAFinite() {
        let q = MLXRandom.normal([1, 2, 8, 16], key: MLXRandom.key(1))
        let k = MLXRandom.normal([1, 2, 8, 16], key: MLXRandom.key(2))
        let v = MLXRandom.normal([1, 2, 8, 16], key: MLXRandom.key(3))
        eval([q, k, v])

        let out = sdpa(q: q, k: k, v: v, dh: 16)
        eval(out)

        let hasNaN = any(isNaN(out)).item(Bool.self)
        #expect(hasNaN == false)
    }
}

@Suite("Attention Name Tests")
struct AttentionNameTests {
    @Test("MHA has correct name")
    func testMHAName() {
        let mha = MHA(b: 1, n: 8, dModel: 32, heads: 4, seed: 42)
        #expect(mha.name == "MHA")
    }

    @Test("GQA has correct name")
    func testGQAName() {
        let gqa = GQA(b: 1, n: 8, dModel: 32, heads: 4, kvHeads: 2, seed: 42)
        #expect(gqa.name == "GQA")
    }

    @Test("MQA has correct name")
    func testMQAName() {
        let mqa = MQA(b: 1, n: 8, dModel: 32, heads: 4, seed: 42)
        #expect(mqa.name == "MQA")
    }
}

@Suite("BenchRow Tests")
struct BenchRowTests {
    @Test("BenchRow stores values correctly")
    func testBenchRowValues() {
        let row = BenchRow(
            name: "TestAttn",
            b: 4,
            n: 128,
            dModel: 512,
            heads: 8,
            kvHeads: 2,
            iters: 100,
            msPerIter: 1.5
        )

        #expect(row.name == "TestAttn")
        #expect(row.b == 4)
        #expect(row.n == 128)
        #expect(row.dModel == 512)
        #expect(row.heads == 8)
        #expect(row.kvHeads == 2)
        #expect(row.iters == 100)
        #expect(row.msPerIter == 1.5)
    }
}
