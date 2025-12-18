import XCTest
import MLX
import MLXRandom

@testable import AttnBenchLib

final class LinearTests: XCTestCase {
    func testLinearShape() {
        let linear = Linear(64, 128, seed: 42)
        let x = MLXRandom.normal([2, 10, 64], key: MLXRandom.key(1))
        let y = linear(x)

        XCTAssertEqual(y.shape, [2, 10, 128])
    }

    func testLinearWeights() {
        let linear = Linear(32, 64, seed: 42)

        XCTAssertEqual(linear.w.shape, [32, 64])
        XCTAssertEqual(linear.b.shape, [64])
    }
}

final class AttentionShapeTests: XCTestCase {
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

        XCTAssertEqual(y.shape, [b, n, dModel])
    }

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

        XCTAssertEqual(y.shape, [b, n, dModel])
    }

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

        XCTAssertEqual(y.shape, [b, n, dModel])
    }
}

final class SDPATests: XCTestCase {
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

        XCTAssertEqual(out.shape, [b, h, n, dh])
    }

    func testSDPAFinite() {
        let q = MLXRandom.normal([1, 2, 8, 16], key: MLXRandom.key(1))
        let k = MLXRandom.normal([1, 2, 8, 16], key: MLXRandom.key(2))
        let v = MLXRandom.normal([1, 2, 8, 16], key: MLXRandom.key(3))
        eval([q, k, v])

        let out = sdpa(q: q, k: k, v: v, dh: 16)
        eval(out)

        let hasNaN = any(isNaN(out)).item(Bool.self)
        XCTAssertFalse(hasNaN)
    }
}

final class AttentionNameTests: XCTestCase {
    func testMHAName() {
        let mha = MHA(b: 1, n: 8, dModel: 32, heads: 4, seed: 42)
        XCTAssertEqual(mha.name, "MHA")
    }

    func testGQAName() {
        let gqa = GQA(b: 1, n: 8, dModel: 32, heads: 4, kvHeads: 2, seed: 42)
        XCTAssertEqual(gqa.name, "GQA")
    }

    func testMQAName() {
        let mqa = MQA(b: 1, n: 8, dModel: 32, heads: 4, seed: 42)
        XCTAssertEqual(mqa.name, "MQA")
    }
}

final class BenchRowTests: XCTestCase {
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

        XCTAssertEqual(row.name, "TestAttn")
        XCTAssertEqual(row.b, 4)
        XCTAssertEqual(row.n, 128)
        XCTAssertEqual(row.dModel, 512)
        XCTAssertEqual(row.heads, 8)
        XCTAssertEqual(row.kvHeads, 2)
        XCTAssertEqual(row.iters, 100)
        XCTAssertEqual(row.msPerIter, 1.5)
    }
}
