// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "AttnBench",
    platforms: [
        .macOS("13.3")
    ],
    products: [
        .executable(name: "AttnBench", targets: ["AttnBench"]),
        .library(name: "AttnBenchLib", targets: ["AttnBenchLib"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0")
    ],
    targets: [
        // Core library containing attention implementations
        .target(
            name: "AttnBenchLib",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            path: "Sources/AttnBenchLib"
        ),
        // Executable that runs benchmarks
        .executableTarget(
            name: "AttnBench",
            dependencies: ["AttnBenchLib"],
            path: "Sources/AttnBench"
        ),
        // Test target
        .testTarget(
            name: "AttnBenchTests",
            dependencies: [
                "AttnBenchLib",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ]
        )
    ]
)
