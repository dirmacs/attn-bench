// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "AttnBench",
    platforms: [
        .macOS(.v13)
    ],
    dependencies: [
        // MLX Swift dependency snippet is documented in the mlx-swift README. [web:31]
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "AttnBench",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ]
        ),
    ]
)
