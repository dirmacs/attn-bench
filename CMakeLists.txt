cmake_minimum_required(VERSION 3.24)
project(AttnBench LANGUAGES C CXX Swift)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Fetch mlx-swift which brings in mlx-c and builds Metal shaders
FetchContent_Declare(
  mlx-swift
  GIT_REPOSITORY "https://github.com/ml-explore/mlx-swift.git"
  GIT_TAG "0.29.1"
)
FetchContent_MakeAvailable(mlx-swift)

# AttnBenchLib - Core library with attention implementations
file(GLOB ATTNBENCHLIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/Sources/AttnBenchLib/*.swift)
add_library(AttnBenchLib STATIC ${ATTNBENCHLIB_SOURCES})
target_link_libraries(AttnBenchLib PUBLIC
  MLX
  MLXNN
  MLXRandom
)

# AttnBench executable
add_executable(AttnBench
  ${CMAKE_CURRENT_LIST_DIR}/Sources/AttnBench/AttnBench.swift
)
target_link_libraries(AttnBench PRIVATE AttnBenchLib)
target_compile_options(AttnBench PRIVATE -parse-as-library)

# Note: Tests are run via xcodebuild or SwiftPM, not CMake
# Use: xcodebuild test -scheme AttnBench -destination 'platform=OS X'
