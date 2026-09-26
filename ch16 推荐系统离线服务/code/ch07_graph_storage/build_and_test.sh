#!/bin/bash
# build_and_test.sh — 编译并运行 ch07 图存储 C++ 测试
# 依赖：clang++ (macOS) 或 g++ (Linux)
set -e

# 检测系统并设置编译器参数
if [[ "$(uname)" == "Darwin" ]]; then
    SDK=$(xcrun --show-sdk-path 2>/dev/null || echo "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
    CXX="clang++ -std=c++17 -stdlib=libc++ -isysroot ${SDK}"
else
    CXX="g++ -std=c++17"
fi

echo "=== Building ch07 Graph Storage Tests ==="
echo "Compiler: ${CXX}"
${CXX} -O2 -Wall -I. test_graph_service.cpp -o test_graph_service

echo "=== Running Tests ==="
./test_graph_service

echo "=== Done ==="
