#!/bin/zsh
rm -rf build include/hkg/generated_kernels/  # 清除cache
cmake . -DCMAKE_BUILD_TYPE=Release -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=true   # 本地构建
cmake --build build -j 
cmake --install build --prefix build/package

cmake . -DCMAKE_BUILD_TYPE=Release -DENABLE_ONLY_BENCHMARK_TEST=ON -B build_test -Dhkg_DIR=build/package/lib/cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=true # 本地编译test
cmake --build build_test -j