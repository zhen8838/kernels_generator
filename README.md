# kernels_generator
halide kernel generator

# local build all
```sh
cmake . -DCMAKE_BUILD_TYPE=Debug -Bbuild
cmake --build build -j 80
cmake --install build --prefix build/package
```

# local build conan like package

```sh
conan create . hkg/0.0.1@ --build=missing
cmake . -DCMAKE_BUILD_TYPE=Release -DENABLE_ONLY_BENCHMARK_TEST=ON -B build_test
cmake --build build_test --config Release
cd build_test/src/tests/
ctest --verbose -C
cd ../../..
```

# build pipeline
```sh
conan remove hkg
conan remove hkg -r sunnycase
conan create . hkg/0.0.1@ --build=missing
conan upload hkg/0.0.1  --all -r sunnycase 
```

# local rebuilding and test

```sh
rm -rf build include/hkg/generated_kernels/ 
cmake . -DCMAKE_BUILD_TYPE=Debug -Bbuild
cmake --build build -j 80
cmake --install build --prefix build/package
cmake . -DCMAKE_BUILD_TYPE=Release -DENABLE_ONLY_BENCHMARK_TEST=ON -B build_test -Dhkg_DIR=build/package/lib/cmake
cmake --build build_test --config Release
cd build_test/src/tests/
ctest --verbose -C
cd ../../..
```

```sh
cmake . -DCMAKE_BUILD_TYPE=Debug -B build -DHalide_DIR=/Users/lisa/Downloads/Halide-12.0.1-arm-64-osx/lib/cmake/Halide -DHalideHelpers_DIR=/Users/lisa/Downloads/Halide-12.0.1-arm-64-osx/lib/cmake/HalideHelpers -DENABLE_BENCHMARK=OFF -DENABLE_TEST=OFF 
cmake --build build
```