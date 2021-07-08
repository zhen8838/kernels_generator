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
cmake . -DCMAKE_BUILD_TYPE=Debug -DENABLE_ONLY_BENCHMARK_TEST=ON -B build_test
cmake --build build_test --config Debug
cd build_test/src/tests/
ctest --verbose -C
```

# build pipeline
```sh
conan remove hkg
conan remove hkg -r sunnycase
conan create . hkg/0.0.1@ --build=missing
conan upload hkg/0.0.1  --all -r sunnycase 
```
