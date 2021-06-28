# kernels_generator
halide kernel generator

# build with manual
```sh
take build
conan install .. -s build_type=Debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j
```

# build with conan
```sh
take build
conan install ..
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j
```

# benchmark & test

require nncase

```sh
take build
cmake .. -DCMAKE_BUILD_TYPE=Debug \
    -DHalide_DIR=/root/.conan/data/Halide/12.0.0/_/_/package/4fc5ce8671fa5c0c1d8d2163ff0c55731437d741/lib/cmake/Halide \
    -DHalideHelpers_DIR=/root/.conan/data/Halide/12.0.0/_/_/package/4fc5ce8671fa5c0c1d8d2163ff0c55731437d741/lib/cmake/HalideHelpers/ \
    -DENABLE_BENCHMARK=ON \
    -DENABLE_TEST=ON
make -j
```

