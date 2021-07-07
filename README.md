# kernels_generator
halide kernel generator

# build with conan
```sh
cmake . -DCMAKE_BUILD_TYPE=Debug -Bbuild
cmake --build build -j 80
cmake --install build --prefix build/package
```

# make conan package

```sh
conan create . 0.0.1@ --build=missing
```



