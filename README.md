# kernels_generator
halide kernel generator

# 本地编译整体

```sh
git clone git@github.com:zhen8838/conanPkg.git
cd conanPkg
./local_create.sh
cd ../kernels_generator
cmake . -DCMAKE_BUILD_TYPE=Release -Bbuild
cmake --build build -j
cmake --install build --prefix build/package
```

# 本地制作Conan包 以及 编译test

test的运作方式直接从云端拉取生成后的算子package. 所以需要本地为hkg打包,然后使用打包后的程序编译test.
注意每次升级hkg都需要更新版本号,否则会影响之前代码.

```sh
conan create . hkg/0.0.x@ --build=missing
cmake . -DCMAKE_BUILD_TYPE=Release -DENABLE_ONLY_BENCHMARK_TEST=ON -B build_test
cmake --build build_test -j
cd build_test/src/tests/
ctest --verbose -C
cd ../../..
```

# 等待测试通过后上传到云端

```sh
conan remove hkg
conan remove hkg -r sunnycase
conan create . hkg/0.0.x@ --build=missing
conan upload hkg/0.0.x  --all -r sunnycase 
```

# 本地rebuild与测试

因为每次编译后cmake会生成算子头文件,因此重新编译需要删除.

```sh
rm -rf build include/hkg/generated_kernels/ build_test  # 清除cache
cmake . -DCMAKE_BUILD_TYPE=Release -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=true   # 本地构建
cmake --build build -j 
cmake --install build --prefix build/package

cmake . -DCMAKE_BUILD_TYPE=Release -DENABLE_ONLY_BENCHMARK_TEST=ON -B build_test -Dhkg_DIR=build/package/lib/cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=true # 本地编译test
cmake --build build_test -j
cd build_test/src/tests/
ctest --verbose -C
cd ../../..
```