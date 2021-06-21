enable_language(ASM)

add_compile_options(-fno-rtti)

include(../cmake/conan.cmake)
conan_check()
conan_add_remote(NAME bincrafts URL https://api.bintray.com/conan/bincrafters/public-conan)
conan_add_remote(NAME sunnycase URL https://conan.sunnycase.moe INDEX 0)
conan_cmake_run(CONANFILE ../conanfile.txt
                    BASIC_SETUP CMAKE_TARGETS
                    SETTINGS compiler.cppstd=20
                    BUILD missing)
    include(${CMAKE_BINARY_DIR}/tests/conan_paths.cmake)

find_package(nncase REQUIRED)

# add test all
file(GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/include/generated_kernels/*.s)
add_executable(test_gen test_gen.cpp ${KERNEL_SRC})
add_dependencies(test_gen generated_kernels)
target_link_libraries(test_gen nncase CONAN_PKG::gtest CONAN_PKG::celero -ldl -lpthread)
