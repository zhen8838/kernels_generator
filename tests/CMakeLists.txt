enable_language(ASM)

add_compile_options(-fno-rtti)

find_package(nncase REQUIRED)
find_package(gtest REQUIRED)
find_package(celero REQUIRED)

# add test all
file(GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/include/generated_kernels/*.s)
add_executable(test_gen test_gen.cpp ${KERNEL_SRC})
add_dependencies(test_gen generated_kernels)
target_link_libraries(test_gen nncase gtest celero -ldl -lpthread)
