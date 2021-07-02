enable_language(ASM)

# add test all
file(GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/include/hkg/halide_gnne_*.s)
add_executable(test_gen test_gen.cpp ${KERNEL_SRC})
add_dependencies(test_gen generated_kernels)
target_link_libraries(test_gen nncase GTest::gtest_main celero::celero -ldl -lpthread)


file(GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/include/hkg/halide_conv2d*.s)
add_executable(test_gen_conv2d test_gen_conv2d.cpp ${KERNEL_SRC})
add_dependencies(test_gen_conv2d generated_kernels)
target_link_libraries(test_gen_conv2d nncase GTest::gtest_main celero::celero -ldl -lpthread)
