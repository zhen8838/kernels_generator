enable_language(ASM)

# add test all
# file(GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/include/hkg/halide_gnne_*.s)
# add_executable(test_gen test_gen.cpp ${KERNEL_SRC})
# add_dependencies(test_gen generated_kernels)
# target_link_libraries(test_gen nncase GTest::gtest_main celero::celero -ldl -lpthread)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
   set(os_name "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(os_name "osx")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(os_name "windows")
endif()

if(${os_name} STREQUAL "Windows")
    set(suffix "lib")
else()
    set(suffix "a")
endif()

set(runtime_lib_name "${CMAKE_SOURCE_DIR}/include/${GENERATED_DIR}/halide_runtime_${os_name}.${suffix}")

file(GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/include/generated_kernels/halide_conv2d*${os_name}*.s)
add_executable(test_gen_conv2d test_gen_conv2d.cpp ${KERNEL_SRC})
add_dependencies(test_gen_conv2d generated_kernels)
target_link_libraries(test_gen_conv2d nncase GTest::gtest_main celero::celero ${runtime_lib_name} -ldl -lpthread)
