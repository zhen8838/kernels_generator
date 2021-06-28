cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

PROJECT(kernels_generator)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# 各种宏 
set(KERNEL_SRCS "")
macro(halide_generate_code group_name func_name variable)
    add_custom_command(
        OUTPUT ${CMAKE_SOURCE_DIR}/include/generated_kernels/halide_${func_name}.s
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/include/generated_kernels
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -g halide_${group_name} -n halide_${func_name} -f halide_${func_name} -o ${CMAKE_SOURCE_DIR}/include/generated_kernels -e c_header,assembly,schedule,stmt target=host-no_asserts-no_bounds_query ${variable}
        DEPENDS kernels_generator
    )
    list(APPEND KERNEL_SRCS ${CMAKE_SOURCE_DIR}/include/generated_kernels/halide_${func_name}.s) 
endmacro()

find_package(Halide REQUIRED)

include_directories(include)
add_compile_options(-fno-rtti)

# add gen all
include_directories(${Halide_INCLUDE_DIR})
add_executable(kernels_generator ${Halide_RES_DIRS}/GenGen.cpp kernels_generator.cpp)
target_link_libraries(kernels_generator Halide)

# generate codes
halide_generate_code(gnne_matmul gnne_matmul "")

foreach(WH "1;1" "3;3" "5;5" "7;7")
    list(GET WH 0 KH)
    list(GET WH -1 KW)
    halide_generate_code(gnne_conv2d "gnne_conv2d_${KH}x${KW}" "kernel_height=${KH};kernel_width=${KW}")
endforeach(WH)

add_custom_target(generated_kernels ALL
    SOURCES ${KERNEL_SRCS}
    DEPENDS kernels_generator)

option(ENABLE_BENCHMARK "ENABLE_BENCHMARK" OFF)
option(ENABLE_TEST "ENABLE_TEST" OFF)
if(ENABLE_BENCHMARK)
    add_subdirectory(benchmark)
    if(ENABLE_TEST)
        add_subdirectory(tests)
    endif()
endif()

install(DIRECTORY include/generated_kernels
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.s"
        PATTERN "*.stmt"
)