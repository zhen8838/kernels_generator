cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

PROJECT(kernels_generator)

if(CONAN_EXPORTED) # in conan local cache
    message(STATUS "Standard Conan Installation")
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
else() # in user space
    message(STATUS "Auto Cmake Conan Installation")
    include(cmake/conan.cmake)
    conan_cmake_run(CONANFILE conanfile.py
                    BASIC_SETUP
                    BUILD missing)
endif()
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

find_package(Halide REQUIRED)

# for generate code
set(KERNEL_SRCS "")
set(GENERATED_RUNTIME_FLAG OFF)
macro(halide_generate_code group_name func_name variable)
    if(NOT GENERATED_RUNTIME_FLAG)
        set(RUNTIME_STR "")
        set(GENERATED_RUNTIME_FLAG true)
    else()
        set(RUNTIME_STR "-no_runtime")
    endif()
    
    add_custom_command(
        OUTPUT ${CMAKE_SOURCE_DIR}/include/hkg/halide_${func_name}.s
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/include/hkg
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -g halide_${group_name} -n halide_${func_name} -f halide_${func_name} -o ${CMAKE_SOURCE_DIR}/include/hkg -e c_header,assembly,schedule,stmt target=host-no_asserts-no_bounds_query${RUNTIME_STR} ${variable}
        DEPENDS kernels_generator
    )
    list(APPEND KERNEL_SRCS ${CMAKE_SOURCE_DIR}/include/hkg/halide_${func_name}.s) 
endmacro()


include_directories(include)
add_compile_options(-fno-rtti -Wall)

# add gen all
include_directories(${Halide_INCLUDES})
add_executable(kernels_generator kernels_generator.cpp ${CONAN_HALIDE_ROOT}/share/tools/GenGen.cpp)
target_link_libraries(kernels_generator Halide)

# generate codes
halide_generate_code(gnne_matmul gnne_matmul "")

foreach(WH "1;1" "3;3" "5;5" "7;7")
    list(GET WH 0 KH)
    list(GET WH -1 KW)
    halide_generate_code(gnne_conv2d "gnne_conv2d_${KH}x${KW}" "kernel_height=${KH};kernel_width=${KW}")
endforeach(WH)

set(GENERATED_RUNTIME_FLAG OFF)
foreach(WH "1;1" "3;3" "5;5" "7;7")
    list(GET WH 0 KH)
    list(GET WH -1 KW)
    halide_generate_code(conv2d "conv2d_${KH}x${KW}" "kernel_height=${KH};kernel_width=${KW}")
endforeach(WH)

add_custom_target(generated_kernels ALL
    SOURCES ${KERNEL_SRCS}
    DEPENDS kernels_generator)

option(ENABLE_BENCHMARK "ENABLE_BENCHMARK" ON)
option(ENABLE_TEST "ENABLE_TEST" ON)
if(ENABLE_BENCHMARK)
    conan_cmake_run(CONANFILE conanfile-dev.txt
        BASIC_SETUP
        BUILD missing)
    include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

    find_package(nncase REQUIRED)
    find_package(GTest REQUIRED)
    find_package(Celero REQUIRED)

    add_subdirectory(src/benchmark)
    if(ENABLE_TEST)
        add_subdirectory(src/tests)
    endif()
endif()

# install kernel header code
install(DIRECTORY include/hkg
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.s"
        PATTERN "*.stmt"
)

install(FILES ${CONAN_HALIDE_ROOT}/include/HalideBuffer.h 
              ${CONAN_HALIDE_ROOT}/include/HalideRuntime.h
        DESTINATION include/hkg)