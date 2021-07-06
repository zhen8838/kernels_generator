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
set(RUNTIME_SRCS "")
set(GENERATED_RUNTIME_FLAG OFF)
set(GENERATED_DIR "generated_kernels")

# macro(halide_generate_code group_name func_name variable)
#     if(NOT GENERATED_RUNTIME_FLAG)
#         set(RUNTIME_STR "")
#         set(GENERATED_RUNTIME_FLAG true)
#     else()
#         set(RUNTIME_STR "-no_runtime")
#     endif()
    
#     add_custom_command(
#         OUTPUT ${CMAKE_SOURCE_DIR}/include/${GENERATED_DIR}/halide_${func_name}.s
#         COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/include/${GENERATED_DIR}
#         COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -g halide_${group_name} -n halide_${func_name} -f halide_${func_name} -o ${CMAKE_SOURCE_DIR}/include/${GENERATED_DIR} -e c_header,assembly,schedule,stmt target=host-no_asserts-no_bounds_query${RUNTIME_STR} ${variable}
#         DEPENDS kernels_generator
#     )
#     list(APPEND KERNEL_SRCS ${CMAKE_SOURCE_DIR}/include/${GENERATED_DIR}/halide_${func_name}.s) 
# endmacro()

macro(halide_generate_runtime)
    foreach(os_name linux;osx;windows)
        if(${os_name} STREQUAL "windows")
            set(suffix "lib")
        else()
            set(suffix "a")
        endif()

        add_custom_command(
            OUTPUT ${CMAKE_SOURCE_DIR}/include/${GENERATED_DIR}/halide_runtime_${os_name}.${suffix}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/include/${GENERATED_DIR}
            COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -r halide_runtime_${os_name} -o ${CMAKE_SOURCE_DIR}/include/${GENERATED_DIR} -e static_library,c_header target=${os_name}-x86-64
            DEPENDS kernels_generator
        )
        list(APPEND RUNTIME_SRCS ${CMAKE_SOURCE_DIR}/include/${GENERATED_DIR}/halide_runtime_${os_name}.${suffix})
    endforeach()
endmacro(halide_generate_runtime)

macro(halide_generate_code_multi_os group_name func_name variable os_name)
    set(FUNC_BASE_NAME halide_${func_name}_${os_name})
    set(OUTPUT_BASE_NAME  ${CMAKE_SOURCE_DIR}/include/${GENERATED_DIR}/${FUNC_BASE_NAME})
    set(TARGET_BASE_NAME  no_asserts-no_bounds_query-no_runtime-${os_name}-x86-64)
    set(OUTPUT_DIR  ${CMAKE_SOURCE_DIR}/include/${GENERATED_DIR})
    
    add_custom_command(
        OUTPUT ${OUTPUT_BASE_NAME}_avx2.s
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -g halide_${group_name} -f ${FUNC_BASE_NAME}_avx2 -o ${OUTPUT_DIR} -e c_header,assembly,schedule,stmt target=${TARGET_BASE_NAME}-avx2-fma ${variable}
        DEPENDS kernels_generator
    )

    add_custom_command(
        OUTPUT ${OUTPUT_BASE_NAME}_sse41.s
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -g halide_${group_name} -f ${FUNC_BASE_NAME}_sse41 -o ${OUTPUT_DIR} -e c_header,assembly,schedule,stmt target=${TARGET_BASE_NAME}-sse41 ${variable}
        DEPENDS kernels_generator
    )

    add_custom_command(
        OUTPUT ${OUTPUT_BASE_NAME}_bare.s
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -g halide_${group_name} -f ${FUNC_BASE_NAME}_bare -o ${OUTPUT_DIR} -e c_header,assembly,schedule,stmt target=${TARGET_BASE_NAME} ${variable}
        DEPENDS kernels_generator
    )

    list(APPEND KERNEL_SRCS 
        ${OUTPUT_BASE_NAME}_avx2.s
        ${OUTPUT_BASE_NAME}_sse41.s
        ${OUTPUT_BASE_NAME}_bare.s
    ) 
endmacro()


macro(halide_generate_code group_name func_name variable)
    foreach(os_name linux;osx;windows)
        halide_generate_code_multi_os("${group_name}" "${func_name}" "${variable}" "${os_name}")
    endforeach()
endmacro()


include_directories(include)
add_compile_options(-fno-rtti -Wall)

# add gen all
# TODO 重构halide打包脚本，生成正确依赖
include_directories(${Halide_INCLUDES})
add_executable(kernels_generator kernels_generator.cpp ${CONAN_HALIDE_ROOT}/share/tools/GenGen.cpp)
target_link_libraries(kernels_generator Halide)

# generate codes
# halide_generate_code(gnne_matmul gnne_matmul "")

# foreach(WH "1;1" "3;3" "5;5" "7;7")
#     list(GET WH 0 KH)
#     list(GET WH -1 KW)
#     halide_generate_code(gnne_conv2d "gnne_conv2d_${KH}x${KW}" "kernel_height=${KH};kernel_width=${KW}")
# endforeach(WH)

set(GENERATED_RUNTIME_FLAG OFF)
halide_generate_runtime()

foreach(WH "1;1" "3;3" "5;5" "7;7")
    list(GET WH 0 KH)
    list(GET WH -1 KW)
    halide_generate_code(conv2d "conv2d_${KH}x${KW}" "kernel_height=${KH};kernel_width=${KW}")
endforeach(WH)    

add_custom_target(generated_kernels
    SOURCES ${KERNEL_SRCS}
    DEPENDS kernels_generator ${RUNTIME_SRCS})

option(ENABLE_BENCHMARK "ENABLE_BENCHMARK" ON)
option(ENABLE_TEST "ENABLE_TEST" ON)
if(ENABLE_BENCHMARK)
    conan_cmake_run(
        CONANFILE conanfile-dev.txt
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
install(DIRECTORY include/${GENERATED_DIR}
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.s"
        PATTERN "*.stmt"
        PATTERN "*.a"
)

install(DIRECTORY include/hkg
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
)

install(FILES ${CONAN_HALIDE_ROOT}/include/HalideBuffer.h 
              ${CONAN_HALIDE_ROOT}/include/HalideRuntime.h
        DESTINATION include/${GENERATED_DIR})