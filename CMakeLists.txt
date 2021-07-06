cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

PROJECT(kernels_generator)

option(ENABLE_BENCHMARK "ENABLE_BENCHMARK" ON)
option(ENABLE_TEST "ENABLE_TEST" ON)
option(ENABLE_ONLY_BENCHMARK_TEST "use conan package only build and run benchmark & test, when enable this " OFF)

if(ENABLE_ONLY_BENCHMARK_TEST)
    include(cmake/conan.cmake)
    conan_cmake_run(CONANFILE conanfile-test-only.txt
                    BASIC_SETUP
                    BUILD missing)
    include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
    find_package(hkg REQUIRED)
    add_subdirectory(benchmark)
    add_subdirectory(tests)
else()
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
    # for generate code
    set(KERNEL_SRCS "")
    set(RUNTIME_LIBS "")
    include(cmake/utils.cmake)

    find_package(Halide REQUIRED)

    include_directories(include)
    add_compile_options(-fno-rtti -Wall)

    # add gen all
    # TODO 重构halide打包脚本，生成正确依赖
    include_directories(${Halide_INCLUDES})
    if(EXISTS ${CONAN_HALIDE_ROOT})
        add_executable(kernels_generator kernels_generator.cpp ${CONAN_HALIDE_ROOT}/share/tools/GenGen.cpp)    
        target_link_libraries(kernels_generator Halide)
    else()
        add_executable(kernels_generator kernels_generator.cpp)    
        target_link_libraries(kernels_generator Halide::Generator)
    endif()

    # generate codes
    # halide_generate_code(gnne_matmul gnne_matmul "")

    # foreach(WH "1;1" "3;3" "5;5" "7;7")
    #     list(GET WH 0 KH)
    #     list(GET WH -1 KW)
    #     halide_generate_code(gnne_conv2d "gnne_conv2d_${KH}x${KW}" "kernel_height=${KH};kernel_width=${KW}")
    # endforeach(WH)

    halide_generate_runtime()

    foreach(WH "1;1" "3;3" "5;5" "7;7")
        list(GET WH 0 KH)
        list(GET WH -1 KW)
        halide_generate_code(conv2d "conv2d_${KH}x${KW}" "kernel_height=${KH};kernel_width=${KW}")
    endforeach(WH)


    add_custom_target(generated_kernels
        SOURCES ${KERNEL_SRCS}
        DEPENDS kernels_generator ${RUNTIME_LIBS})
    
    # sub code
    if(ENABLE_BENCHMARK)
        conan_cmake_run(
            CONANFILE conanfile-test.txt
            BASIC_SETUP
            BUILD missing)
        include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

        find_package(GTest REQUIRED)
        find_package(Celero REQUIRED)

        add_subdirectory(src/benchmark)
        if(ENABLE_TEST)
            add_subdirectory(src/tests)
        endif()
    endif()

    # ----------- install -----------

    # install generated kernels
    install(DIRECTORY include/${GENERATED_DIR}
            DESTINATION include
            FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.s"
            PATTERN "*.stmt"
            PATTERN "*.a"
    )

    # install headers
    install(DIRECTORY include/hkg
            DESTINATION include
            FILES_MATCHING
            PATTERN "*.h"
    )
endif()