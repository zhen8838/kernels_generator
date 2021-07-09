cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

PROJECT(kernels_generator)

option(ENABLE_BENCHMARK "ENABLE_BENCHMARK" ON)
option(ENABLE_TEST "ENABLE_TEST" ON)
option(ENABLE_ONLY_BENCHMARK_TEST "use conan package only build and run benchmark & test, when enable this " OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

if(ENABLE_ONLY_BENCHMARK_TEST)
    include(cmake/conan.cmake)
    conan_add_remote(NAME sunnycase URL https://conan.sunnycase.moe INDEX 0)
    conan_cmake_run(CONANFILE conanfile-test-only.txt
                    BASIC_SETUP
                    BUILD missing)
    include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
    find_package(hkg REQUIRED)
    find_package(GTest REQUIRED)
    find_package(Celero REQUIRED)
    add_subdirectory(src/benchmark)
    add_subdirectory(src/tests)
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
    include(cmake/hkgHelper.cmake)
    find_package(Halide REQUIRED)
    
    include_directories(include)
    add_compile_options(-fno-rtti -Wall)

    # add gen all
    # TODO 重构halide打包脚本，生成正确依赖
    include_directories(${Halide_INCLUDES})
    if(EXISTS ${CONAN_HALIDE_ROOT})
        add_executable(kernels_generator src/kernels_generator.cpp ${CONAN_HALIDE_ROOT}/share/tools/GenGen.cpp)    
        target_link_libraries(kernels_generator Halide)
    else()
        add_executable(kernels_generator src/kernels_generator.cpp)    
        target_link_libraries(kernels_generator Halide::Generator)
    endif()

    # generate codes and runtime libs
    set(KERNEL_SRCS "")
    set(RUNTIME_LIBS "")
    set(RUNTIME_Targets "")
    # halide_generate_code(gnne_matmul gnne_matmul "" "${KERNEL_SRCS}")

    # foreach(WH "1;1" "3;3" "5;5" "7;7")
    #     list(GET WH 0 KH)
    #     list(GET WH -1 KW)
    #     halide_generate_code(gnne_conv2d "gnne_conv2d_${KH}x${KW}" "kernel_height=${KH};kernel_width=${KW}" "${KERNEL_SRCS}")
    # endforeach(WH)

    halide_generate_runtime(RUNTIME_LIBS RUNTIME_Targets)

    include(cmake/codegen.cmake)

    # add_custom_target(hkg_generated_kernels
    #     SOURCES ${KERNEL_SRCS}
    #     DEPENDS kernels_generator ${RUNTIME_LIBS})        
    # add dummy to force exec the generator
    add_library(hkg_generated_kernels src/dummy.cpp ${KERNEL_SRCS} ${RUNTIME_LIBS})
    add_dependencies(hkg_generated_kernels kernels_generator)
    add_library(hkg::generated_kernels ALIAS hkg_generated_kernels)
    set_target_properties(hkg_generated_kernels PROPERTIES EXPORT_NAME generated_kernels)

    # make src target for sub project use
    set(SRCS_Targets "")
    set(os_name_list "linux;osx;windows")
    set(src_name_list "LINUX_SRCS;OSX_SRCS;WIN_SRCS")
    foreach(os_name src_name IN ZIP_LISTS os_name_list src_name_list)
        # the `src_name` as a **pointer, so get value by `${${src_name}}`
        add_library(hkg_${os_name}_src INTERFACE)
        replace_src_path(${src_name} INSTALL_SRCS)
        message("ffff ${${src_name}}")
        target_link_libraries(hkg_${os_name}_src INTERFACE 
            $<BUILD_INTERFACE:"${${src_name}}">
            $<INSTALL_INTERFACE:${INSTALL_SRCS}>)
        add_library(hkg::${os_name}_src ALIAS hkg_${os_name}_src)    
        list(APPEND SRCS_Targets "hkg_${os_name}_src")
        set_target_properties(hkg_${os_name}_src PROPERTIES EXPORT_NAME ${os_name}_src)
    endforeach()

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
    else()
    endif()

    # ----------- install -----------

    # install all headers
    install(DIRECTORY include/hkg
            DESTINATION include)
    # NOTE install halide header from conan test
    install(FILES ${Halide_INCLUDES}/HalideBuffer.h ${Halide_INCLUDES}/HalideRuntime.h
            DESTINATION include)

    install(FILES ${RUNTIME_LIBS}
            DESTINATION lib)

    install(TARGETS ${RUNTIME_Targets} ${SRCS_Targets} EXPORT hkgTargets
        INCLUDES DESTINATION include
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

    install(TARGETS hkg_generated_kernels EXPORT hkgTargets)

    install(EXPORT hkgTargets
        DESTINATION lib/cmake/
        NAMESPACE hkg::)

    configure_file(cmake/hkgConfig.cmake.in hkgConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hkgConfig.cmake DESTINATION lib/cmake/)
    install(FILES cmake/hkgHelper.cmake DESTINATION lib/cmake/)
endif()