enable_language(ASM)
enable_testing()

set(os_runtime_lib "")
hkg_get_runtime_lib(os_runtime_lib)

macro(add_one_test name)
    file(GLOB ASM_SRC ${CMAKE_SOURCE_DIR}/include/hkg/generated_kernels/halide_${name}*${os_name}*.s)
    add_executable(test_gen_${name} test_gen_${name}.cpp ${ASM_SRC})
    if(TARGET hkg_generated_kernels)
        # NOTE the cmake must dependence the generated_kernels
        add_dependencies(test_gen_${name} hkg_generated_kernels)    
    endif()
    target_link_libraries(test_gen_conv2d GTest::gtest_main celero::celero ${os_runtime_lib} -lpthread)
    add_test(NAME test_gen_${name} COMMAND test_gen_${name})
endmacro()

add_executable(test_feature test_feature.cpp)
target_link_libraries(test_feature GTest::gtest_main)
add_test(NAME test_feature COMMAND test_feature)
add_one_test(conv2d)