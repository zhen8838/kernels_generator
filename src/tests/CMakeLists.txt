enable_language(ASM)
enable_testing()

hkg_get_runtime_lib(os_runtime_lib os_name)
hkg_get_suffix(obj_suffix lib_suffix)

macro(add_one_test name)
    add_executable(test_gen_${name} test_gen_${name}.cpp)
    add_dependencies(test_gen_${name} hkg::generated_kernels)    
    target_link_libraries(test_gen_conv2d GTest::gtest_main celero::celero hkg::${os_name}_src ${os_runtime_lib} -lpthread)
    add_test(NAME test_gen_${name} COMMAND test_gen_${name})
endmacro()

add_executable(test_feature test_feature.cpp)
target_link_libraries(test_feature GTest::gtest_main)
add_test(NAME test_feature COMMAND test_feature)
add_one_test(conv2d)