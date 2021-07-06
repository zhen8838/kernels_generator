enable_language(ASM)

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


macro(add_one_test name)
    file(GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/include/generated_kernels/halide_${name}*${os_name}*.s)
    add_executable(test_gen_${name} test_gen_${name}.cpp ${KERNEL_SRC})
    add_dependencies(test_gen_${name} generated_kernels)
    target_link_libraries(test_gen_conv2d GTest::gtest_main celero::celero ${runtime_lib_name} -ldl -lpthread)
    add_test(NAME test_gen_${name} COMMAND test_gen_${name})
endmacro()

add_one_test(conv2d)