enable_language(ASM)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
   set(os_name "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(os_name "osx")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(os_name "windows")
endif()

set(runtime_lib hkg::${os_name}_runtime)

macro(add_one_test name)
    file(GLOB ASM_SRC ${CMAKE_SOURCE_DIR}/include/hkg/generated_kernels/halide_${name}*${os_name}*.s)
    add_executable(test_gen_${name} test_gen_${name}.cpp ${ASM_SRC})
    add_dependencies(test_gen_${name} generated_kernels)
    target_link_libraries(test_gen_conv2d GTest::gtest_main celero::celero ${runtime_lib} -lpthread)
    add_test(NAME test_gen_${name} COMMAND test_gen_${name})
endmacro()

add_one_test(conv2d)