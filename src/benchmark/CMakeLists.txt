 
hkg_get_runtime_lib(os_runtime_lib os_name)
hkg_get_suffix(obj_suffix lib_suffix)

function(add_one_bench name)
    add_executable(bench_${name} bench_${name}.cpp)
    add_dependencies(bench_${name} hkg::generated_kernels)
    target_link_libraries(bench_${name} celero::celero hkg::${os_name}_src ${os_runtime_lib} -pthread)    
endfunction()

add_one_bench(conv2d)
add_one_bench(conv2d_depthwise)
