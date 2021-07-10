function(hkg_get_runtime_lib os_runtime os_name)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(cur_os_name "linux")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(cur_os_name "osx")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(cur_os_name "windows")
    endif()
    set(${os_runtime} hkg::${cur_os_name}_runtime PARENT_SCOPE)    
    set(${os_name} ${cur_os_name} PARENT_SCOPE)
endfunction()


function(hkg_get_suffix obj_suffix lib_suffix)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(${obj_suffix} "obj" PARENT_SCOPE)
        set(${lib_suffix} "lib" PARENT_SCOPE)
    else()
        set(${obj_suffix} "o" PARENT_SCOPE)
        set(${lib_suffix} "a" PARENT_SCOPE)
    endif()
    if(NOT "${ARGN}" STREQUAL "")
        if("${ARGN}" MATCHES "windows")
            set(${obj_suffix} "obj" PARENT_SCOPE)
            set(${lib_suffix} "lib" PARENT_SCOPE)
        else()
            set(${obj_suffix} "o" PARENT_SCOPE)
            set(${lib_suffix} "a" PARENT_SCOPE)
        endif()
    endif()
endfunction()


function(halide_generate_runtime RUNTIME_LIBS RUNTIME_Targets)
    set(internel_runtime_libs "")
    set(internel_runtime_targets "")
    set(extra_lib "")
    foreach(os_name linux;osx;windows)
        set(obj_suffix "")
        set(lib_suffix "")
        hkg_get_suffix(obj_suffix lib_suffix ${os_name})
        if(${os_name} STREQUAL windows)
           set(extra_lib "") 
        else()
            set(extra_lib "-ldl")
        endif()

        add_custom_command(
            OUTPUT ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/halide_runtime_${os_name}.${lib_suffix}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -r halide_runtime_${os_name} -o ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -e static_library,c_header target=${os_name}-x86-64
            DEPENDS kernels_generator
        )
        add_library(hkg_${os_name}_runtime INTERFACE)
        target_link_libraries(hkg_${os_name}_runtime INTERFACE
        $<BUILD_INTERFACE:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/halide_runtime_${os_name}.${lib_suffix}> 
        $<INSTALL_INTERFACE:$\{_IMPORT_PREFIX\}/lib/halide_runtime_${os_name}.${lib_suffix}> ${extra_lib}) # NOTE need find better method to export lib path.
        set_target_properties(hkg_${os_name}_runtime PROPERTIES EXPORT_NAME ${os_name}_runtime)
        add_library(hkg::${os_name}_runtime ALIAS hkg_${os_name}_runtime)

        list(APPEND internel_runtime_libs ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/halide_runtime_${os_name}.${lib_suffix})    
        list(APPEND internel_runtime_targets hkg_${os_name}_runtime)    
    endforeach()
    set(${RUNTIME_LIBS} ${internel_runtime_libs} PARENT_SCOPE)
    set(${RUNTIME_Targets} ${internel_runtime_targets} PARENT_SCOPE)
endfunction()

function(halide_generate_code_multi_os group_name func_name variable os_name RET_SRC RET_HEADER)
    set(FUNC_BASE_NAME halide_${func_name}_${os_name})
    set(OUTPUT_BASE_NAME  ${CMAKE_SOURCE_DIR}/include/hkg/generated_kernels/${FUNC_BASE_NAME})
    set(HEADER_BAST_NAME hkg/generated_kernels/${FUNC_BASE_NAME})
    set(TARGET_BASE_NAME  no_asserts-no_bounds_query-no_runtime-${os_name}-x86-64)
    set(OUTPUT_DIR  ${CMAKE_SOURCE_DIR}/include/hkg/generated_kernels)
    set(SRCS "")
    set(HEADER "")

    set(obj_suffix "")
    set(lib_suffix "")
    hkg_get_suffix(obj_suffix lib_suffix ${os_name})
    
    # four version
    list(APPEND feature_list 
        "avx512" 
        "avx2" 
        "sse41" 
        "bare")
    list(APPEND full_feature_list 
        "-sse41-avx-f16c-fma-avx2-avx512" 
        "-avx-avx2-f16c-fma-sse41"
        "-avx-f16c-sse41"
        "")

    foreach(feature full_feature IN ZIP_LISTS feature_list full_feature_list)
        add_custom_command(
            OUTPUT ${OUTPUT_BASE_NAME}_${feature}.${obj_suffix}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
            COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -g halide_${group_name} -f ${FUNC_BASE_NAME}_${feature} -o ${OUTPUT_DIR} -e c_header,object,schedule,stmt target=${TARGET_BASE_NAME}${full_feature} ${variable}
            DEPENDS kernels_generator
        )
        list(APPEND SRCS ${OUTPUT_BASE_NAME}_${feature}.${obj_suffix})
        list(APPEND HEADER ${HEADER_BAST_NAME}_${feature}.h)
    endforeach()
    set(${RET_SRC} ${SRCS} PARENT_SCOPE)
    set(${RET_HEADER} ${HEADER} PARENT_SCOPE)
endfunction()

function(halide_generate_code group_name func_name variable 
        RET_LINUX_SRCS RET_OSX_SRCS RET_WIN_SRCS 
        RET_LINUX_HEADER RET_OSX_HEADER RET_WIN_HEADER)
    halide_generate_code_multi_os("${group_name}" "${func_name}" "${variable}" linux LINUX_SRCS LINUX_HEADER)
    halide_generate_code_multi_os("${group_name}" "${func_name}" "${variable}" osx OSX_SRCS OSX_HEADER)
    halide_generate_code_multi_os("${group_name}" "${func_name}" "${variable}" windows WIN_SRCS WIN_HEADER)

    set(${RET_LINUX_SRCS} ${LINUX_SRCS} PARENT_SCOPE)
    set(${RET_OSX_SRCS} ${OSX_SRCS} PARENT_SCOPE)
    set(${RET_WIN_SRCS} ${WIN_SRCS} PARENT_SCOPE)
    set(${RET_LINUX_HEADER} ${LINUX_HEADER} PARENT_SCOPE)
    set(${RET_OSX_HEADER} ${OSX_HEADER} PARENT_SCOPE)
    set(${RET_WIN_HEADER} ${WIN_HEADER} PARENT_SCOPE)
endfunction()


function(concat_header header_list RET_include_list)
    set(include_list "")
    foreach(header IN LISTS header_list)
        set(include_list "${include_list}\r\n#include \"${header}\"")
    endforeach()
    set(${RET_include_list} ${include_list} PARENT_SCOPE)
endfunction()

function(insert_header func_name linux_header_list osx_header_list windows_header_list)
    concat_header("${linux_header_list}" linux_include_list)
    concat_header("${osx_header_list}" osx_include_list)
    concat_header("${windows_header_list}" windows_include_list)
    configure_file(include/hkg/export/halide_${func_name}.h.in ${CMAKE_SOURCE_DIR}/include/hkg/export/halide_${func_name}.h)
endfunction()

function(replace_src_path cur_src_path_list RET_INSTALL_SRCS_PATH)
    set(INSTALL_SRCS_PATH "")
    string(LENGTH ${CMAKE_SOURCE_DIR} length)
    foreach(src_path IN LISTS ${cur_src_path_list})
        string(SUBSTRING ${src_path} ${length} -1 no_prefix_path)
        set(install_path "$\{_IMPORT_PREFIX\}${no_prefix_path}") 
        # message(${install_path})
        list(APPEND INSTALL_SRCS_PATH ${install_path})
    endforeach()
    set(${RET_INSTALL_SRCS_PATH} ${INSTALL_SRCS_PATH} PARENT_SCOPE)
endfunction(replace_src_path)
