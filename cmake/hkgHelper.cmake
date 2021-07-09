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


macro(halide_generate_runtime)
    if(NOT DEFINED RUNTIME_LIBS OR NOT DEFINED RUNTIME_Targets)
        message(FATAL_ERROR "you must set the RUNTIME_LIBS and RUNTIME_Targets variable to save the generated source file path" )
    endif()
    foreach(os_name linux;osx;windows)
        set(obj_suffix "")
        set(lib_suffix "")
        hkg_get_suffix(obj_suffix lib_suffix ${os_name})

        add_custom_command(
            OUTPUT ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/halide_runtime_${os_name}.${lib_suffix}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -r halide_runtime_${os_name} -o ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -e static_library,c_header target=${os_name}-x86-64
            DEPENDS kernels_generator
        )
        add_library(hkg_${os_name}_runtime INTERFACE)
        target_link_libraries(hkg_${os_name}_runtime INTERFACE
        $<BUILD_INTERFACE:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/halide_runtime_${os_name}.${lib_suffix}> 
        $<INSTALL_INTERFACE:$\{_IMPORT_PREFIX\}/lib/halide_runtime_${os_name}.${lib_suffix}> -ldl) # NOTE need find better method to export lib path.
        set_target_properties(hkg_${os_name}_runtime PROPERTIES EXPORT_NAME ${os_name}_runtime)
        add_library(hkg::${os_name}_runtime ALIAS hkg_${os_name}_runtime)

        list(APPEND RUNTIME_LIBS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/halide_runtime_${os_name}.${lib_suffix})    
        list(APPEND RUNTIME_Targets hkg_${os_name}_runtime)    
    endforeach()
endmacro(halide_generate_runtime)

macro(halide_generate_code_multi_os group_name func_name variable os_name)
    set(FUNC_BASE_NAME halide_${func_name}_${os_name})
    set(OUTPUT_BASE_NAME  ${CMAKE_SOURCE_DIR}/include/hkg/${GENERATED_DIR}/${FUNC_BASE_NAME})
    set(TARGET_BASE_NAME  no_asserts-no_bounds_query-no_runtime-${os_name}-x86-64)
    set(OUTPUT_DIR  ${CMAKE_SOURCE_DIR}/include/hkg/${GENERATED_DIR})
    set(obj_suffix "")
    set(lib_suffix "")
    hkg_get_suffix(obj_suffix lib_suffix ${os_name})

    add_custom_command(
        OUTPUT ${OUTPUT_BASE_NAME}_avx2.${obj_suffix}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -g halide_${group_name} -f ${FUNC_BASE_NAME}_avx2 -o ${OUTPUT_DIR} -e c_header,object,schedule,stmt target=${TARGET_BASE_NAME}-avx2-fma ${variable}
        DEPENDS kernels_generator
    )

    add_custom_command(
        OUTPUT ${OUTPUT_BASE_NAME}_sse41.${obj_suffix}
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -g halide_${group_name} -f ${FUNC_BASE_NAME}_sse41 -o ${OUTPUT_DIR} -e c_header,object,schedule,stmt target=${TARGET_BASE_NAME}-sse41 ${variable}
        DEPENDS kernels_generator
    )

    add_custom_command(
        OUTPUT ${OUTPUT_BASE_NAME}_bare.${obj_suffix}
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -g halide_${group_name} -f ${FUNC_BASE_NAME}_bare -o ${OUTPUT_DIR} -e c_header,object,schedule,stmt target=${TARGET_BASE_NAME} ${variable}
        DEPENDS kernels_generator
    )

    list(APPEND KERNEL_SRCS 
        ${OUTPUT_BASE_NAME}_avx2.${obj_suffix}
        ${OUTPUT_BASE_NAME}_sse41.${obj_suffix}
        ${OUTPUT_BASE_NAME}_bare.${obj_suffix}
    ) 
endmacro()


macro(halide_generate_code group_name func_name variable)
    if(NOT DEFINED KERNEL_SRCS)
        message(FATAL_ERROR "you must set the KERNEL_SRCS variable to save the generated source file path" )
    endif()
    if(NOT DEFINED GENERATED_DIR)
        message(FATAL_ERROR "you must set the GENERATED_DIR variable to save the generated source file path" )
    endif()
    foreach(os_name linux;osx;windows)
        halide_generate_code_multi_os("${group_name}" "${func_name}" "${variable}" "${os_name}")
    endforeach()
endmacro()


