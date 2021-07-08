macro(halide_generate_runtime)
    if(NOT DEFINED RUNTIME_LIBS OR NOT DEFINED RUNTIME_Targets)
        message(FATAL_ERROR "you must set the RUNTIME_LIBS and RUNTIME_Targets variable to save the generated source file path" )
    endif()
    foreach(os_name linux;osx;windows)
        if(${os_name} STREQUAL "windows")
            set(suffix "lib")
        else()
            set(suffix "a")
        endif()

        add_custom_command(
            OUTPUT ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/halide_runtime_${os_name}.${suffix}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels_generator -r halide_runtime_${os_name} -o ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -e static_library,c_header target=${os_name}-x86-64
            DEPENDS kernels_generator
        )
        add_library(hkg_${os_name}_runtime INTERFACE)
        target_link_libraries(hkg_${os_name}_runtime INTERFACE
        $<BUILD_INTERFACE:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/halide_runtime_${os_name}.${suffix}> 
        $<INSTALL_INTERFACE:$\{_IMPORT_PREFIX\}/lib/halide_runtime_${os_name}.${suffix}> -ldl) # NOTE need find better method to export lib path.
        set_target_properties(hkg_${os_name}_runtime PROPERTIES EXPORT_NAME ${os_name}_runtime)
        add_library(hkg::${os_name}_runtime ALIAS hkg_${os_name}_runtime)

        list(APPEND RUNTIME_LIBS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/halide_runtime_${os_name}.${suffix})    
        list(APPEND RUNTIME_Targets hkg_${os_name}_runtime)    
    endforeach()
endmacro(halide_generate_runtime)

macro(halide_generate_code_multi_os group_name func_name variable os_name)
    set(FUNC_BASE_NAME halide_${func_name}_${os_name})
    set(OUTPUT_BASE_NAME  ${CMAKE_SOURCE_DIR}/include/hkg/${GENERATED_DIR}/${FUNC_BASE_NAME})
    set(TARGET_BASE_NAME  no_asserts-no_bounds_query-no_runtime-${os_name}-x86-64)
    set(OUTPUT_DIR  ${CMAKE_SOURCE_DIR}/include/hkg/${GENERATED_DIR})

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


macro(hkg_get_runtime_lib os_runtime os_name)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(cur_os_name "linux")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(cur_os_name "osx")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(cur_os_name "windows")    
    endif()
    set(${os_runtime} hkg::${cur_os_name}_runtime)    
    set(${os_name} ${cur_os_name})    
endmacro()

