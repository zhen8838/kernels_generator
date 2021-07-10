
set(LINUX_SRCS "")
set(OSX_SRCS "")
set(WIN_SRCS "")

# -------------------- conv2d --------------------
message(STATUS "Configure Conv2d Kernels")
set(conv2d_all_linux_srcs "")
set(conv2d_all_osx_srcs "")
set(conv2d_all_windows_srcs "")
set(conv2d_all_linux_header "")
set(conv2d_all_osx_header "")
set(conv2d_all_windows_header "")
foreach(WH "1;1" "3;3" "5;5" "7;7")
    list(GET WH 0 KH)
    list(GET WH -1 KW)
    halide_generate_code(conv2d "conv2d_${KH}x${KW}" "kernel_height=${KH};kernel_width=${KW}" 
        conv2d_linux_srcs
        conv2d_osx_srcs
        conv2d_windows_srcs
        conv2d_linux_header
        conv2d_osx_header
        conv2d_windows_header)
    list(APPEND conv2d_all_linux_srcs ${conv2d_linux_srcs})
    list(APPEND conv2d_all_osx_srcs ${conv2d_osx_srcs})
    list(APPEND conv2d_all_windows_srcs ${conv2d_windows_srcs})
    list(APPEND conv2d_all_linux_header ${conv2d_linux_header})
    list(APPEND conv2d_all_osx_header ${conv2d_osx_header})
    list(APPEND conv2d_all_windows_header ${conv2d_windows_header})
endforeach()

insert_header(conv2d "${conv2d_all_linux_header}" "${conv2d_all_osx_header}" "${conv2d_all_windows_header}")  

list(APPEND LINUX_SRCS ${conv2d_all_linux_srcs})
list(APPEND OSX_SRCS ${conv2d_all_osx_srcs})
list(APPEND WIN_SRCS ${conv2d_all_windows_srcs})



# all

list(APPEND KERNEL_SRCS "${LINUX_SRCS};${OSX_SRCS};${WIN_SRCS}")
