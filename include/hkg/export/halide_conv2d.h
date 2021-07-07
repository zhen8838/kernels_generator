#ifdef __linux__
#include "hkg/generated_kernels/halide_conv2d_1x1_linux_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_1x1_linux_bare.h"
#include "hkg/generated_kernels/halide_conv2d_1x1_linux_sse41.h"

#include "hkg/generated_kernels/halide_conv2d_3x3_linux_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_3x3_linux_bare.h"
#include "hkg/generated_kernels/halide_conv2d_3x3_linux_sse41.h"

#include "hkg/generated_kernels/halide_conv2d_5x5_linux_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_5x5_linux_bare.h"
#include "hkg/generated_kernels/halide_conv2d_5x5_linux_sse41.h"

#include "hkg/generated_kernels/halide_conv2d_7x7_linux_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_7x7_linux_bare.h"
#include "hkg/generated_kernels/halide_conv2d_7x7_linux_sse41.h"
#elif _WIN32
#include "hkg/generated_kernels/halide_conv2d_1x1_windows_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_1x1_windows_bare.h"
#include "hkg/generated_kernels/halide_conv2d_1x1_windows_sse41.h"

#include "hkg/generated_kernels/halide_conv2d_3x3_windows_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_3x3_windows_bare.h"
#include "hkg/generated_kernels/halide_conv2d_3x3_windows_sse41.h"

#include "hkg/generated_kernels/halide_conv2d_5x5_windows_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_5x5_windows_bare.h"
#include "hkg/generated_kernels/halide_conv2d_5x5_windows_sse41.h"

#include "hkg/generated_kernels/halide_conv2d_7x7_windows_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_7x7_windows_bare.h"
#include "hkg/generated_kernels/halide_conv2d_7x7_windows_sse41.h"
#elif __APPLE__
#include "hkg/generated_kernels/halide_conv2d_1x1_osx_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_1x1_osx_bare.h"
#include "hkg/generated_kernels/halide_conv2d_1x1_osx_sse41.h"

#include "hkg/generated_kernels/halide_conv2d_3x3_osx_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_3x3_osx_bare.h"
#include "hkg/generated_kernels/halide_conv2d_3x3_osx_sse41.h"

#include "hkg/generated_kernels/halide_conv2d_5x5_osx_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_5x5_osx_bare.h"
#include "hkg/generated_kernels/halide_conv2d_5x5_osx_sse41.h"

#include "hkg/generated_kernels/halide_conv2d_7x7_osx_avx2.h"
#include "hkg/generated_kernels/halide_conv2d_7x7_osx_bare.h"
#include "hkg/generated_kernels/halide_conv2d_7x7_osx_sse41.h"
#endif

#include "target.h"
#include <functional>

#define halide_conv2d_hxw_os(kh, kw, os)                 \
    if (internal::host_target.feature_avx2)              \
    {                                                    \
        func = halide_conv2d_##kh##x##kw##_##os##_avx2;  \
    }                                                    \
    else if (internal::host_target.feature_sse41)        \
    {                                                    \
        func = halide_conv2d_##kh##x##kw##_##os##_sse41; \
    }                                                    \
    else                                                 \
    {                                                    \
        func = halide_conv2d_##kh##x##kw##_##os##_bare;  \
    }

#ifdef __linux__
#define select_halide_conv2d_hxw(kh, kw) halide_conv2d_hxw_os(kh, kw, linux)
#elif _WIN32
#define select_halide_conv2d_hxw(kh, kw) halide_conv2d_hxw_os(kh, kw, windows)
#elif __APPLE__
#define select_halide_conv2d_hxw(kh, kw) halide_conv2d_hxw_os(kh, kw, osx)
#endif

using conv2d_func_t = std::function<int(struct halide_buffer_t *_input_buffer, struct halide_buffer_t *_weights_buffer,
    struct halide_buffer_t *_bias_buffer, struct halide_buffer_t *_value_range_buffer,
    int32_t _pad_h_before, int32_t _pad_h_end, int32_t _pad_w_before, int32_t _pad_w_end,
    int32_t _stride_h, int32_t _stride_w, struct halide_buffer_t *_Clamped_buffer)>;

#define get_halide_conv2d_func(kh, kw)                \
    auto get_halide_conv2d_##kh##x##kw()              \
    {                                                 \
        conv2d_func_t func = nullptr;                 \
        select_halide_conv2d_hxw(kh, kw) return func; \
    }

#define halide_conv2d_hxw(kh, kw)               \
    get_halide_conv2d_func(kh, kw)              \
        conv2d_func_t halide_conv2d_##kh##x##kw \
        = get_halide_conv2d_##kh##x##kw();

// clang-format off
halide_conv2d_hxw(1,1)
halide_conv2d_hxw(3,3)
halide_conv2d_hxw(5,5)
halide_conv2d_hxw(7,7)
    // clang-format on
