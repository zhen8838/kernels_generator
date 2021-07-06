#include "generated_kernels/halide_conv2d_1x1_linux_avx2.h"
#include "generated_kernels/halide_conv2d_1x1_linux_bare.h"
#include "generated_kernels/halide_conv2d_1x1_linux_sse41.h"
#include "target.h"

#ifdef __linux__
void halide_conv2d_1x1(struct halide_buffer_t *_input_buffer, struct halide_buffer_t *_weights_buffer,
    struct halide_buffer_t *_bias_buffer, struct halide_buffer_t *_value_range_buffer,
    int32_t _pad_h_before, int32_t _pad_h_end, int32_t _pad_w_before, int32_t _pad_w_end,
    int32_t _stride_h, int32_t _stride_w, struct halide_buffer_t *_Clamped_buffer)
{
    int (*func)(struct halide_buffer_t *, struct halide_buffer_t *,
        struct halide_buffer_t *, struct halide_buffer_t *,
        int32_t, int32_t, int32_t, int32_t,
        int32_t, int32_t, struct halide_buffer_t *)
        = nullptr;
    if (internal::host_target.feature_avx2)
    {
        func = &halide_conv2d_1x1_linux_avx2;
    }
    else if (internal::host_target.feature_sse41)
    {
        func = &halide_conv2d_1x1_linux_sse41;
    }
    else
    {
        func = &halide_conv2d_1x1_linux_bare;
    }
    (*func)(_input_buffer, _weights_buffer, _bias_buffer, _value_range_buffer,
        _pad_h_before, _pad_h_end, _pad_w_before, _pad_w_end,
        _stride_h, _stride_w, _Clamped_buffer);
}

#endif

#ifdef _WIN32

#endif

#ifdef __APPLE__

#endif
