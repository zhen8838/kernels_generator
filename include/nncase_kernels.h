#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/datatypes.h>
#include <type_traits>

namespace Nncaseimpl
{

using nncase::bfloat16;
using nncase::padding;
using nncase::runtime_shape_t;
using nncase::value_range;

inline bfloat16 apply_gnne_activation(float value, bfloat16 x0, bfloat16 kl, bfloat16 bl, bfloat16 kr, bfloat16 br)
{
    return value < x0 ? bfloat16::round_to_bfloat16(value * kl + bl) : bfloat16::round_to_bfloat16(value * kr + br);
}

inline float apply_gnne_activation(float value, float x0, float kl, float bl, float kr, float br)
{
    return value < x0 ? value * kl + bl : value * kr + br;
}

inline void gnne_conv2d(bfloat16 *input, bfloat16 *output, bfloat16 *weights, float *psum, bfloat16 *act, runtime_shape_t &in_shape,
    int32_t groups, int32_t out_channels, int32_t filter_h, int32_t filter_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    padding &padding_h, padding &padding_w, value_range<bfloat16> fused_clamp, bool psum_is_uninitialized)
{
    const auto out_h = nncase::kernels::detail::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = nncase::kernels::detail::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
    const auto g_ic = in_shape[1] / groups;
    const auto g_oc = out_channels / groups;

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        const bfloat16 *in_batch_p = input + batch * in_shape[1] * in_shape[2] * in_shape[3];
        for (int32_t og = 0; og < groups; og++)
        {
            const bfloat16 *in_group_p = in_batch_p + og * g_ic * in_shape[2] * in_shape[3];
            const bfloat16 *w_group_p = weights + og * g_oc * g_ic * filter_h * filter_w;
            for (int32_t oc = 0; oc < g_oc; oc++)
            {
                const bfloat16 *w_oc_p = w_group_p + oc * g_ic * filter_h * filter_w;
                for (size_t oy = 0; oy < out_h; oy++)
                {
                    for (size_t ox = 0; ox < out_w; ox++)
                    {
                        const int32_t in_y_origin = (oy * stride_h) - padding_h.before;
                        const int32_t in_x_origin = (ox * stride_w) - padding_w.before;
                        const int32_t filter_y_start = std::max(0, (-in_y_origin + dilation_h - 1) / dilation_h);
                        const int32_t filter_y_end = std::min(filter_h, ((int32_t)in_shape[2] - in_y_origin + dilation_h - 1) / dilation_h);
                        const int32_t filter_x_start = std::max(0, (-in_x_origin + dilation_w - 1) / dilation_w);
                        const int32_t filter_x_end = std::min(filter_w, ((int32_t)in_shape[3] - in_x_origin + dilation_w - 1) / dilation_w);
                        float value = 0.f;

                        for (int32_t ic = 0; ic < (int32_t)g_ic; ic++)
                        {
                            const bfloat16 *in_c_p = in_group_p + ic * in_shape[2] * in_shape[3];
                            const bfloat16 *w_ic_p = w_oc_p + ic * filter_h * filter_w;
                            for (int32_t ky = filter_y_start; ky < filter_y_end; ky++)
                            {
                                for (int32_t kx = filter_x_start; kx < filter_x_end; kx++)
                                {
                                    const int32_t in_y = in_y_origin + dilation_h * ky;
                                    const int32_t in_x = in_x_origin + dilation_w * kx;
                                    const bfloat16 in_v = in_c_p[in_y * in_shape[3] + in_x];
                                    const bfloat16 w = w_ic_p[ky * filter_w + kx];
                                    value = value + (float)in_v * w;
                                }
                            }
                        }

                        int32_t base = og * g_oc + oc;
                        auto result = apply_gnne_activation(value, act[base * 5], act[base * 5 + 1], act[base * 5 + 2], act[base * 5 + 3], act[base * 5 + 4]);
                        *output = nncase::kernels::detail::apply_activation(result, fused_clamp);
                        if (!psum_is_uninitialized)
                            *output += bfloat16::round_to_bfloat16(*psum++);
                        output++;
                    }
                }
            }
        }
    }
}

inline void gnne_matmul(const bfloat16 *input_a, const bfloat16 *input_b, bfloat16 *output, const bfloat16 *act, int32_t a_rows, int32_t a_cols, int32_t b_cols, value_range<bfloat16> fused_activation)
{
    for (int32_t oy = 0; oy < a_rows; oy++)
    {
        for (int32_t ox = 0; ox < b_cols; ox++)
        {
            float value = 0.f;
            for (int32_t i = 0; i < a_cols; i++)
            {
                const auto a = input_a[oy * a_cols + i];
                const auto b = input_b[i * b_cols + ox];
                value = value + (float)a * b;
            }
            auto result = apply_gnne_activation(value, act[0], act[1], act[2], act[3], act[4]);
            output[oy * b_cols + ox] = nncase::kernels::detail::apply_activation(result, fused_activation);
        }
    }
}

inline void gnne_matmul(const float *input_a, const float *input_b, float *output, const float *act, int32_t a_rows, int32_t a_cols, int32_t b_cols, value_range<float> fused_activation)
{
    for (int32_t oy = 0; oy < a_rows; oy++)
    {
        for (int32_t ox = 0; ox < b_cols; ox++)
        {
            float value = 0.f;
            for (int32_t i = 0; i < a_cols; i++)
            {
                const auto a = input_a[oy * a_cols + i];
                const auto b = input_b[i * b_cols + ox];
                value = value + a * b;
            }
            auto result = apply_gnne_activation(value, act[0], act[1], act[2], act[3], act[4]);
            output[oy * b_cols + ox] = nncase::kernels::detail::apply_activation(result, fused_activation);
        }
    }
}

} // namespace Nncaseimpl
