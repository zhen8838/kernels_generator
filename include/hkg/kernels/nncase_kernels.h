#pragma once

#include "hkg/utils/nncase.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

inline void conv2d(float *input, float *weights, float *bias, float *output,
    runtime_shape_t &in_shape, runtime_shape_t &in_strides, runtime_shape_t &w_shape, runtime_shape_t &w_strides,
    runtime_shape_t &bias_strides, runtime_shape_t &out_strides, padding &padding_h, padding &padding_w,
    int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation) noexcept
{
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_channels = w_shape[0];
    const auto out_h = nncase::kernels::detail::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = nncase::kernels::detail::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
    const auto g_ic = in_shape[1] / groups;
    const auto g_oc = out_channels / groups;

    runtime_shape_t in_index(4);
    runtime_shape_t w_index(4);
    runtime_shape_t bias_index(1);
    runtime_shape_t out_index(4);
    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        in_index[0] = out_index[0] = batch;
        for (size_t og = 0; og < (size_t)groups; og++)
        {
            for (size_t oc = 0; oc < g_oc; oc++)
            {
                out_index[1] = w_index[0] = bias_index[0] = og * g_oc + oc;
                for (size_t oy = 0; oy < out_h; oy++)
                {
                    out_index[2] = oy;
                    for (size_t ox = 0; ox < out_w; ox++)
                    {
                        out_index[3] = ox;
                        const int32_t in_y_origin = (oy * stride_h) - padding_h.before;
                        const int32_t in_x_origin = (ox * stride_w) - padding_w.before;
                        const size_t filter_y_start = (size_t)std::max(0, (-in_y_origin + dilation_h - 1) / dilation_h);
                        const size_t filter_y_end = (size_t)std::min(filter_h, ((int32_t)in_shape[2] - in_y_origin + dilation_h - 1) / dilation_h);
                        const size_t filter_x_start = (size_t)std::max(0, (-in_x_origin + dilation_w - 1) / dilation_w);
                        const size_t filter_x_end = (size_t)std::min(filter_w, ((int32_t)in_shape[3] - in_x_origin + dilation_w - 1) / dilation_w);
                        float value = bias[nncase::kernels::offset(bias_strides, bias_index)];

                        for (size_t ic = 0; ic < g_ic; ic++)
                        {
                            in_index[1] = og * g_ic + ic;
                            w_index[1] = ic;
                            for (size_t ky = filter_y_start; ky < filter_y_end; ky++)
                            {
                                w_index[2] = ky;
                                for (size_t kx = filter_x_start; kx < filter_x_end; kx++)
                                {
                                    w_index[3] = kx;
                                    in_index[2] = in_y_origin + dilation_h * ky;
                                    in_index[3] = in_x_origin + dilation_w * kx;

                                    const float in_v = input[nncase::kernels::offset(in_strides, in_index)];
                                    const float w = weights[nncase::kernels::offset(w_strides, w_index)];

                                    value += in_v * w;
                                }
                            }
                        }
                        output[nncase::kernels::offset(out_strides, out_index)] = nncase::kernels::detail::apply_activation(value, fused_activation);
                    }
                }
            }
        }
    }
}

template <typename T>
inline void gnne_conv2d(T *input, T *output, T *weights, float *psum, T *act, runtime_shape_t &in_shape,
    int32_t groups, int32_t out_channels, int32_t filter_h, int32_t filter_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    padding &padding_h, padding &padding_w, value_range<T> fused_clamp, bool psum_is_uninitialized)
{
    const auto out_h = nncase::kernels::detail::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = nncase::kernels::detail::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
    const auto g_ic = in_shape[1] / groups;
    const auto g_oc = out_channels / groups;

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        const T *in_batch_p = input + batch * in_shape[1] * in_shape[2] * in_shape[3];
        for (int32_t og = 0; og < groups; og++)
        {
            const T *in_group_p = in_batch_p + og * g_ic * in_shape[2] * in_shape[3];
            const T *w_group_p = weights + og * g_oc * g_ic * filter_h * filter_w;
            for (int32_t oc = 0; oc < g_oc; oc++)
            {
                const T *w_oc_p = w_group_p + oc * g_ic * filter_h * filter_w;
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
                            const T *in_c_p = in_group_p + ic * in_shape[2] * in_shape[3];
                            const T *w_ic_p = w_oc_p + ic * filter_h * filter_w;
                            for (int32_t ky = filter_y_start; ky < filter_y_end; ky++)
                            {
                                for (int32_t kx = filter_x_start; kx < filter_x_end; kx++)
                                {
                                    const int32_t in_y = in_y_origin + dilation_h * ky;
                                    const int32_t in_x = in_x_origin + dilation_w * kx;
                                    const T in_v = in_c_p[in_y * in_shape[3] + in_x];
                                    const T w = w_ic_p[ky * filter_w + kx];
                                    value = value + (float)in_v * w;
                                }
                            }
                        }

                        int32_t base = og * g_oc + oc;
                        if (!psum_is_uninitialized)
                            value += *psum++;
                        T result = apply_gnne_activation(value, act[base * 5], act[base * 5 + 1], act[base * 5 + 2], act[base * 5 + 3], act[base * 5 + 4]);
                        *output = nncase::kernels::detail::apply_activation(result, fused_clamp);
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
