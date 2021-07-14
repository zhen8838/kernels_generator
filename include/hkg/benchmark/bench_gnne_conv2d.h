#pragma once
#include "hkg/kernels/nncase_kernels.h"
#include "hkg/utils/utils.h"
#include <cmath>
#include <tuple>
#include <vector>

using NNCASE_TYPE_t = nncase::bfloat16;

template <typename T1, typename T2>
static auto get_data(size_t batch, size_t in_channels, size_t height, size_t width, size_t out_channels, size_t k_height, size_t k_width, size_t stride_h, size_t stride_w, bool pad_same, bool no_psum_, init_method method = init_method::rand)
{
    Tensor_t<T1, T2> input({ batch, in_channels, height, width }, "input");
    Tensor_t<T1, T2> weights({ out_channels, in_channels, k_height, k_width }, "weight");
    size_t out_h, out_w;
    nncase::padding padding_h = nncase::padding::zero(), padding_w = nncase::padding::zero();
    if (pad_same)
    {
        out_h = int(ceilf(float(height) / float(k_height)));
        out_w = int(ceilf(float(width) / float(k_width)));
        size_t pad_height = std::max(k_height - (height % k_height ? (height % k_height) : k_height), size_t(0));
        size_t pad_width = std::max(k_width - (width % k_width ? (width % k_width) : k_width), size_t(0));
        padding_h.before = pad_height / 2;
        padding_h.after = pad_height - padding_h.before;

        padding_w.before = pad_width / 2;
        padding_w.after = pad_width - padding_w.before;
    }

    out_h = nncase::kernels::detail::get_windowed_output_size(height, k_height, stride_h, 1, padding_h);
    out_w = nncase::kernels::detail::get_windowed_output_size(width, k_width, stride_w, 1, padding_w);

    Tensor_t<T1, T2> output({ batch, out_channels, out_h, out_w }, "output");
    Tensor_t<float, float> psum({ batch, out_channels, out_h, out_w }, "psum");
    Tensor_t<T1, T2> act({ out_channels, 5 }, "act");
    Tensor_t<T1, T2> v_range({ 2 }, "value_range");
    Scalar_t<bool, bool> no_psum(no_psum_);

    input.allocate(method);
    weights.allocate(method);
    output.allocate(init_method::zero);
    psum.allocate(method);
    act.allocate(method);
    v_range.allocate();
    v_range.as_value_range(range_t::full);

    return std::make_tuple(std::move(input), std::move(weights),
        std::move(output), std::move(psum), std::move(act),
        padding_h, padding_w, std::move(v_range), std::move(no_psum));
}

class GNNEConv2DParamBase
{
public:
    Tensor_t<NNCASE_TYPE_t, NNCASE_TYPE_t> input, weights, output, act, v_range;
    Tensor_t<float, float> psum;
    Scalar_t<bool, bool> no_psum;
    nncase::runtime_shape_t in_shape;
    int32_t groups, out_channels, filter_h, filter_w, stride_h, stride_w, dilation_h, dilation_w;
    nncase::padding padding_h, padding_w;
    size_t repeat_times = 2;

    void set_param(std::pair<int32_t, int32_t> &filter_hw, std::vector<size_t> &shape_param, std::pair<int32_t, int32_t> &stride_hw, bool pad_same, bool no_psum_, bool is_print = true)
    {
        auto B = shape_param[0], IC = shape_param[1], H = shape_param[2], W = shape_param[3], OC = shape_param[4];
        groups = 1;
        filter_h = filter_hw.first;
        filter_w = filter_hw.second;
        stride_h = stride_hw.first;
        stride_w = stride_hw.second;
        dilation_h = 1;
        dilation_w = 1;
        std::tie(input, weights, output, psum, act, padding_h, padding_w, v_range, no_psum) = get_data<NNCASE_TYPE_t, NNCASE_TYPE_t>(B, IC, H, W, OC, filter_h, filter_w, stride_h, stride_w, pad_same, no_psum_);
        in_shape = nncase::runtime_shape_t({ B, IC, H, W });
        out_channels = OC;
        if (is_print)
            printf("In : [%ld, %ld, %ld, %ld], OC: %ld, S: [%d x %d]\n", B, IC, H, W, OC, stride_h, stride_w);
    }

    void run_reference()
    {
        Nncaseimpl::gnne_conv2d(input.nraw, output.nraw,
            weights.nraw, psum.nraw, act.nraw,
            in_shape, groups, out_channels,
            filter_h, filter_w, stride_h, stride_w,
            dilation_h, dilation_w, padding_h, padding_w,
            nncase::value_range<NNCASE_TYPE_t> { v_range.nraw[0], v_range.nraw[1] }, no_psum.nraw);
    }

    void run_float_reference()
    {
        Nncaseimpl::gnne_conv2d(input.raw, output.raw,
            weights.raw, psum.raw, act.raw,
            in_shape, groups, out_channels,
            filter_h, filter_w, stride_h, stride_w,
            dilation_h, dilation_w, padding_h, padding_w,
            nncase::value_range<float> { v_range.raw[0], v_range.raw[1] }, no_psum.raw);
    }
};