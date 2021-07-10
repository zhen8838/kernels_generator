#pragma once
#include "hkg/kernels/nncase_kernels.h"
#include "hkg/utils/utils.h"
#include <cmath>
#include <tuple>
#include <vector>

using NNCASE_TYPE_t = float;

template <typename T1, typename T2>
auto get_data(size_t batch, size_t channels, size_t height, size_t width, size_t k_height, size_t k_width, size_t stride_h, size_t stride_w, bool pad_same, init_method method = init_method::rand)
{
    Tensor_t<T1, T2> input({ batch, channels, height, width }, "input");
    Tensor_t<T1, T2> weights({ channels, 1, k_height, k_width }, "weight");
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

    Tensor_t<T1, T2> output({ batch, channels, out_h, out_w }, "output");
    Tensor_t<T1, T2> bias({ channels }, "act");
    Tensor_t<T1, T2> v_range({ 2 }, "value_range");

    input.allocate(method);
    weights.allocate(method);
    output.allocate(init_method::zero);
    if (method == init_method::sequence)
        bias.allocate(init_method::zero);
    else
        bias.allocate(method);
    v_range.allocate();
    v_range.as_value_range(range_t::full);

    return std::make_tuple(std::move(input), std::move(weights), std::move(output), std::move(bias), std::move(v_range), padding_h, padding_w);
}

class Conv2DDepthWiseParamBase
{
public:
    Tensor_t<NNCASE_TYPE_t, NNCASE_TYPE_t> input, weights, output, bias, v_range;
    nncase::runtime_shape_t in_shape, in_strides, w_shape, w_strides, bias_strides, output_strides;

    int32_t groups, out_channels, filter_h, filter_w, stride_h, stride_w, dilation_h, dilation_w;
    nncase::padding padding_h, padding_w;

    size_t total_count = 0, repeat_times = 2;

    void set_param(std::pair<int32_t, int32_t> &filter_hw, std::vector<size_t> &shape_param, std::pair<int32_t, int32_t> &stride_hw, bool PadSame, bool is_print = true)
    {
        auto B = shape_param[0], C = shape_param[1], H = shape_param[2], W = shape_param[3];
        groups = C;
        filter_h = filter_hw.first;
        filter_w = filter_hw.second;
        stride_h = stride_hw.first;
        stride_w = stride_hw.second;
        dilation_h = 1;
        dilation_w = 1;
#ifdef DEBUG
        init_method method = init_method::sequence;
#else
        init_method method = init_method::rand;
#endif
        std::tie(input, weights, output, bias, v_range, padding_h, padding_w) = get_data<NNCASE_TYPE_t, NNCASE_TYPE_t>(B, C, H, W, filter_h, filter_w, stride_h, stride_w, PadSame, method);
        in_shape = nncase::runtime_shape_t({ B, C, H, W });
        in_strides = nncase::runtime::get_default_strides(in_shape);
        w_shape = weights.shape;
        w_strides = nncase::runtime::get_default_strides(w_shape);
        bias_strides = nncase::runtime::get_default_strides(bias.shape);
        output_strides = nncase::runtime::get_default_strides(output.shape);
        out_channels = C;
        if (is_print)
            printf("In : [%ld, %ld, %ld, %ld], S: [%d x %d]\n", B, C, H, W, stride_h, stride_w);
    }
    size_t check_error()
    {
        size_t err_count = 0;
        err_count += compare_blob(output.nraw, output.hraw, output.shape, 0.01f);
        total_count += nncase::runtime::compute_size(output.shape);

        std::cout << "err :" << err_count << ", total :" << total_count << std::endl;
        return err_count;
    }

    void run_reference()
    {
        Nncaseimpl::conv2d(input.nraw, weights.nraw, bias.nraw, output.nraw,
            in_shape, in_strides, w_shape, w_strides, bias_strides, output_strides,
            padding_h, padding_w,
            groups, stride_h, stride_w, dilation_h, dilation_w,
            nncase::value_range<NNCASE_TYPE_t> { v_range.nraw[0], v_range.nraw[1] });
    }
};