#include "benchmark/bench_conv2d.h"
#include "hkg/halide_conv2d.h"
// #include "hkg/halide_conv2d_3x3.h"
// #include "hkg/halide_conv2d_5x5.h"
// #include "hkg/halide_conv2d_7x7.h"
#include <gtest/gtest.h>

auto Shape_Params = testing::Values(
    std::vector<size_t> { 1, 32, 112, 112, 16 },
    std::vector<size_t> { 1, 16, 112, 112, 96 },
    std::vector<size_t> { 1, 96, 56, 56, 24 },
    std::vector<size_t> { 1, 24, 56, 56, 144 },
    std::vector<size_t> { 1, 144, 28, 28, 32 },
    std::vector<size_t> { 1, 32, 28, 28, 192 },
    std::vector<size_t> { 1, 192, 28, 28, 32 });

auto Stride_Params = testing::Values(
    std::pair<int32_t, int32_t> { 1, 1 },
    std::pair<int32_t, int32_t> { 2, 2 });

auto Pad_Params = testing::Bool();

class Conv2DTestSuite : public testing::TestWithParam<
                            std::tuple<std::vector<size_t>, // shape
                                std::pair<int32_t, int32_t>, // stride hw
                                bool>> // same padding
{
public:
    Tensor_t<NNCASE_TYPE_t, NNCASE_TYPE_t> input, weights, output, bias, v_range;
    nncase::runtime_shape_t in_shape, in_strides, w_shape, w_strides, bias_strides, output_strides;

    int32_t groups, out_channels, filter_h, filter_w, stride_h, stride_w, dilation_h, dilation_w;
    nncase::padding padding_h, padding_w;

    size_t total_count = 0, repeat_times = 2;

    void set_param(std::pair<int32_t, int32_t> filter_hw)
    {
        auto [shape_param, stride_hw, PadSame] = GetParam();
        auto B = shape_param[0], IC = shape_param[1], H = shape_param[2], W = shape_param[3], OC = shape_param[4];
        groups = 1;
        filter_h = filter_hw.first;
        filter_w = filter_hw.second;
        stride_h = stride_hw.first;
        stride_w = stride_hw.second;
        dilation_h = 1;
        dilation_w = 1;
        std::tie(input, weights, output, bias, v_range, padding_h, padding_w) = get_data<NNCASE_TYPE_t, NNCASE_TYPE_t>(B, IC, H, W, OC, filter_h, filter_w, stride_h, stride_w, PadSame);
        in_shape = nncase::runtime_shape_t({ B, IC, H, W });
        in_strides = nncase::runtime::get_default_strides(in_shape);
        w_shape = weights.shape;
        w_strides = nncase::runtime::get_default_strides(w_shape);
        bias_strides = nncase::runtime::get_default_strides(bias.shape);
        output_strides = nncase::runtime::get_default_strides(output.shape);
        out_channels = OC;
        printf("In : [%ld, %ld, %ld, %ld], OC: %ld, S: [%d x %d]\n", B, IC, H, W, OC, stride_h, stride_w);
    }

    size_t check_error()
    {
        size_t err_count = 0;
        err_count += compare_blob(output.nraw, output.hraw, output.shape, 0.01f);
        total_count += compute_size(output.shape);

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

#define TEST_P_CONV2D(ks, kw)                                                            \
    TEST_P(Conv2DTestSuite, kernel_##ks##x##kw)                                          \
    {                                                                                    \
        size_t err_count = 0;                                                            \
        set_param({ ks, kw });                                                           \
        run_reference();                                                                 \
        for (size_t repeat = 0; repeat < repeat_times; repeat++)                         \
        {                                                                                \
            halide_conv2d_##ks##x##kw(input.hbuf, weights.hbuf, bias.hbuf, v_range.hbuf, \
                padding_h.before, padding_h.after,                                       \
                padding_w.before, padding_w.after,                                       \
                stride_h, stride_w, output.hbuf);                                        \
            err_count += check_error();                                                  \
        }                                                                                \
        ASSERT_EQ(err_count, 0);                                                         \
    }

TEST_P_CONV2D(1, 1)
// TEST_P_CONV2D(3, 3)
// TEST_P_CONV2D(5, 5)
// TEST_P_CONV2D(7, 7)

INSTANTIATE_TEST_SUITE_P(Conv2DTest, Conv2DTestSuite, testing::Combine(Shape_Params, Stride_Params, Pad_Params));
