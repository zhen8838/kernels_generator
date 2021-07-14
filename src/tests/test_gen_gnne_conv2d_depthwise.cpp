#include "hkg/benchmark/bench_gnne_conv2d_depthwise.h"
#include "hkg/export/halide_gnne_conv2d_depthwise.h"
#include <gtest/gtest.h>

auto Shape_Params = testing::Values(
    std::vector<size_t> { 1, 3, 6, 6 },
    std::vector<size_t> { 1, 32, 112, 112 },
    std::vector<size_t> { 1, 16, 112, 112 },
    std::vector<size_t> { 1, 96, 56, 56 },
    std::vector<size_t> { 1, 24, 56, 56 },
    std::vector<size_t> { 1, 144, 28, 28 },
    std::vector<size_t> { 1, 32, 28, 28 },
    std::vector<size_t> { 1, 192, 28, 28 });

auto Stride_Params = testing::Values(
    std::pair<int32_t, int32_t> { 1, 1 },
    std::pair<int32_t, int32_t> { 2, 2 });

auto Pad_Params = testing::Bool();

auto NoPsum_Params = testing::Bool();

class GNNEConv2DDepthWiseTestSuite : public testing::TestWithParam<
                                         std::tuple<std::vector<size_t>, // shape
                                             std::pair<int32_t, int32_t>, // stride hw
                                             bool, // same padding
                                             bool>>, // has no psum
                                     public GNNEConv2DDepthWiseParamBase
{
};

#define TEST_P_GNNE_CONV2D_DEPTHWISE(kh, kw)                                          \
    TEST_P(GNNEConv2DDepthWiseTestSuite, kernel_##kh##x##kw)                          \
    {                                                                                 \
        size_t err_count = 0;                                                         \
        auto [shape_param, stride_param, pad_param, no_psum_param] = GetParam();      \
        std::pair<int32_t, int32_t> filter_param { kh, kw };                          \
        set_param(filter_param, shape_param, stride_param, pad_param, no_psum_param); \
        run_float_reference();                                                        \
        run_reference();                                                              \
        for (size_t repeat = 0; repeat < repeat_times; repeat++)                      \
        {                                                                             \
            halide_gnne_conv2d_depthwise_##kh##x##kw(input.hbuf, weights.hbuf,        \
                psum.hbuf, act.hbuf, v_range.hbuf, no_psum.hbuf,                      \
                padding_h.before, padding_h.after,                                    \
                padding_w.before, padding_w.after,                                    \
                stride_h, stride_w, output.hbuf);                                     \
            err_count += check_error_with_float(output, 0.1f);                        \
        }                                                                             \
        ASSERT_EQ(err_count, 0);                                                      \
    }

TEST_P_GNNE_CONV2D_DEPTHWISE(1, 1)
TEST_P_GNNE_CONV2D_DEPTHWISE(3, 3)
TEST_P_GNNE_CONV2D_DEPTHWISE(5, 5)
TEST_P_GNNE_CONV2D_DEPTHWISE(7, 7)

INSTANTIATE_TEST_SUITE_P(GNNEConv2DDepthWiseTest, GNNEConv2DDepthWiseTestSuite, testing::Combine(Shape_Params, Stride_Params, Pad_Params, NoPsum_Params));
