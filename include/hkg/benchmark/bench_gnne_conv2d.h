#pragma once
#include "hkg/kernels/nncase_kernels.h"
#include "hkg/utils/utils.h"
#include "nncase.h"
#include <celero/Celero.h>
#include <cmath>
#include <tuple>
#include <vector>

std::vector<std::vector<size_t>> GNNE_Conv2D_Test_Params = {
    { 1, 3, 32, 32, 4 }, // B, IC, H, W, OC, ...
    { 1, 3, 224, 224, 32 }, // B, IC, H, W, OC, ...
    { 1, 32, 112, 112, 16 },
    { 1, 16, 112, 112, 96 },
    { 1, 96, 56, 56, 24 },
    { 1, 24, 56, 56, 144 },
    { 1, 144, 28, 28, 32 },
    { 1, 32, 28, 28, 192 },
    { 1, 192, 28, 28, 32 },
    { 1, 32, 28, 28, 192 },
    { 1, 192, 14, 14, 64 },
    { 1, 64, 14, 14, 384 },
    { 1, 384, 14, 14, 64 },
    { 1, 96, 14, 14, 576 },
    { 1, 576, 14, 14, 96 },
    { 1, 576, 7, 7, 160 },
    { 1, 160, 7, 7, 960 },
    { 1, 960, 7, 7, 160 },
    { 1, 960, 7, 7, 320 },
    { 1, 320, 7, 7, 1280 },
    // { 1, 1280, 1, 1, 1001 }
};

using NNCASE_TYPE_t = nncase::bfloat16;

template <int32_t KernelHeight, int32_t KernelWidth, int32_t StrideHeight, int32_t StrideWidth, bool PadSame, bool NoPsum = false>
class GnneConv2DTestFixture : public celero::TestFixture
{
public:
    Tensor_t<NNCASE_TYPE_t, NNCASE_TYPE_t> input, weights, output, act, v_range;
    Tensor_t<float, float> psum;
    Scalar_t<bool, bool> no_psum;
    nncase::runtime_shape_t in_shape;
    int32_t groups, out_channels, filter_h, filter_w, stride_h, stride_w, dilation_h, dilation_w;
    nncase::padding padding_h, padding_w;

    std::vector<celero::TestFixture::ExperimentValue> getExperimentValues() const override
    {
        std::vector<celero::TestFixture::ExperimentValue> bufferSizes(GNNE_Conv2D_Test_Params.size());
        std::iota(bufferSizes.begin(), bufferSizes.end(), (size_t)0);
        return bufferSizes;
    }

    template <typename T1, typename T2>
    static auto get_data(size_t batch, size_t in_channels, size_t height, size_t width, size_t out_channels, size_t k_height, size_t k_width, size_t stride_h, size_t stride_w, bool pad_same, bool no_psum, init_method method = init_method::rand)
    {
        Tensor_t<T1, T2> input_({ batch, in_channels, height, width }, "input");
        Tensor_t<T1, T2> weights_({ out_channels, in_channels, k_height, k_width }, "weight");
        size_t out_h, out_w;
        nncase::padding padding_h_ = nncase::padding::zero(), padding_w_ = nncase::padding::zero();
        if (pad_same)
        {
            out_h = int(ceilf(float(height) / float(k_height)));
            out_w = int(ceilf(float(width) / float(k_width)));
            size_t pad_height = std::max(k_height - (height % k_height ? (height % k_height) : k_height), size_t(0));
            size_t pad_width = std::max(k_width - (width % k_width ? (width % k_width) : k_width), size_t(0));
            padding_h_.before = pad_height / 2;
            padding_h_.after = pad_height - padding_h_.before;

            padding_w_.before = pad_width / 2;
            padding_w_.after = pad_width - padding_w_.before;
        }

        out_h = nncase::kernels::detail::get_windowed_output_size(height, k_height, stride_h, 1, padding_h_);
        out_w = nncase::kernels::detail::get_windowed_output_size(width, k_width, stride_w, 1, padding_w_);

        Tensor_t<T1, T2> output_({ batch, out_channels, out_h, out_w }, "output");
        Tensor_t<float, float> psum_({ batch, out_channels, out_h, out_w }, "psum");
        Tensor_t<T1, T2> act_({ out_channels, 5 }, "act");
        Tensor_t<T1, T2> v_range_({ 2 }, "value_range");
        Scalar_t<bool, bool> no_psum_(no_psum);

        input_.allocate(method);
        weights_.allocate(method);
        output_.allocate(init_method::zero);
        psum_.allocate(method);
        act_.allocate(method);
        v_range_.allocate();
        v_range_.as_value_range(range_t::full);

        return std::make_tuple(
            input_, weights_, output_,
            psum_, act_, padding_h_, padding_w_,
            v_range_, no_psum_);
    }

    void setUp(const celero::TestFixture::ExperimentValue &experimentValue) override
    {
        auto param = GNNE_Conv2D_Test_Params[experimentValue.Value];
        auto B = param[0], IC = param[1], H = param[2], W = param[3], OC = param[4];
        groups = 1;
        filter_h = KernelHeight;
        filter_w = KernelWidth;
        stride_h = StrideHeight;
        stride_w = StrideWidth;
        dilation_h = 1;
        dilation_w = 1;
        std::tie(input, weights, output, psum, act, padding_h, padding_w, v_range, no_psum) = get_data<NNCASE_TYPE_t, NNCASE_TYPE_t>(B, IC, H, W, OC, filter_h, filter_w, stride_h, stride_w, PadSame, NoPsum);
        in_shape = nncase::runtime_shape_t({ B, IC, H, W });
        out_channels = OC;
    }

    void tearDown() override
    {
    }
};

#define BENCHMARK_GNNE_CONV2D_CASE(kh, kw, sh, sw, padsame) BENCHMARK_GNNE_CONV2D_CASE_IMPL(kh, kw, sh, sw, padsame, false, 5, 1)

#define BENCHMARK_GNNE_CONV2D_CASE_IMPL(kh, kw, sh, sw, padsame, nopsum, sample, iteration)                                                \
                                                                                                                                           \
    using GnneConv2DTestFixture_##kh##x##kw##_##sh##x##sw##_##padsame##_##nopsum = GnneConv2DTestFixture<kh, kw, sh, sw, padsame, nopsum>; \
                                                                                                                                           \
    BENCHMARK_F(_##kh##x##kw##_##sh##x##sw##_##padsame##_##nopsum, ref,                                                                    \
        GnneConv2DTestFixture_##kh##x##kw##_##sh##x##sw##_##padsame##_##nopsum, sample, iteration)                                         \
    {                                                                                                                                      \
        Nncaseimpl::gnne_conv2d(input.nraw, output.nraw, weights.nraw, psum.nraw, act.nraw,                                                \
            in_shape, groups, out_channels, filter_h, filter_w, stride_h, stride_w,                                                        \
            dilation_h, dilation_w, padding_h, padding_w,                                                                                  \
            nncase::value_range<NNCASE_TYPE_t> { v_range.nraw[0], v_range.nraw[1] },                                                       \
            no_psum.nraw);                                                                                                                 \
    }                                                                                                                                      \
                                                                                                                                           \
    BASELINE_F(_##kh##x##kw##_##sh##x##sw##_##padsame##_##nopsum, opt,                                                                     \
        GnneConv2DTestFixture_##kh##x##kw##_##sh##x##sw##_##padsame##_##nopsum, sample, iteration)                                         \
    {                                                                                                                                      \
        halide_gnne_conv2d_##kh##x##kw(input.hbuf, weights.hbuf, psum.hbuf, act.hbuf, v_range.hbuf,                                        \
            no_psum.hbuf, padding_h.before, padding_h.after,                                                                               \
            padding_w.before, padding_w.after, stride_h, stride_w, output.hbuf);                                                           \
    }
