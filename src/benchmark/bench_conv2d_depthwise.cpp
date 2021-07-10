#include "hkg/benchmark/bench_conv2d_depthwise.h"
#include "hkg/export/halide_conv2d_depthwise.h"
#include <celero/Celero.h>
#include <functional>
#include <map>

std::map<int32_t, conv2d_depthwise_func_t> Func_Map {
    { 1, halide_conv2d_depthwise_1x1 },
    { 3, halide_conv2d_depthwise_3x3 },
    { 5, halide_conv2d_depthwise_5x5 },
    { 7, halide_conv2d_depthwise_7x7 }
};

std::vector<std::vector<size_t>> Shape_Params = {
    { 1, 3, 32, 32 }, // B, IC, H, W, OC, ...
    { 1, 3, 224, 224 }, // B, IC, H, W, OC, ...
    { 1, 32, 112, 112 },
    { 1, 16, 112, 112 },
    { 1, 96, 56, 56 },
    { 1, 24, 56, 56 },
    { 1, 144, 28, 28 },
    { 1, 32, 28, 28 },
    { 1, 192, 28, 28 },
    { 1, 32, 28, 28 },
    { 1, 192, 14, 14 },
    { 1, 64, 14, 14 },
    { 1, 384, 14, 14 },
    { 1, 96, 14, 14 },
    { 1, 576, 14, 14 },
    { 1, 576, 7, 7 },
    { 1, 160, 7, 7 },
    { 1, 960, 7, 7 },
    { 1, 960, 7, 7 },
    { 1, 320, 7, 7 },
};

std::vector<std::pair<int32_t, int32_t>> Stride_Params = {
    { 1, 1 }, { 2, 2 }
};

std::vector<bool> Pad_Params = {
    true, false
};

auto make_product_conv2d_depthwise_params(std::vector<std::vector<size_t>> &shape_params, std::vector<std::pair<int32_t, int32_t>> &stride_params,
    std::vector<bool> &pad_params)
{
    std::vector<std::tuple<std::vector<size_t>, std::pair<int32_t, int32_t>, bool>> params;
    for (auto &&shape : shape_params)
    {
        for (auto &&stride : stride_params)
        {
            for (auto &&pad : pad_params)
            {
                params.emplace_back(std::make_tuple(shape, stride, pad));
            }
        }
    }
    return params;
}

auto experimentValue_Params = make_product_conv2d_depthwise_params(Shape_Params, Stride_Params, Pad_Params);

template <int32_t kh, int32_t kw>
class Conv2DDepthwiseBenchFixture : public celero::TestFixture,
                                    public Conv2DDepthWiseParamBase
{
public:
    std::vector<celero::TestFixture::ExperimentValue> getExperimentValues() const override
    {
        std::vector<celero::TestFixture::ExperimentValue> bufferSizes(experimentValue_Params.size());
        std::iota(bufferSizes.begin(), bufferSizes.end(), (size_t)0);
        return bufferSizes;
    }

    void setUp(const celero::TestFixture::ExperimentValue &experimentValue) override
    {
        auto [shape_param, stride_param, pad_param] = experimentValue_Params[experimentValue.Value];
        std::pair<int32_t, int32_t> filter_param { kh, kw };
        set_param(filter_param, shape_param, stride_param, pad_param, false);
    }

    void run_optimized()
    {
        Func_Map[kh](input.hbuf, weights.hbuf, bias.hbuf, v_range.hbuf,
            padding_h.before, padding_h.after,
            padding_w.before, padding_w.after,
            stride_h, stride_w, output.hbuf);
    }

    void tearDown() override
    {
    }
};

#define Conv2DDepthwiseBenchMark_hxw(kh, kw)                                                              \
                                                                                                          \
    using Conv2DDepthwiseBenchFixture_##kh##x##kw = Conv2DDepthwiseBenchFixture<kh, kw>;                  \
    BASELINE_F(Conv2DDepthwiseBenchMark_##kh##x##kw, opt, Conv2DDepthwiseBenchFixture_##kh##x##kw, 2, 5)  \
    {                                                                                                     \
        run_optimized();                                                                                  \
    }                                                                                                     \
                                                                                                          \
    BENCHMARK_F(Conv2DDepthwiseBenchMark_##kh##x##kw, ref, Conv2DDepthwiseBenchFixture_##kh##x##kw, 2, 5) \
    {                                                                                                     \
        run_reference();                                                                                  \
    }

// clang-format off
Conv2DDepthwiseBenchMark_hxw(1,1)
Conv2DDepthwiseBenchMark_hxw(3,3)
Conv2DDepthwiseBenchMark_hxw(5,5)
Conv2DDepthwiseBenchMark_hxw(7,7)
    // clang-format on

    CELERO_MAIN
