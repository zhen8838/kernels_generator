#include "benchmark/bench_gnne_conv2d.h"
#include "benchmark/bench_gnne_matmul.h"
#include "hkg/halide_gnne_conv2d_1x1.h"
#include "hkg/halide_gnne_conv2d_3x3.h"
#include "hkg/halide_gnne_conv2d_5x5.h"
#include "hkg/halide_gnne_conv2d_7x7.h"
#include "hkg/halide_gnne_matmul.h"
#include <gtest/gtest.h>

template <typename T1, typename T2>
void print_blob(Tensor_t<T1, T2> &t)
{
    std::cout << t.name << std::endl;
    print_blob(t.nraw, t.shape);
    print_blob(t.hraw, t.shape);
}

TEST(test, gnne_matmul)
{
    size_t err_count = 0, total_count = 0;
    std::vector<std::vector<size_t>> test_params = {
        { 3, 5, 4 }, // M , N , K
        { 10, 20, 10 }, // M , N , K
        { 16, 32, 64 }, // M , N , K
        { 256, 128, 64 }, // M , N , K
        { 384, 256, 96 }, // M , N , K
        { 512, 512, 512 }, // M , N , K
    };
    for (auto param : test_params)
    {
        auto M = param[0];
        auto K = param[1];
        auto N = param[2];
        auto &&[input_a, input_b, act, output, v_range] = GnneMatmulTestFixture::get_data<NNCASE_TYPE_t, NNCASE_TYPE_t>(M, N, K);
        Nncaseimpl::gnne_matmul(input_a.raw, input_b.raw, output.raw, act.raw, input_a.shape[0], input_a.shape[1], input_b.shape[1], nncase::value_range<float> { v_range.raw[0], v_range.raw[1] });

        Nncaseimpl::gnne_matmul(input_a.nraw, input_b.nraw, output.nraw, act.nraw, input_a.shape[0], input_a.shape[1], input_b.shape[1], nncase::value_range<NNCASE_TYPE_t> { v_range.nraw[0], v_range.nraw[1] });

        halide_gnne_matmul(input_a.hbuf, input_b.hbuf, act.hbuf, v_range.hbuf, output.hbuf);

        // err_count += compare_blob(output.raw, output.nraw, output.hraw, output.shape);
        err_count += compare_blob(output.nraw, output.hraw, output.shape);
        total_count += compute_size(output.shape);
    }
    std::cout << err_count << " " << total_count << std::endl;
    ASSERT_EQ(err_count, 0);
}

TEST(test, gnne_conv2d)
{
    size_t index = 0, err_count = 0, total_count = 0;
    std::vector<std::vector<size_t>> test_params = {
        // { 1, 3, 4, 4, 2 },
        // TODO input 16的情况下1x1 s2卷积会挂，得检查一下什么情况
        { 1, 1, 16, 16, 1 }, // B, IC, H, W, OC, ...
        { 1, 1, 16, 16, 16 }, // B, IC, H, W, OC, ...
        { 1, 16, 16, 16, 1 }, // B, IC, H, W, OC, ...
        { 1, 16, 16, 16, 16 }, // B, IC, H, W, OC, ...
        { 3, 1, 16, 16, 1 }, // B, IC, H, W, OC, ...
        { 3, 1, 16, 16, 16 }, // B, IC, H, W, OC, ...
        { 3, 16, 16, 16, 1 }, // B, IC, H, W, OC, ...
        { 3, 16, 16, 16, 16 }, // B, IC, H, W, OC, ...
        // { 1, 3, 224, 224, 32 }, // B, IC, H, W, OC, ...
        // { 1, 32, 112, 112, 16 },
        // { 1, 16, 112, 112, 96 },
        // { 1, 96, 56, 56, 24 },

    };
    for (auto &&param : test_params)
    {
        auto B = param[0], IC = param[1], H = param[2], W = param[3], OC = param[4];
        for (auto &&[filter_h, filter_w] : std::vector<std::tuple<int32_t, int32_t>> { { 1, 1 }, { 3, 3 }, { 5, 5 }, { 7, 7 } })
        {
            for (auto &&[stride_h, stride_w] : std::vector<std::tuple<int32_t, int32_t>> { { 1, 1 }, { 2, 2 } })
            {
                for (auto &&padsame : std::vector<bool> { true, false })
                {
                    if (filter_h > H and filter_w > W)
                    {
                        continue;
                    }
                    auto &&[input, weights, output, psum, act, padding_h, padding_w, v_range, no_psum] = GnneConv2DTestFixture<1, 1, 1, 1, false>::get_data<NNCASE_TYPE_t, NNCASE_TYPE_t>(B, IC, H, W, OC, filter_h, filter_w, stride_h, stride_w, padsame, true);
                    nncase::runtime_shape_t in_shape { B, IC, H, W };
                    int32_t out_channels = OC;
                    int32_t groups = 1;
                    int32_t dilation_h = 1;
                    int32_t dilation_w = 1;
                    // run twice
                    for (size_t repeat = 0; repeat < 3; repeat++)
                    {

                        Nncaseimpl::gnne_conv2d(input.nraw, output.nraw, weights.nraw, psum.nraw, act.nraw, in_shape, groups, out_channels, filter_h, filter_w, stride_h, stride_w, dilation_h, dilation_w, padding_h,
                            padding_w, nncase::value_range<NNCASE_TYPE_t> { v_range.nraw[0], v_range.nraw[1] },
                            no_psum.nraw);

                        if (filter_h == 1 and filter_w == 1)
                        {
                            halide_gnne_conv2d_1x1(input.hbuf, weights.hbuf,
                                psum.hbuf, act.hbuf, v_range.hbuf, no_psum.hbuf, padding_h.before, padding_h.after,
                                padding_w.before, padding_w.after, stride_h, stride_w, output.hbuf);
                        }
                        else if (filter_h == 3 and filter_w == 3)
                        {
                            halide_gnne_conv2d_3x3(input.hbuf, weights.hbuf,
                                psum.hbuf, act.hbuf, v_range.hbuf, no_psum.hbuf, padding_h.before, padding_h.after,
                                padding_w.before, padding_w.after, stride_h, stride_w, output.hbuf);
                        }
                        else if (filter_h == 5 and filter_w == 5)
                        {
                            halide_gnne_conv2d_5x5(input.hbuf, weights.hbuf,
                                psum.hbuf, act.hbuf, v_range.hbuf, no_psum.hbuf, padding_h.before, padding_h.after,
                                padding_w.before, padding_w.after, stride_h, stride_w, output.hbuf);
                        }
                        else if (filter_h == 7 and filter_w == 7)
                        {
                            halide_gnne_conv2d_7x7(input.hbuf, weights.hbuf,
                                psum.hbuf, act.hbuf, v_range.hbuf, no_psum.hbuf, padding_h.before, padding_h.after,
                                padding_w.before, padding_w.after, stride_h, stride_w, output.hbuf);
                        }
                    }
                    // print_blob(output);
                    err_count += compare_blob(output.nraw, output.hraw, output.shape, 0.01f);
                    total_count += compute_size(output.shape);

                    std::cout << "[" << index++ << "]: err :" << err_count << ", total :" << total_count << std::endl;
                    // ASSERT_EQ(err_count, 0);
                }
            }
        }
    }
}

