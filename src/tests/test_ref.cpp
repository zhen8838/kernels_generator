#include "include/bench_gnne_matmul.h"
#include "include/ref_halide_impl.h"
#include "include/ref_nncase_impl.h"
#include "nncase.h"
#include "test_utils.h"
#include <bench_gnne_matmul.h>
#include <gtest/gtest.h>

// #define DEBUG
typedef nncase::bfloat16 NNCASE_TYPE_t;
typedef Halide::bfloat16_t HALIDE_TYPE_t;

TEST(test, matmul)
{
    SRAND((uint64_t)std::chrono::system_clock::now().time_since_epoch().count());
    size_t M = 30, N = 50, K = 20;
    nncase::runtime_shape_t in_a_shape { M, K }, in_b_shape { K, N }, out_shape { M, N }, act_shape { 5 };
    NNCASE_TYPE_t *input_a = nullptr, *input_b = nullptr, *act = nullptr, *output_nncase = nullptr, *output_halide = nullptr;
    init_blob(input_a, init_method::rand, in_a_shape);
    init_blob(input_b, init_method::rand, in_b_shape);
    init_blob(act, init_method::rand, act_shape);

    init_blob(output_nncase, init_method::zero, out_shape);
    init_blob(output_halide, init_method::zero, out_shape);

    auto before = std::chrono::system_clock::now();
    Nncaseimpl::matmul(input_a, input_b, output_nncase, act, M, K, N, nncase::value_range<NNCASE_TYPE_t>::full());
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> diff1 = now - before;

    Halide::Buffer<HALIDE_TYPE_t> input_a_h((HALIDE_TYPE_t *)input_a, in_a_shape[1], in_a_shape[0]);
    Halide::Buffer<HALIDE_TYPE_t> input_b_h((HALIDE_TYPE_t *)input_b, in_b_shape[1], in_b_shape[0]);
    Halide::Buffer<HALIDE_TYPE_t> act_h((HALIDE_TYPE_t *)input_b, act_shape[0]);
    Halide::Buffer<HALIDE_TYPE_t> output_h((HALIDE_TYPE_t *)output_halide, N, M);

    before = std::chrono::system_clock::now();
    auto inner_acted = Halideimpl::matmul<HALIDE_TYPE_t>(input_a_h, input_b_h, act_h);
    inner_acted.realize(output_h);
    now = std::chrono::system_clock::now();
    std::chrono::duration<double> diff2 = now - before;

    auto errcnt = compare_blob(output_halide, output_nncase, out_shape, .4f);
    std::cout << diff1.count() << " " << diff2.count() << std::endl;
    ASSERT_EQ(errcnt, 0);
}

TEST(test, select)
{
    nncase::runtime_shape_t in_shape { 5 };
    NNCASE_TYPE_t *input = nullptr, *output = nullptr;
    init_blob(input, init_method::sequence, in_shape);
    init_blob(output, init_method::zero, in_shape);
    print_blob(input, in_shape);
    Halide::Buffer<HALIDE_TYPE_t> input_h((HALIDE_TYPE_t *)input, in_shape[0]);
    Halide::Buffer<HALIDE_TYPE_t> output_h((HALIDE_TYPE_t *)output, in_shape[0]);
    Halide::Func selected("selected");
    Halide::Var K;
    selected(K) = Halide::select(input_h(K) < 3,
                      -input_h(K) * 2,
                      input_h(K))
        * 2;
    selected.realize(output_h);
    print_blob(output, in_shape);
    // selected = input_h.sele
}

void fast_bench(std::string name, std::function<void(void)> func, int repeat = 10)
{
    for (size_t i = 0; i < 3; i++)
    {
        func();
    }
    auto before = std::chrono::system_clock::now();
    for (size_t i = 0; i < repeat; i++)
    {
        func();
    }

    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = (now - before) / repeat;
    std::cout << name << " eval time : " << diff.count() << std::endl;
}

TEST(test, gnne_matmul)
{
    SRAND((uint64_t)std::chrono::system_clock::now().time_since_epoch().count());
    size_t M = 10, N = 20, K = 10;
    auto [input_a, input_b, act, output_n, output_h, value_range] = GnneMatmulTestFixture::get_data<NNCASE_TYPE_t, halide_type_t>(M, N, K);
#ifdef DEBUG
    print_blob(input_a, in_a_shape);
    print_blob(input_b, in_b_shape);
    print_blob(act, act_shape);
#endif
    Nncaseimpl::gnne_matmul(input_a.raw, input_b.raw, output_n.raw, act.raw, input_a.shape[0], input_a.shape[1], input_b.shape[1], nncase::value_range<NNCASE_TYPE_t> { value_range.raw[0], value_range.raw[1] });
    // Halide::Buffer<HALIDE_TYPE_t> input_a_bf(input_a.buf,input_a.shape);
    // output_n.buf = Halideimpl::gnne_matmul(input_a.buf, input_b.buf, act.buf, value_range.buf);

    // auto errcnt = compare_blob(output_h, output_n, output_n.shape, 0.03f, false);
#ifdef DEBUG
    print_blob(output_halide, out_shape);
    print_blob(output_nncase, out_shape);
#endif
    // ASSERT_EQ(errcnt, 0);
}