#pragma once

#include <Halide.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace Halideimpl
{

using namespace Halide;

inline Halide::Expr get_windowed_output_size(Halide::Expr size, Halide::Expr filter, Halide::Expr stride, Halide::Expr dilation, Halide::Expr pad_before, Halide::Expr pad_end)
{
    auto effective_filter_size = (filter - 1) * dilation + 1;
    return (size + pad_before + pad_end - effective_filter_size + stride) / stride;
}

template <typename T>
Halide::Func matmul(Halide::Buffer<T> input_a, Halide::Buffer<T> input_b, Halide::Buffer<T> act)
{
    Halide::Var N("N"), M("M");
    Halide::RDom K(0, input_a.width());
    Halide::Func prod("prod");
    prod(N, M) += input_a(K, M) * input_b(N, K);

    return prod;
}

template <template <typename> typename BT, typename T>
Halide::Func gnne_matmul(BT<T> &input_a, BT<T> &input_b, BT<T> &act, BT<T> &value_range, Halide::GeneratorParam<bool> &auto_schedule)
{
    Halide::Var N("N"), M("M");
    Halide::Func prod("prod"), producer("producer"), consumer("consumer");
    Halide::RDom k(0, input_a.width(), "k");
    prod(N, M) += Halide::cast<float>(input_a(k, M)) * Halide::cast<float>(input_b(N, k));
    producer(N, M) = prod(N, M);
    consumer(N, M) = clamp(
        Halide::cast<typename T::ElemType>(select(producer(N, M) < Halide::cast<float>(act(0)),
            producer(N, M) * Halide::cast<float>(act(1)) + Halide::cast<float>(act(2)),
            producer(N, M) * Halide::cast<float>(act(3)) + Halide::cast<float>(act(4)))),
        value_range(0), value_range(1));
    Halide::Var xo("xo"), yo("yo"), xi("xi"), yi("yi"), xy("xy"), yii("yii");
    if constexpr (std::is_same<BT<T>, Halide::Buffer<T>>::value)
    {
        if (input_a.height() % 8 == 0 and input_b.width() % 16 == 0)
        {
            consumer.tile(N, M, xi, yi, 16, 8)
                .parallel(yi);
        }
    }
    else
    {
        if (auto_schedule)
        {
            input_a.set_estimates({ { 0, 1024 }, { 0, 1024 } });
            input_b.set_estimates({ { 0, 1024 }, { 0, 1024 } });
            consumer.set_estimates({ { 0, 1024 }, { 0, 1024 } });
        }
        else
        {
            // 36.92倍
            // consumer.specialize(input_b.width() % 32 == 0 && input_a.height() % 16 == 0)
            //     .tile(N, M, xo, yo, xi, yi, 32, 16)
            //     .parallel(yi, 8)
            //     .vectorize(xi, 8);
            // 200 倍
            consumer.specialize(input_b.width() % 32 == 0 && input_a.height() % 32 == 0)
                .tile(N, M, xi, yi, 32, 32)
                .fuse(N, M, xy)
                .parallel(xy)
                .split(yi, yi, yii, 4)
                .vectorize(xi, 8)
                .unroll(xi)
                .unroll(yii);
        }
    }

    return consumer;
}

template <template <typename> typename BT, typename T1, typename T2, template <typename> typename BT2>
Halide::Func gnne_conv2d(BT<T1> &input, BT<T1> &weights,
    BT<T2> &psum, BT<T1> &act, BT<T1> &value_range,
    BT2<bool> &no_psum, BT2<int32_t> &pad_h_before,
    BT2<int32_t> &pad_h_end, BT2<int32_t> &pad_w_before,
    BT2<int32_t> &pad_w_end, BT2<int32_t> &stride_h,
    BT2<int32_t> &stride_w,
    Halide::GeneratorParam<int32_t> &kernel_height,
    Halide::GeneratorParam<int32_t> &kernel_width,
    Halide::GeneratorParam<bool> &auto_schedule)
{
    Halide::Var WO("WO"), HO("HO"), CI("CI"), B("B"), CO("CO");
    Halide::Func Padding("Padding"), Paded("Paded"), Conv("Conv"), Acted("Acted"), Clamped("Clamped"), Psumed("Psumed");
    Halide::RDom r(0, weights.width(), 0, weights.height(), 0, weights.dim(2).extent()); // w,h,ic

    Padding = Halide::BoundaryConditions::constant_exterior(input, 0,
        { { 0, input.width() },
            { 0, input.height() },
            { Halide::Expr(), Halide::Expr() },
            { Halide::Expr(), Halide::Expr() } });

    Halide::Expr in_channels = input.dim(2).extent(), out_channels = weights.dim(3).extent();

    Paded(WO, HO, CI, B) = Padding(WO - pad_w_before, HO - pad_h_before, CI, B);

    Conv(WO, HO, CO, B) += Halide::cast<float>(weights(r[0], r[1], r[2], CO)) * Halide::cast<float>(Paded(WO * stride_w + r[0], HO * stride_h + r[1], r[2], B)); // use float to sum

    Acted(WO, HO, CO, B) = select(
        Conv(WO, HO, CO, B) < Halide::cast<float>(act(0, CO)),
        Conv(WO, HO, CO, B) * Halide::cast<float>(act(1, CO)) + Halide::cast<float>(act(2, CO)),
        Conv(WO, HO, CO, B) * Halide::cast<float>(act(3, CO)) + Halide::cast<float>(act(4, CO))); // float

    Clamped(WO, HO, CO, B) = clamp(
        Halide::cast<typename T1::ElemType>(Acted(WO, HO, CO, B)),
        value_range(0), value_range(1));

    Psumed(WO, HO, CO, B) = select(no_psum,
        Clamped(WO, HO, CO, B),
        Clamped(WO, HO, CO, B) + Halide::cast<typename T1::ElemType>(psum(WO, HO, CO, B)));
    /* Schedule */
    Halide::Var Hi("Hi"), Hii("Hii"), Wi("Wi"), WH("WH");
    Halide::Var COo("COo"), COi("COi");
    if (auto_schedule.value())
    {
    }
    else
    {
        if (kernel_height.value() == 1 and kernel_width.value() == 1)
        {
            // 300 倍优化 @ stride=1 and 2
            // NOTE when k=1 and stride=2 can't use tiling, because the halide will malloc new memory block
            auto p_co = Psumed.parallel(CO);
            auto p_co_unroll = p_co.specialize(input.height() % 32 == 0)
                                   .split(HO, HO, Hi, 32)
                                   .unroll(Hi, 4);

            auto p_co_vw = p_co.specialize(input.width() % 32 == 0)
                               .split(WO, WO, Wi, 32)
                               .vectorize(Wi, 4)
                               .unroll(Wi);

            p_co.specialize(out_channels % 4 == 0)
                .vectorize(CO, 4);

            auto p_co_unroll_vw = p_co_unroll.specialize(input.width() % 32 == 0)
                                      .split(WO, WO, Wi, 32)
                                      .vectorize(Wi, 4)
                                      .unroll(Wi);

            Clamped.compute_at(Psumed, WO);

            Conv.update()
                .unroll(r.y)
                .unroll(r.x);
        }
        else
        {
            // method 1. Psum specialize + vector 达到500+
            Psumed.specialize(no_psum)
                .specialize(out_channels >= input.height())
                .parallel(CO);
            Psumed.specialize(no_psum)
                .parallel(HO);

            // 有Psum的时候需要进行向量化累加
            Psumed.specialize(out_channels >= input.height())
                .parallel(CO)
                .specialize(out_channels % 4 == 0)
                .vectorize(CO, 4);
            Psumed.parallel(HO)
                .specialize(out_channels % 4 == 0)
                .vectorize(CO, 4);
            Clamped.compute_at(Psumed, WO);
            Conv.update()
                .unroll(r.y)
                .unroll(r.x);
        }
    }
    // std::cout << "\nGenerate Gnne Conv2D kernel " << kernel_height.value() << " " << kernel_width.value() << std::endl;
    // Psumed.print_loop_nest();
    return Psumed;
}

template <template <typename> typename BT, typename T1, typename T2, template <typename> typename BT2>
Halide::Func gnne_depthwise_conv2d(BT<T1> &input, BT<T1> &weights,
    BT<T2> &psum, BT<T1> &act, BT<T1> &value_range,
    BT2<bool> &no_psum, BT2<int32_t> &pad_h_before,
    BT2<int32_t> &pad_h_end, BT2<int32_t> &pad_w_before,
    BT2<int32_t> &pad_w_end, BT2<int32_t> &stride_h,
    BT2<int32_t> &stride_w,
    Halide::GeneratorParam<int32_t> &kernel_height,
    Halide::GeneratorParam<int32_t> &kernel_width,
    Halide::GeneratorParam<bool> &auto_schedule)
{
    Halide::Var WO("WO"), HO("HO"), C("C"), B("B"); // depthwise only has C
    Halide::Func Padding("Padding"), Paded("Paded"), Conv("Conv"), Acted("Acted"), Clamped("Clamped"), Psumed("Psumed");
    Halide::RDom r(0, weights.width(), 0, weights.height()); // w,h
    Halide::Expr channels = input.dim(2).extent();

    Padding = Halide::BoundaryConditions::constant_exterior(input, 0,
        { { 0, input.width() },
            { 0, input.height() },
            { Halide::Expr(), Halide::Expr() },
            { Halide::Expr(), Halide::Expr() } });

    Paded(WO, HO, C, B) = Padding(WO - pad_w_before, HO - pad_h_before, C, B);

    Conv(WO, HO, C, B) += Halide::cast<float>(weights(r[0], r[1], 0, C)) * Halide::cast<float>(Paded(WO * stride_w + r[0], HO * stride_h + r[1], C, B)); // use float to sum

    Acted(WO, HO, C, B) = select(
        Conv(WO, HO, C, B) < Halide::cast<float>(act(0, C)),
        Conv(WO, HO, C, B) * Halide::cast<float>(act(1, C)) + Halide::cast<float>(act(2, C)),
        Conv(WO, HO, C, B) * Halide::cast<float>(act(3, C)) + Halide::cast<float>(act(4, C))); // float

    Clamped(WO, HO, C, B) = clamp(
        Halide::cast<typename T1::ElemType>(Acted(WO, HO, C, B)),
        value_range(0), value_range(1));

    Psumed(WO, HO, C, B) = select(no_psum,
        Clamped(WO, HO, C, B),
        Clamped(WO, HO, C, B) + Halide::cast<typename T1::ElemType>(psum(WO, HO, C, B)));
    /* Schedule */
    Halide::Var Hi("Hi"), Hii("Hii"), Wi("Wi"), WH("WH");
    Halide::Var Co("Co"), Ci("Ci");
    if (auto_schedule.value())
    {
    }
    else
    {
        if (kernel_height.value() == 1 and kernel_width.value() == 1)
        {
            // 300 倍优化 @ stride=1 and 2
            auto p_c = Psumed.parallel(C);
            auto p_c_unroll = p_c.specialize(input.height() % 32 == 0)
                                  .split(HO, HO, Hi, 32)
                                  .unroll(Hi, 4);

            auto p_c_vw = p_c.specialize(input.width() % 32 == 0)
                              .split(WO, WO, Wi, 32)
                              .vectorize(Wi, 4)
                              .unroll(Wi);

            p_c.specialize(channels % 4 == 0)
                .vectorize(C, 4);

            auto p_c_unroll_vw = p_c_unroll.specialize(input.width() % 32 == 0)
                                     .split(WO, WO, Wi, 32)
                                     .vectorize(Wi, 4)
                                     .unroll(Wi);

            Clamped.compute_at(Psumed, WO);

            Conv.update()
                .unroll(r.y)
                .unroll(r.x);
        }
        else
        {
            // method 1. Psum specialize + vector 达到500+
            Psumed.specialize(no_psum)
                .specialize(channels >= input.height())
                .parallel(C);
            Psumed.specialize(no_psum)
                .parallel(HO);

            // 有Psum的时候需要进行向量化累加
            Psumed.specialize(channels >= input.height())
                .parallel(C)
                .specialize(channels % 4 == 0)
                .vectorize(C, 4);
            Psumed.parallel(HO)
                .specialize(channels % 4 == 0)
                .vectorize(C, 4);
            Clamped.compute_at(Psumed, WO);
            Conv.update()
                .unroll(r.y)
                .unroll(r.x);
        }
    }
    // Psumed.print_loop_nest();
    return Psumed;
}

/**
 * @tparam BT Halide::Generator::input<Buffer<>>>
 * @tparam T1 value type
 * @tparam BT2 Halide::Generator::input<>
 * @return Halide::Func 
 */
template <template <typename> typename BT, typename T1, template <typename> typename BT2>
Halide::Func conv2d(
    BT<T1> &input, BT<T1> &weights, BT<T1> &bias, BT<T1> &value_range,
    BT2<int32_t> &pad_h_before,
    BT2<int32_t> &pad_h_end, BT2<int32_t> &pad_w_before,
    BT2<int32_t> &pad_w_end, BT2<int32_t> &stride_h,
    BT2<int32_t> &stride_w,
    Halide::GeneratorParam<int32_t> &kernel_height,
    Halide::GeneratorParam<int32_t> &kernel_width,
    Halide::GeneratorParam<bool> &auto_schedule)
{
    Halide::Var WO("WO"), HO("HO"), CI("CI"), B("B"), CO("CO");
    Halide::Func Padding("Padding"), Paded("Paded"), Conv("Conv"), Clamped("Clamped");
    Halide::RDom r(0, weights.width(), 0, weights.height(), 0, weights.dim(2).extent()); // w,h,ic

    Padding = Halide::BoundaryConditions::constant_exterior(input, 0,
        { { 0, input.width() },
            { 0, input.height() },
            { Halide::Expr(), Halide::Expr() },
            { Halide::Expr(), Halide::Expr() } });

    Halide::Expr in_channels = input.dim(2).extent(), out_channels = weights.dim(3).extent();
    Paded(WO, HO, CI, B) = Padding(WO - pad_w_before, HO - pad_h_before, CI, B);
    Conv(WO, HO, CO, B) += weights(r[0], r[1], r[2], CO) * Paded(WO * stride_w + r[0], HO * stride_h + r[1], r[2], B); // use float to sum
    Conv(WO, HO, CO, B) += bias(CO);
    Clamped(WO, HO, CO, B) = clamp(Conv(WO, HO, CO, B), value_range(0), value_range(1));
    if (auto_schedule)
    {
    }
    else
    {
        Var Hi("Hi"), Wi("Wi");
        if (kernel_height.value() == 1 and kernel_width.value() == 1)
        {

            // 300 倍优化 @ stride=1 and 2
            auto p_co = Clamped.parallel(CO);
            auto p_co_unroll = p_co.specialize(input.height() % 32 == 0)
                                   .specialize(stride_h == 1 and stride_w == 1)
                                   .split(HO, HO, Hi, 32)
                                   .unroll(Hi, 4);

            auto p_co_vw = p_co.specialize(input.width() % 32 == 0)
                               .split(WO, WO, Wi, 32)
                               .vectorize(Wi, 4)
                               .unroll(Wi);

            p_co.specialize(out_channels % 4 == 0)
                .vectorize(CO, 4);

            auto p_co_unroll_vw = p_co_unroll.specialize(input.width() % 32 == 0)
                                      .split(WO, WO, Wi, 32)
                                      .vectorize(Wi, 4)
                                      .unroll(Wi);
        }
        else
        {
            // method 1. Psum specialize + vector 达到500+
            // 有Psum的时候需要进行向量化累加
            Clamped.specialize(out_channels >= input.height())
                .parallel(CO)
                .specialize(out_channels % 4 == 0)
                .vectorize(CO, 4);
            Clamped.parallel(HO)
                .specialize(out_channels % 4 == 0)
                .vectorize(CO, 4);
        }

        Conv.compute_at(Clamped, WO);
        Conv.update()
            .unroll(r.y)
            .unroll(r.x);
    }

    return Clamped;
}

template <template <typename> typename BT, typename T1, template <typename> typename BT2>
Halide::Func conv2d_depthwise(
    BT<T1> &input, BT<T1> &weights, BT<T1> &bias, BT<T1> &value_range,
    BT2<int32_t> &pad_h_before,
    BT2<int32_t> &pad_h_end, BT2<int32_t> &pad_w_before,
    BT2<int32_t> &pad_w_end, BT2<int32_t> &stride_h,
    BT2<int32_t> &stride_w,
    Halide::GeneratorParam<int32_t> &kernel_height,
    Halide::GeneratorParam<int32_t> &kernel_width,
    Halide::GeneratorParam<bool> &auto_schedule)
{

    Halide::Var WO("WO"), HO("HO"), C("C"), B("B");
    Halide::Func Padding("Padding"), Paded("Paded"), Conv("Conv"), Clamped("Clamped");
    Halide::RDom r(0, weights.width(), 0, weights.height()); // w,h

    Padding = Halide::BoundaryConditions::constant_exterior(input, 0,
        { { 0, input.width() },
            { 0, input.height() },
            { Halide::Expr(), Halide::Expr() },
            { Halide::Expr(), Halide::Expr() } });

    Halide::Expr channels = input.dim(2).extent();
    Paded(WO, HO, C, B) = Padding(WO - pad_w_before, HO - pad_h_before, C, B);
    Conv(WO, HO, C, B) += weights(r.x, r.y, 0, C) * Paded(WO * stride_w + r.x, HO * stride_h + r.y, C, B); // use float to sum
    Conv(WO, HO, C, B) += bias(C);
    Clamped(WO, HO, C, B) = clamp(Conv(WO, HO, C, B), value_range(0), value_range(1));

    /* Schedule */
    Halide::Var Hi("Hi"), Hii("Hii"), Wi("Wi"), WH("WH");
    if (auto_schedule.value())
    {
    }
    else
    {
        if (kernel_height.value() == 1 and kernel_width.value() == 1)
        {
            // 300 倍优化 @ stride=1 and 2
            auto p_c = Clamped.parallel(C);
            auto p_c_unroll = p_c.specialize(input.height() % 32 == 0)
                                  .specialize(stride_h == 1 and stride_w == 1)
                                  .split(HO, HO, Hi, 32)
                                  .unroll(Hi, 4);

            auto p_c_vw = p_c.specialize(input.width() % 32 == 0)
                              .split(WO, WO, Wi, 32)
                              .vectorize(Wi, 4)
                              .unroll(Wi);

            p_c.specialize(channels % 4 == 0)
                .vectorize(C, 4);

            auto p_c_unroll_vw = p_c_unroll.specialize(input.width() % 32 == 0)
                                     .split(WO, WO, Wi, 32)
                                     .vectorize(Wi, 4)
                                     .unroll(Wi);
        }
        else
        {
            // method 1. Psum specialize + vector 达到500+
            Clamped.specialize(channels >= input.height())
                .parallel(C)
                .specialize(channels % 4 == 0)
                .vectorize(C, 4);
            Clamped.parallel(HO)
                .specialize(channels % 4 == 0)
                .vectorize(C, 4);
        }
        Conv.compute_at(Clamped, WO);
        Conv.update()
            .unroll(r.y)
            .unroll(r.x);
    }
    // Psumed.print_loop_nest();
    return Clamped;
}
}
