#pragma once
#include "halide_kernels.h"
using namespace Halide;

class halide_matmul : public Halide::Generator<halide_matmul>
{
public:
    Input<Buffer<bfloat16_t>> input_a { "input_a", 2 };
    Input<Buffer<bfloat16_t>> input_b { "input_b", 2 };

    Output<Buffer<bfloat16_t>> output { "output", 2 };

    void generate()
    {
        /* THE ALGORITHM */
        // [x x K] x [K x y] = [x x y]
        Var x("x"), y("y");
        RDom k(0, input_a.dim(0).extent());
        Func matmul("matmul"), inner_sum("inner_sum");
        inner_sum(x, y) = 0;
        inner_sum(x, y) += input_a(k, y) * input_b(x, k);
        matmul(x, y) = inner_sum(x, y);
    }

    void schedule()
    {
        if (auto_schedule)
        {
            input_a.set_estimates({ { 0, 512 }, { 0, 512 } });
            input_b.set_estimates({ { 0, 512 }, { 0, 512 } });
        }
    }
};

class halide_gnne_matmul : public Halide::Generator<halide_gnne_matmul>
{
public:
    Input<Buffer<bfloat16_t>> input_a { "input_a", 2 };
    Input<Buffer<bfloat16_t>> input_b { "input_b", 2 };
    Input<Buffer<bfloat16_t>> act { "act", 1 };
    Input<Buffer<bfloat16_t>> value_range { "var_range", 1 };

    Output<Buffer<bfloat16_t>> output { "output", 2 };

    void generate()
    {
        /* THE ALGORITHM */
        // C(x,y)+=A(k,y)*B(x,k)
        act.dim(0).set_bounds(0, 5).set_stride(1);
        value_range.dim(0).set_bounds(0, 2).set_stride(1);
        output = Halideimpl::gnne_matmul(input_a, input_b, act, value_range, auto_schedule);
    }
};

class halide_gnne_conv2d : public Halide::Generator<halide_gnne_conv2d>
{
public:
    // Param
    GeneratorParam<int32_t> kernel_width { "kernel_width", 3, 1, 7 };
    GeneratorParam<int32_t> kernel_height { "kernel_height", 3, 1, 7 };

    //
    Input<Buffer<bfloat16_t>> input { "input", 4 }; // [b,ic,h,w]
    Input<Buffer<bfloat16_t>> weights { "weights", 4 }; // [oc,ic,h,w]
    Input<Buffer<float>> psum { "psum", 4 }; // same as output
    Input<Buffer<bfloat16_t>> act { "act", 2 }; // [out channels, 5]
    Input<Buffer<bfloat16_t>> value_range { "value_range", 1 }; // [2]
    Input<bool> no_psum { "no_psum" };

    Input<int32_t> pad_h_before { "pad_h_before" };
    Input<int32_t> pad_h_end { "pad_h_end" };
    Input<int32_t> pad_w_before { "pad_w_before" };
    Input<int32_t> pad_w_end { "pad_w_end" };
    Input<int32_t> stride_h { "stride_h" };
    Input<int32_t> stride_w { "stride_w" };

    Output<Buffer<bfloat16_t>> output { "output", 4 };

    void generate()
    {
        weights.dim(0).set_bounds(0, kernel_width);
        weights.dim(1).set_bounds(0, kernel_height);
        act.dim(0).set_bounds(0, 5).set_stride(1).dim(1).set_stride(5);
        value_range.dim(0).set_bounds(0, 2).set_stride(1);

        output = Halideimpl::gnne_conv2d(input, weights, psum, act, value_range, no_psum, pad_h_before, pad_h_end, pad_w_before, pad_w_end, stride_h, stride_w, kernel_width, kernel_height, auto_schedule);
    }
};