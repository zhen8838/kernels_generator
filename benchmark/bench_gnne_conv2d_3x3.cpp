#include "benchmark/bench_gnne_conv2d.h"
#include "generated_kernels/halide_gnne_conv2d_3x3.h"

BENCHMARK_CONV2D_CASE(3, 3, 1, 1, false)
BENCHMARK_CONV2D_CASE(3, 3, 2, 2, false)
BENCHMARK_CONV2D_CASE(3, 3, 1, 1, true)
BENCHMARK_CONV2D_CASE(3, 3, 2, 2, true)

CELERO_MAIN