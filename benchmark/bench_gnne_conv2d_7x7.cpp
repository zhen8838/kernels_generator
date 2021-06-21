#include "benchmark/bench_gnne_conv2d.h"
#include "generated_kernels/halide_gnne_conv2d_7x7.h"

BENCHMARK_CONV2D_CASE(7, 7, 1, 1, false)
BENCHMARK_CONV2D_CASE(7, 7, 1, 1, true)
BENCHMARK_CONV2D_CASE(7, 7, 2, 2, false)
BENCHMARK_CONV2D_CASE(7, 7, 2, 2, true)

CELERO_MAIN