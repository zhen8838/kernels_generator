#include "benchmark/bench_gnne_conv2d.h"
#include "hkg/halide_gnne_conv2d_1x1.h"

BENCHMARK_GNNE_CONV2D_CASE_IMPL(1, 1, 1, 1, false, true, 5, 1)
BENCHMARK_GNNE_CONV2D_CASE_IMPL(1, 1, 1, 1, false, false, 5, 1)
BENCHMARK_GNNE_CONV2D_CASE_IMPL(1, 1, 2, 2, false, true, 5, 1)
BENCHMARK_GNNE_CONV2D_CASE_IMPL(1, 1, 2, 2, false, false, 5, 1)

CELERO_MAIN