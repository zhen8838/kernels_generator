#include "benchmark/bench_gnne_conv2d.h"
#include "hkg/halide_gnne_conv2d_5x5.h"

BENCHMARK_GNNE_CONV2D_CASE(5, 5, 1, 1, false)
BENCHMARK_GNNE_CONV2D_CASE(5, 5, 1, 1, true)
BENCHMARK_GNNE_CONV2D_CASE(5, 5, 2, 2, false)
BENCHMARK_GNNE_CONV2D_CASE(5, 5, 2, 2, true)

CELERO_MAIN