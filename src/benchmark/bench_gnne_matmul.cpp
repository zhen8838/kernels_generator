#include "benchmark/bench_gnne_matmul.h"
#include "hkg/halide_gnne_matmul.h"
// #include "kernels/halide_gnne_matmul_auto.h"

BENCHMARK_F(Gnne_Matmul, ref, GnneMatmulTestFixture, 5, 10)
{
  Nncaseimpl::gnne_matmul(input_a.nraw, input_b.nraw, output.nraw, act.nraw, input_a.shape[0], input_a.shape[1], input_b.shape[1], nncase::value_range<NNCASE_TYPE_t>{v_range.nraw[0], v_range.nraw[1]});
}

BASELINE_F(Gnne_Matmul, opt, GnneMatmulTestFixture, 5, 10)
{
  halide_gnne_matmul(input_a.hbuf, input_b.hbuf, act.hbuf, v_range.hbuf, output.hbuf);
}

BASELINE_F(Gnne_Matmul, auto, GnneMatmulTestFixture, 5, 10)
{
    halide_gnne_matmul_auto(input_a.hbuf, input_b.hbuf, act.hbuf, v_range.hbuf, output.hbuf);
}

CELERO_MAIN