#pragma once
#include "../nncase_kernels.h"
#include "../utils.h"
#include <celero/Celero.h>
#include <nncase/kernels/kernel_utils.h>
#include <tuple>
#include <vector>

std::vector<std::vector<size_t>> MatMul_Test_Params = {
    {10, 20, 10},    // M , N , K
    {16, 32, 64},    // M , N , K
    {256, 128, 64},  // M , N , K
    {384, 256, 96},  // M , N , K
    {512, 512, 512}, // M , N , K
};

using NNCASE_TYPE_t = nncase::bfloat16;

class GnneMatmulTestFixture : public celero::TestFixture
{
public:
  Tensor_t<NNCASE_TYPE_t, NNCASE_TYPE_t> input_a, input_b, act, output, v_range;
  std::vector<celero::TestFixture::ExperimentValue> getExperimentValues() const override
  {
    std::vector<celero::TestFixture::ExperimentValue> bufferSizes(MatMul_Test_Params.size());
    std::iota(bufferSizes.begin(), bufferSizes.end(), (size_t)0);
    return bufferSizes;
  }
  template <typename T1, typename T2>
  static auto get_data(size_t M, size_t N, size_t K)
  {
    Tensor_t<T1, T2> input_a_({M, K});
    Tensor_t<T1, T2> input_b_({K, N});
    Tensor_t<T1, T2> output_({M, N});
    Tensor_t<T1, T2> act_({5});
    Tensor_t<T1, T2> v_range_({2});
    input_a_.allocate();
    input_b_.allocate();
    output_.allocate(init_method::zero);
    act_.allocate();
    v_range_.allocate();
    v_range_.as_value_range(range_t::full);

    return std::make_tuple(input_a_, input_b_, act_, output_, v_range_);
  }

  void setUp(const celero::TestFixture::ExperimentValue &experimentValue) override
  {
    auto param = MatMul_Test_Params[experimentValue.Value];
    auto M = param[0];
    auto K = param[1];
    auto N = param[2];
    std::tie(input_a, input_b, act, output, v_range) = get_data<NNCASE_TYPE_t, NNCASE_TYPE_t>(M, N, K);
  }

  void tearDown() override
  {
  }
};