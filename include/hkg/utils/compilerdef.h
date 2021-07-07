#pragma once
#include <type_traits>

#define NNCASE_LITTLE_ENDIAN 1

// clang-format off
#if __cplusplus >= 201703L
  #define NNCASE_INLINE_VAR inline
  #define NNCASE_UNUSED [[maybe_unused]]
  namespace nncase
  {
  template <class Callable, class... Args>
  using invoke_result_t = std::invoke_result_t<Callable, Args...>;
  }
#else
  #define NNCASE_INLINE_VAR
  #if defined(_MSC_VER)
    #define NNCASE_UNUSED
  #else
    #define NNCASE_UNUSED __attribute__((unused))
  #endif
  namespace nncase
  {
  template <class Callable, class... Args>
  using invoke_result_t = std::result_of_t<Callable(Args...)>;
  }
#endif