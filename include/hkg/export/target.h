#pragma once
#include <cstddef>
#include <cstdint>
#include <iostream>
namespace internal
{

struct Target
{
    bool feature_sse41 = false; ///< Use SSE 4.1 and earlier instructions. Only relevant on x86.
    bool feature_avx = false; ///< Use AVX 1 instructions. Only relevant on x86.
    bool feature_avx2 = false; ///< Use AVX 2 instructions. Only relevant on x86.
    bool feature_fma = false; ///< Enable x86 FMA instruction
    bool feature_f16c = false; ///< Enable x86 16-bit float support
};

std::ostream &operator<<(std::ostream &os, const Target &t)
{
    os << "Target : have_sse41: " << t.feature_sse41 << " , have_avx : " << t.feature_avx << " , have_avx2 : " << t.feature_avx2 << " , have_fma : " << t.feature_fma << " , have_f16c : " << t.feature_f16c << ";" << std::endl;
    return os;
}

Target get_host_target();
static Target host_target = get_host_target();

// clang-format off
#ifdef _MSC_VER
  static void cpuid(int info[4], int infoType, int extra)
  {
      __cpuidex(info, infoType, extra);
  }
#else
  #if defined(__x86_64__) || defined(__i386__)
    // CPU feature detection code taken from ispc
    // (https://github.com/ispc/ispc/blob/master/builtins/dispatch.ll)
    #ifdef _LP64
      static void cpuid(int info[4], int infoType, int extra)
      {
          __asm__ __volatile__(
              "cpuid                 \n\t"
              : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
              : "0"(infoType), "2"(extra));
      }
    #else
      static void cpuid(int info[4], int infoType, int extra)
      {
          // We save %ebx in case it's the PIC register
          __asm__ __volatile__(
              "mov{l}\t{%%}ebx, %1  \n\t"
              "cpuid                 \n\t"
              "xchg{l}\t{%%}ebx, %1  \n\t"
              : "=a"(info[0]), "=r"(info[1]), "=c"(info[2]), "=d"(info[3])
              : "0"(infoType), "2"(extra));
      }
    #endif
  #endif
#endif
// clang-format on

Target get_host_target()
{
    int info[4];
    cpuid(info, 1, 0);

    static_assert(sizeof(size_t) == 8); // must be 64 bit

    bool have_sse41 = (info[2] & (1 << 19)) != 0;
    //  bool have_sse2 = (info[3] & (1 << 26)) != 0;
    bool have_avx = (info[2] & (1 << 28)) != 0;
    bool have_f16c = (info[2] & (1 << 29)) != 0;
    bool have_rdrand = (info[2] & (1 << 30)) != 0;
    bool have_fma = (info[2] & (1 << 12)) != 0;
    bool have_avx2 = false;

    // Target host_target { have_sse41, have_avx };
    if (have_avx && have_f16c && have_rdrand)
    {
        // So far, so good.  AVX2/512?
        // Call cpuid with eax=7, ecx=0
        int info2[4];
        cpuid(info2, 7, 0);
        const uint32_t avx2 = 1U << 5;
        const uint32_t avx512f = 1U << 16;
        // const uint32_t avx512dq = 1U << 17;
        // const uint32_t avx512pf = 1U << 26;
        // const uint32_t avx512er = 1U << 27;
        const uint32_t avx512cd = 1U << 28;
        // const uint32_t avx512bw = 1U << 30;
        // const uint32_t avx512vl = 1U << 31;
        // const uint32_t avx512ifma = 1U << 21;
        const uint32_t avx512 = avx512f | avx512cd;
        //  const uint32_t avx512_knl = avx512 | avx512pf | avx512er;
        // const uint32_t avx512_skylake = avx512 | avx512vl | avx512bw | avx512dq;
        //  const uint32_t avx512_cannonlake = avx512_skylake | avx512ifma; // Assume ifma => vbmi
        if ((info2[1] & avx2) == avx2)
        {
            have_avx2 = true;
        }
        if ((info2[1] & avx512) == avx512)
        {
            // todo add more arch
        }
    }
    Target target { have_sse41, have_avx, have_avx2, have_fma, have_f16c };
    return target;
}

} // namespace internal
