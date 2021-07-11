#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
namespace internal
{

struct Target
{
    bool sse41 = false; ///< Use SSE 4.1 and earlier instructions. Only relevant on x86.
    bool avx = false; ///< Use AVX 1 instructions. Only relevant on x86.
    bool f16c = false; ///< Enable x86 16-bit float support
    bool fma = false; ///< Enable x86 FMA instruction
    bool avx2 = false; ///< Use AVX 2 instructions. Only relevant on x86.
    bool avx512 = false;
    bool avx512_knl = false;
    bool avx512_skylake = false;
    bool avx512_cannonlake = false;
};

std::ostream &operator<<(std::ostream &os, const Target &t)
{
    os << "Target : \n"
       << " have_sse41: " << t.sse41
       << "\n have_avx : " << t.avx
       << "\n have_f16c : " << t.f16c
       << "\n have_fma : " << t.fma
       << "\n have_avx2 : " << t.avx2
       << "\n have_avx512 : " << t.avx512
       << "\n have_avx512_knl : " << t.avx512_knl
       << "\n have_avx512_skylake : " << t.avx512_skylake
       << "\n have_avx512_cannonlake : " << t.avx512_cannonlake
       << ";";
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
    bool use_64_bits = (sizeof(size_t) == 8);
    // int bits = use_64_bits ? 64 : 32;

#if __riscv__
    Target::Arch arch = Target::RISCV;
#else
#if __mips__ || __mips || __MIPS__
    Target::Arch arch = Target::MIPS;
#else
#if defined(__arm__) || defined(__aarch64__)
    Target::Arch arch = Target::ARM;
#else
#if defined(__powerpc__) && (defined(__FreeBSD__) || defined(__linux__))
    Target::Arch arch = Target::POWERPC;

#if defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
#elif defined(__FreeBSD__)
    unsigned long hwcap, hwcap2;
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
    elf_aux_info(AT_HWCAP2, &hwcap2, sizeof(hwcap2));
#endif
    bool have_altivec = (hwcap & PPC_FEATURE_HAS_ALTIVEC) != 0;
    bool have_vsx = (hwcap & PPC_FEATURE_HAS_VSX) != 0;
    bool arch_2_07 = (hwcap2 & PPC_FEATURE2_ARCH_2_07) != 0;

    user_assert(have_altivec)
        << "The POWERPC backend assumes at least AltiVec support. This machine does not appear to have AltiVec.\n";

    if (have_vsx)
        target.vsx = true;
    if (arch_2_07)
        target.power_arch_2_07 = true;
#else
    Target target;

    int info[4];
    cpuid(info, 1, 0);
    bool have_sse41 = (info[2] & (1 << 19)) != 0;
    bool have_sse2 = (info[3] & (1 << 26)) != 0;
    bool have_avx = (info[2] & (1 << 28)) != 0;
    bool have_f16c = (info[2] & (1 << 29)) != 0;
    bool have_rdrand = (info[2] & (1 << 30)) != 0;
    bool have_fma = (info[2] & (1 << 12)) != 0;

    if (have_sse41)
    {
        target.sse41 = true;
    }
    if (have_avx)
    {
        target.avx = true;
    }
    if (have_f16c)
    {
        target.f16c = true;
    }
    if (have_fma)
    {
        target.fma = true;
    }

    if (use_64_bits && have_avx && have_f16c && have_rdrand)
    {
        // So far, so good.  AVX2/512?
        // Call cpuid with eax=7, ecx=0
        int info2[4];
        cpuid(info2, 7, 0);
        const uint32_t avx2 = 1U << 5;
        const uint32_t avx512f = 1U << 16;
        const uint32_t avx512dq = 1U << 17;
        const uint32_t avx512pf = 1U << 26;
        const uint32_t avx512er = 1U << 27;
        const uint32_t avx512cd = 1U << 28;
        const uint32_t avx512bw = 1U << 30;
        const uint32_t avx512vl = 1U << 31;
        const uint32_t avx512ifma = 1U << 21;
        const uint32_t avx512 = avx512f | avx512cd;
        const uint32_t avx512_knl = avx512 | avx512pf | avx512er;
        const uint32_t avx512_skylake = avx512 | avx512vl | avx512bw | avx512dq;
        const uint32_t avx512_cannonlake = avx512_skylake | avx512ifma; // Assume ifma => vbmi
        if ((info2[1] & avx2) == avx2)
        {
            target.avx2 = true;
        }
        if ((info2[1] & avx512) == avx512)
        {
            target.avx512 = true;
            if ((info2[1] & avx512_knl) == avx512_knl)
            {
                target.avx512_knl = true;
            }
            if ((info2[1] & avx512_skylake) == avx512_skylake)
            {
                target.avx512_skylake = true;
            }
            if ((info2[1] & avx512_cannonlake) == avx512_cannonlake)
            {
                target.avx512_cannonlake = true;

#if LLVM_VERSION >= 120
                // Sapphire Rapids support was added in LLVM 12, so earlier versions cannot support this CPU's features.
                const uint32_t avx512vnni = 1U << 11; // vnni result in ecx
                const uint32_t avx512bf16 = 1U << 5; // bf16 result in eax, with cpuid(eax=7, ecx=1)
                int info3[4];
                cpuid(info3, 7, 1);
                if ((info2[2] & avx512vnni) == avx512vnni && (info3[0] & avx512bf16) == avx512bf16)
                {
                    target.avx512_sapphirerapids = true;
                }
#endif
            }
        }
    }
#endif
#endif
#endif
#endif

    return target;
}

} // namespace internal
