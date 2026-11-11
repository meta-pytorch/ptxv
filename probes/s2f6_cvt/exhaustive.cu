// probes/s2f6_cvt/exhaustive: every (data16, scale16) pair, 2^32 in all, of
// cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 against a per-lane model table
// M[scale byte << 8 | code byte] computed on the host (exhaustive_host.c). See run.sh.
#include <cstdint>
extern "C" __global__ void full(const uint16_t* M, unsigned long long* bad, uint32_t* first) {
  uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;  // data16
  unsigned long long nb = 0;
  for (uint32_t s = 0; s < 65536; s++) {
    uint32_t r;
    asm volatile("{ .reg .b16 a, b; cvt.u16.u32 a, %1; cvt.u16.u32 b, %2;\n"
                 " cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 %0, a, b; }"
                 : "=r"(r) : "r"(d), "r"(s));
    uint32_t e = M[((s & 0xff) << 8) | (d & 0xff)] | ((uint32_t)M[(s & 0xff00) | (d >> 8)] << 16);
    if (r != e) {
      nb++;
      if (atomicCAS(first, 0xffffffffu, 0) == 0xffffffffu) {
        first[1] = d; first[2] = s; first[3] = r; first[4] = e;
      }
    }
  }
  atomicAdd(bad, nb);
}
