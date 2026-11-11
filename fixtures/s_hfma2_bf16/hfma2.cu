/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// HFMA2.BF16_V2 forms against GB300 (see case.toml; kernel.cubin patches the first fma2 into
// the immediate form `HFMA2.BF16_V2 R17, R0, 2, 0.25, R15`): out[4t+0] = fma(a, (2, 0.25), c);
// out[4t+1] = out[4t+3] = a + c (HADD2.BF16_V2); out[4t+2] = fma(a, (1.0, 1.0), -0) (HMUL2);
// out[1024 + t] = fma(c, b, a) (register form); out[1280 + t] / out[1536 + t] = f32(low /
// high bf16 half of a) + c as f32 (add.f32.bf16: FHADD.BF16); out[1792 + t] = high half - c
// (FHADD.BF16 with -c); out[2048 + t] = low half + -0 (FHADD.BF16 with -RZ).
#include <cstdint>
__device__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
extern "C" __global__ void hk(const uint32_t *a, const uint32_t *b, const uint32_t *c, uint32_t *out) {
  unsigned t = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t x = a[t], y = b[t], z = c[t];
  out[4 * t + 0] = fma2(x, y, z);
  out[4 * t + 1] = fma2(x, 0x3f803f80u, z);
  out[4 * t + 2] = fma2(x, 0x3f803f80u, 0x80008000u);
  uint32_t d;
  asm volatile("add.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(x), "r"(z));
  out[4 * t + 3] = d;
  out[1024 + t] = fma2(z, y, x);
  unsigned short lo = x & 0xffff, hi = x >> 16;
  float f;
  asm volatile("{ .reg .b16 h; mov.b16 h, %1; add.rn.f32.bf16 %0, h, %2; }" : "=f"(f) : "h"(lo), "f"(__uint_as_float(z)));
  out[1280 + t] = __float_as_uint(f);
  asm volatile("{ .reg .b16 h; mov.b16 h, %1; add.rn.f32.bf16 %0, h, %2; }" : "=f"(f) : "h"(hi), "f"(__uint_as_float(z)));
  out[1536 + t] = __float_as_uint(f);
  asm volatile("{ .reg .b16 h; mov.b16 h, %1; sub.rn.f32.bf16 %0, h, %2; }" : "=f"(f) : "h"(hi), "f"(__uint_as_float(z)));
  out[1792 + t] = __float_as_uint(f);
  asm volatile("{ .reg .b16 h; mov.b16 h, %1; add.rn.f32.bf16 %0, h, %2; }" : "=f"(f) : "h"(lo), "f"(-0.0f));
  out[2048 + t] = __float_as_uint(f);
}
