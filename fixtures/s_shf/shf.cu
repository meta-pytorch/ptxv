/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Shifts by a run-time amount, checked against GB300 bitwise on both front-ends:
// `ptxv hw fixtures/s_shf` compares every output word with the simulator. ptxas lowers them to
// SHF.{L,R}.{U32,S32,U64,S64}[.HI] / SHF.*.W and their uniform USHF forms. Shift amounts span
// 0..33, the 32/64-bit boundaries, values >= 64 (FA4's causal mask shifts 0xffffffff by up to
// 128: r2p_bitmask_below) and "negative" (>= 2^31) u32 amounts.
// 64 CTAs x 128 threads; thread t of CTA c shifts the pair (vals[2t], vals[2t+1]) by
// sh[(t + 3c) % 64]; thread 0 of CTA c also shifts the uniform pair (ux, uy) by a uniform
// amount derived from c.
// nvcc -ptx -arch=sm_103a -O3 -o kernel.ptx shf.cu
#include <stdint.h>

#define NOUT 14
#define NSH 64

__device__ __forceinline__ void forms(uint32_t x, uint32_t y, uint32_t s, uint32_t* o) {
    uint32_t r;
    uint64_t xy = (uint64_t)y << 32 | x, q;
    asm("shl.b32 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(s)); o[0] = r;
    asm("shr.u32 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(s)); o[1] = r;
    asm("shr.s32 %0, %1, %2;" : "=r"(r) : "r"(y), "r"(s)); o[2] = r;
    asm("shl.b64 %0, %1, %2;" : "=l"(q) : "l"(xy), "r"(s)); o[3] = (uint32_t)q; o[4] = (uint32_t)(q >> 32);
    asm("shr.u64 %0, %1, %2;" : "=l"(q) : "l"(xy), "r"(s)); o[5] = (uint32_t)q; o[6] = (uint32_t)(q >> 32);
    asm("shr.s64 %0, %1, %2;" : "=l"(q) : "l"(xy), "r"(s)); o[7] = (uint32_t)q; o[8] = (uint32_t)(q >> 32);
    asm("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(y), "r"(s)); o[9] = r;
    asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(y), "r"(s)); o[10] = r;
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(y), "r"(s)); o[11] = r;
    asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(y), "r"(s)); o[12] = r;
    // FA4's mask: 0xffffffff >> max(32*(k+1) - limit, 0).
    asm("shr.u32 %0, %1, %2;" : "=r"(r) : "r"(0xffffffffu), "r"(s)); o[13] = r;
}

extern "C" __global__ void __launch_bounds__(128)
    shf(const uint32_t* vals, const uint32_t* sh, uint32_t* out, uint32_t ux, uint32_t uy) {
    uint32_t t = threadIdx.x, c = blockIdx.x;
    forms(vals[2 * t], vals[2 * t + 1], sh[(t + 3 * c) % NSH], out + (c * 129 + t) * NOUT);
    if (t == 0) {
        // Uniform operands: kernel parameters and the CTA index (USHF).
        uint32_t s = c < 32 ? c * 5 : 0xffffffffu - (c - 32) * 9;
        forms(ux, uy, s, out + (c * 129 + 128) * NOUT);
    }
}
