/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// MUFU forms in a softmax row and an RMSNorm row (one warp = one row of 32), plus every
// modeled MUFU function on raw random bits (NaN, denormal, negative, overflow inputs).
// `ptxv hw fixtures/s_mufu` compares every output word with the SASS front-end's run.
// nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin mufu.cu
#include <stdint.h>

#define NOUT 20

__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float rcp(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float rsq(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float lg2(float x) { float y; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float sqr(float x) { float y; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t u(float x) { return __float_as_uint(x); }
__device__ __forceinline__ float f(uint32_t x) { return __uint_as_float(x); }

extern "C" __global__ void __launch_bounds__(128) mufu(const uint32_t* in, uint32_t* out) {
    int t = threadIdx.x;
    uint32_t a = in[4 * t], b = in[4 * t + 1], c = in[4 * t + 2], d = in[4 * t + 3];
    uint32_t* o = out + t * NOUT;
    // Softmax row (inputs in +-[2, 4), so no max subtraction is needed).
    float x = f((a & 0x807fffff) | 0x40000000);
    float e = ex2(x * 1.44269504f);
    float s = e;
    for (int k = 16; k > 0; k >>= 1) s += __shfl_xor_sync(0xffffffff, s, k);
    float r = rcp(s);
    o[0] = u(e); o[1] = u(s); o[2] = u(r); o[3] = u(e * r);
    // RMSNorm row.
    float q = x * x;
    for (int k = 16; k > 0; k >>= 1) q += __shfl_xor_sync(0xffffffff, q, k);
    float rs = rsq(__fmaf_rn(q, 0.03125f, 1e-5f));
    o[4] = u(rs); o[5] = u(x * rs);
    // Raw bits through every unit.
    o[6] = u(ex2(f(b))); o[7] = u(rcp(f(c))); o[8] = u(rsq(f(d))); o[9] = u(lg2(f(b ^ c)));
    o[10] = u(sqr(f(c ^ d)));
    o[11] = u(ex2(f((b & 0x80ffffff) | 0x42000000)));   // +-[32, 128): overflow / underflow
    o[12] = u(lg2(f(d & 0x7fffffff)));
    o[13] = u(rsq(f(a & 0x7fffffff)));
    uint32_t h;
    asm("ex2.approx.ftz.bf16x2 %0, %1;" : "=r"(h) : "r"(b));
    o[14] = h;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(h) : "r"(c));
    o[15] = h;
    asm("{.reg .b16 l, k; mov.b32 {l, k}, %1; ex2.approx.f16 k, k; mov.b32 %0, {l, k};}" : "=r"(h) : "r"(d));
    o[16] = h;
    asm("{.reg .b16 l, k; mov.b32 {l, k}, %1; ex2.approx.ftz.bf16 l, l; mov.b32 %0, {l, k};}" : "=r"(h) : "r"(a));
    o[17] = h;
    o[18] = u(rcp(f((a & 0x80ffffff) | 0x7e800000)));  // results near the denormal edge
    o[19] = u(sqr(f(b & 0x807fffff)));                  // denormals of both signs
}
