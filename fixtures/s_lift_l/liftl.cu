/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SASS forms lifted for real CuTeDSL kernels (FA4 / quack softmax / mslk decode / block-sparsity),
// checked against GB300 bitwise: `ptxv hw fixtures/s_lift_l` compares every output word with
// the SASS front-end's run. 128 threads; every operand of the s_alu2 set (64 special bit
// patterns, 64 random) goes through each form:
//   CREDUX.{MIN,MAX}[ABS].F32[.NAN] (redux.sync.{min,max}[.abs][.NaN].f32) over the full warp,
//   the even lanes and a single lane; PLOP3 with a second predicate output (all / any of a
//   run of compares, as the block-sparsity kernel); R2P PR, Rx.B1..B3 (bit tests of a mask word);
//   LEA.HI with a negated operand (torch's fp8 e4m3 -> f32 software conversion), signed
//   remainders / quotients by powers of two.
// nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin liftl.cu
#include <stdint.h>

#define N 128
#define NOUT 16

#define F(x) __uint_as_float(x)
#define U(x) __float_as_uint(x)

#define REDUX(name, op)                                                                      \
    __device__ __forceinline__ uint32_t name(uint32_t x, uint32_t m) {                      \
        float r;                                                                             \
        asm volatile("redux.sync." op ".f32 %0, %1, %2;" : "=f"(r) : "f"(F(x)), "r"(m));    \
        return U(r);                                                                         \
    }
REDUX(rmax, "max")
REDUX(rmin, "min")
REDUX(rmaxn, "max.NaN")
REDUX(rminn, "min.NaN")
REDUX(rmaxa, "max.abs")
REDUX(rmina, "min.abs")
REDUX(rmaxan, "max.abs.NaN")
REDUX(rminan, "min.abs.NaN")

// The block-sparsity kernel's all / any of a run of compares: PLOP3.LUT Pa, Pb, !P0, PT, !P1,
// 0x80, 0xc8 (two predicate outputs), then ISETP.EQ.U32.AND / .OR into each.
__device__ __forceinline__ uint32_t allany(uint32_t a, uint32_t b, uint32_t c, uint32_t x) {
    uint32_t r;
    asm volatile("{ .reg .pred e0, e1, e2, A, O; .reg .b32 t;\n"
                 " setp.eq.u32 e0, %1, %4; setp.eq.u32 e1, %2, %4; setp.eq.u32 e2, %3, %4;\n"
                 " and.pred A, e0, e1; or.pred O, e0, e1; and.pred A, A, e2; or.pred O, O, e2;\n"
                 " selp.u32 %0, 1, 0, A; selp.u32 t, 2, 0, O; or.b32 %0, %0, t; }"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c), "r"(x));
    return r;
}
// c10's fp8 e4m3fn -> f32 (torch's software path): LEA.HI d, a, -b, RZ, 0x1c.
__device__ __forceinline__ uint32_t e4m3_to_f32(uint32_t in) {
    uint32_t w = in << 24, sign = w & 0x80000000u, nonsign = w & 0x7fffffffu;
    uint32_t sh = __clz(nonsign);
    sh = sh > 4 ? sh - 4 : 0;
    uint32_t inf_nan = (uint32_t)(((int32_t)(nonsign + 0x01000000) >> 8) & 0x7f800000);
    uint32_t zero = (uint32_t)((int32_t)(nonsign - 1) >> 31);
    return sign | ((((nonsign << sh >> 4) + ((0x78 - sh) << 23)) | inf_nan) & ~zero);
}

extern "C" __global__ void __launch_bounds__(128)
liftl(const uint32_t* sp, const uint32_t* rnd, uint32_t* out) {
    int i = threadIdx.x;
    int lane = i & 31;
    __shared__ uint32_t v[N];
    v[i] = i < 64 ? sp[i] : rnd[i - 64];
    __syncthreads();
    uint32_t x = v[i];
    for (int j = 0; j < N; j++) {
        uint32_t y = v[j], z = v[(i * 5 + 3 * j + 1) % N];
        uint32_t* o = out + (j * N + i) * NOUT;
        // a warp-wide operand: lane l takes v[(l * 7 + j) % N] or z
        uint32_t w = (j & 1) ? v[(lane * 7 + j) % N] : z;
        o[0] = rmax(w, 0xffffffffu);
        o[1] = rmin(w, 0xffffffffu);
        o[2] = rmaxn(w, 0xffffffffu);
        o[3] = rminn(w, 0xffffffffu);
        o[4] = rmaxa(w, 0xffffffffu) ^ rmina(w, 0xffffffffu) << 1;
        o[5] = rmaxan(w, 0xffffffffu);
        o[6] = rminan(w, 0xffffffffu);
        uint32_t e = 0, s = 0;
        if ((lane & 1) == 0) {
            e = rmax(w, 0x55555555u) ^ rminn(w, 0x55555555u) * 3u ^ rmaxa(w, 0x55555555u) * 7u;
        }
        if (lane == (j & 31)) {
            s = rmax(w, 1u << lane) ^ rmin(w, 1u << lane) * 3u ^ rmaxn(w, 1u << lane) * 5u ^
                rmina(w, 1u << lane) * 9u ^ rminan(w, 1u << lane) * 11u;
        }
        o[7] = e;
        o[8] = s;
        // all / any of three compares of 2-bit fields (every combination occurs).
        o[9] = allany(x & 3, y & 3, z & 3, (x ^ y ^ j) & 3) | allany(y & 1, z & 1, x & 1, j & 1) << 2 |
               allany((x >> 2) & 3, (y >> 2) & 3, (z >> 2) & 3, (i ^ j) & 3) << 4;
        // bit tests of a mask word (R2P PR, Rm.Bk, 0x7f).
        uint32_t m = x ^ y, acc = 0;
#pragma unroll
        for (int k = 0; k < 32; k++) {
            if ((m >> k) & 1u) acc += v[(k * 3 + j) & (N - 1)] ^ (uint32_t)k;
        }
        o[10] = acc;
        // signed remainder / quotient by powers of two (LEA.HI with a negated operand).
        int32_t sx = (int32_t)x, sy = (int32_t)y;
        o[11] = (uint32_t)(sx % 16) ^ (uint32_t)(sy % 32) << 8 ^ (uint32_t)(sx / 8) << 3;
        o[12] = (uint32_t)((sx - sy) % 64) ^ (uint32_t)((sx + 7) / 32);
        o[13] = e4m3_to_f32(x) ^ e4m3_to_f32(y >> 8) * 3u ^ e4m3_to_f32(j) * 5u;
        o[14] = o[15] = 0;
    }
}
