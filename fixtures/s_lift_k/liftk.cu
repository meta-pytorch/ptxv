/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SASS forms lifted for the compile-vs-eager gaps (Triton bf16 RoPE, cuBLAS gemmSN_NN_kernel,
// CUTLASS fmha_cutlassF_f32), checked against the simulator bitwise: `ptxv hw
// fixtures/s_lift_k` compares every output word with the SASS front-end's run. Every pair
// (x, y) of the s_alu2 operands (64 special bit patterns, 64 random) goes through each form;
// inline PTX pins the form (the SASS each becomes is listed in case.toml):
//   FFMA2 with negated operands, VIMNMX3 (signed / unsigned, min / max), FSET.BF.cmp.AND,
//   FCHK with an f32 immediate (div.rn.f32 by a constant), ATOMS.MIN / .MAX[.S32] on shared
//   words (results unused; dumped after a barrier), LDL.U8 / STL.U8 / .U16 (a local byte
//   array indexed by data).
// nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin liftk.cu
#include <stdint.h>

#define N 128
#define NOUT 16

#define F(x) __uint_as_float(x)
#define U(x) __float_as_uint(x)

// d = a * b + c with operand a, b or c negated by neg.f32 on each half, which ptxas folds into
// the FFMA2 operand (`-R6.F32x2.HI_LO`); `bc`: c's low half in both lanes (`-R16.F32`, as in
// Triton's bf16 RoPE).
#define FFMA2N(name, A, B, C, chi)                                                          \
    __device__ __forceinline__ uint64_t name(uint64_t a, uint64_t b, uint64_t c) {          \
        float r0, r1;                                                                       \
        asm volatile("{ .reg .f32 n0, n1; .reg .b64 A, B, C, D;\n"                          \
                     " neg.f32 n0, %2; neg.f32 n1, %3;\n"                                   \
                     " mov.b64 A, " A "; mov.b64 B, " B "; mov.b64 C, " C ";\n"             \
                     " fma.rn.f32x2 D, A, B, C; mov.b64 {%0, %1}, D; }"                     \
                     : "=f"(r0), "=f"(r1)                                                   \
                     : "f"(F((uint32_t)a)), "f"(F((uint32_t)(a >> 32))),                    \
                       "f"(F((uint32_t)a)), "f"(F((uint32_t)(a >> 32))),                    \
                       "f"(F((uint32_t)b)), "f"(F((uint32_t)(b >> 32))),                    \
                       "f"(F((uint32_t)c)), "f"(F((uint32_t)(chi))));                       \
        return (uint64_t)U(r1) << 32 | U(r0);                                               \
    }
// operands: %2,%3 = a (also negated as n0, n1), %4,%5 = a, %6,%7 = b, %8,%9 = c
FFMA2N(ffma2_nc, "{%4, %5}", "{%6, %7}", "{n0, n1}", c >> 32)
FFMA2N(ffma2_na, "{n0, n1}", "{%6, %7}", "{%8, %9}", c >> 32)
__device__ __forceinline__ uint32_t fset(uint32_t a, uint32_t b) {
    uint32_t d;
    asm volatile("set.gt.f32.f32 %0, %1, %2;" : "=r"(d) : "f"(F(a)), "f"(F(b)));
    return d;
}
__device__ __forceinline__ uint32_t fset_ne(uint32_t a, uint32_t b) {
    uint32_t d;
    asm volatile("set.neu.f32.f32 %0, %1, %2;" : "=r"(d) : "f"(F(a)), "f"(F(b)));
    return d;
}
__device__ __forceinline__ int32_t min3s(int32_t a, int32_t b, int32_t c) { return min(min(a, b), c); }
__device__ __forceinline__ uint32_t min3u(uint32_t a, uint32_t b, uint32_t c) { return min(min(a, b), c); }
__device__ __forceinline__ int32_t max3s(int32_t a, int32_t b, int32_t c) { return max(max(a, b), c); }
__device__ __forceinline__ uint32_t max3u(uint32_t a, uint32_t b, uint32_t c) { return max(max(a, b), c); }

extern "C" __global__ void __launch_bounds__(128)
liftk(const uint32_t* sp, const uint32_t* rnd, uint32_t* out, uint32_t* atoms) {
    int i = threadIdx.x;
    __shared__ uint32_t v[N];
    __shared__ uint32_t am[4 * N];
    v[i] = i < 64 ? sp[i] : rnd[i - 64];
    for (int k = i; k < 4 * N; k += N) am[k] = k & 1 ? 0x80000000u : 0x7fffffffu;
    __syncthreads();
    uint32_t x = v[i];
    // A local byte array indexed by data (LDL.U8 / STL.U8 / LDL.U16 / STL.U16).
    uint8_t loc[32];
    for (int k = 0; k < 32; k++) loc[k] = (uint8_t)(x >> (k & 24));
    for (int j = 0; j < N; j++) {
        uint32_t y = v[j], z = v[(i + 3 * j + 1) % N];
        uint32_t* o = out + (j * N + i) * NOUT;
        uint64_t a2 = (uint64_t)y << 32 | x, b2 = (uint64_t)z << 32 | y, c2 = (uint64_t)x << 32 | z;
        // ffma2_nc(n, x, y) = x * y - n: FFMA2 x, y, -n
        uint64_t d2 = ffma2_nc(c2, a2, b2);              // FFMA2 a, b, -c
        o[0] = (uint32_t)d2;
        o[1] = (uint32_t)(d2 >> 32);
        d2 = ffma2_na(a2, b2, c2);                       // FFMA2 -a, b, c
        o[2] = (uint32_t)d2 ^ (uint32_t)(d2 >> 32);
        d2 = ffma2_nc((uint64_t)z << 32 | z, a2, b2);    // FFMA2 a, b, -z.F32
        o[11] = (uint32_t)d2;
        o[12] = (uint32_t)(d2 >> 32);
        d2 = ffma2_nc(b2, c2, a2) ^ ffma2_na(c2, b2, a2); // more operand orders
        o[13] = (uint32_t)d2 ^ (uint32_t)(d2 >> 32);
        o[3] = min3s(x, y, z);
        o[4] = min3u(x, y, z);
        o[5] = max3s(x, y, z) ^ max3u(z, x, y);
        o[6] = fset(x, y) ^ fset_ne(y, z) << 1;
        o[7] = U(F(x) / 1.4426950216293334961f);         // FCHK with an f32 immediate
        o[8] = U(F(y) / 3.0f);
        uint32_t w = j * N + i;
        // Shared atomics: CUTLASS's float max (MAX.S32 if sign clear, else MIN.U32) and the
        // plain integer forms.
        if ((int32_t)y >= 0)
            asm volatile("atom.shared.max.s32 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&am[(i + j) & 15])), "r"(y) : "memory");
        else
            asm volatile("atom.shared.min.u32 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&am[(i + j) & 15])), "r"(y) : "memory");
        asm volatile("atom.shared.min.s32 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&am[N + ((i ^ j) & 31)])), "r"(z) : "memory");
        asm volatile("atom.shared.max.u32 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&am[2 * N + (i & 63)])), "r"(x ^ y) : "memory");
        asm volatile("atom.shared.max.s32 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&am[3 * N + (w % N)])), "r"(z ^ w) : "memory");
        // Local bytes / halves.
        loc[(y ^ j) & 31] = (uint8_t)(z + j);
        ((uint16_t*)loc)[(z >> 3) & 15] ^= (uint16_t)(x + y);
        o[9] = loc[(x + j) & 31] | (uint32_t)loc[(z ^ y) & 31] << 8 |
               (uint32_t)((uint16_t*)loc)[(y + i) & 15] << 16;
        o[10] = 0;
        for (int k = 0; k < 7; k++) o[10] ^= (uint32_t)loc[(k * 5 + j) & 31] << (k * 4);
        o[14] = o[15] = 0;
    }
    __syncthreads();
    for (int k = i; k < 4 * N; k += N) atoms[k] = am[k];
}
