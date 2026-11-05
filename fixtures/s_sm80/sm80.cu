/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Rung-5 memory forms of flash attention, against GB300 bitwise (s_alu2-style): one CTA of
// 128 threads.
//   cp.async.cg 16 B global -> shared (LDGSTS.E.BYPASS.128), src-size 0 (zero fill), commit
//   groups (LDGDEPBAR) and wait_group 1 / 0 (DEPBAR.LE SB0, n);
//   ldmatrix x1 / x2 / x4, with and without .trans (LDSM.16.M88.N / .MT88.N), from rows at
//   lane-dependent addresses;
//   local memory: a dynamically indexed per-thread array (STL / LDL);
//   a 64-bit pointer decrement by a shifted register (LEA Rd, P, -Ra, Rb, s).
// Output: 64 words per thread.
//   nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin sm80.cu
#include <stdint.h>
#define NT 128
#define NOUT 64

__device__ __forceinline__ void cp16(uint32_t s, const void* g, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(g), "r"(bytes));
}

extern "C" __global__ void __launch_bounds__(NT)
sm80(const uint32_t* __restrict__ in, const uint32_t* __restrict__ idx, uint32_t* __restrict__ out,
     int64_t step) {
    __shared__ __align__(128) uint32_t sm[2][NT * 4];
    int t = threadIdx.x, lane = t % 32, w = t / 32;
    uint32_t* o = out + t * NOUT;
    uint32_t s0 = (uint32_t)__cvta_generic_to_shared(&sm[0][t * 4]);
    uint32_t s1 = (uint32_t)__cvta_generic_to_shared(&sm[1][t * 4]);
    // Group 0: 16 B per thread from a permuted source; group 1: zero fill for odd threads.
    cp16(s0, in + 4 * idx[t], 16);
    asm volatile("cp.async.commit_group;");
    cp16(s1, in + 4 * idx[NT - 1 - t], (t & 1) ? 0 : 16);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    for (int i = 0; i < 4; i++) o[i] = sm[0][(t * 4 + 4 * 37 + i) % (NT * 4)];
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    for (int i = 0; i < 4; i++) o[4 + i] = sm[1][(t * 4 + 4 * 53 + i) % (NT * 4)];
    // ldmatrix: each warp reads 8x8 b16 matrices whose rows sit at lane-dependent addresses.
    const uint32_t* base = &sm[w & 1][0];
    int row = (lane * 5 + w * 3) % 64;  // a row of 16 bytes
    uint32_t a = (uint32_t)__cvta_generic_to_shared(base + 4 * row);
    uint32_t r[16];
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r[0]) : "r"(a));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[1]), "=r"(r[2]) : "r"(a));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]) : "r"(a));
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(r[7]) : "r"(a));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[8]), "=r"(r[9]) : "r"(a));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]) : "r"(a));
    for (int i = 0; i < 14; i++) o[8 + i] = r[i];
    // Local memory: a per-thread array indexed by loaded values (forces STL / LDL).
    uint32_t loc[24];
#pragma unroll
    for (int i = 0; i < 24; i++) loc[i] = in[(t + 7 * i) % (NT * 4)] ^ (i * 0x9e3779b9u);
    for (int i = 0; i < 24; i++) {
        uint32_t j = idx[(t + i) % NT] % 24;
        loc[j] += loc[(j + 5) % 24] * 3 + i;
    }
    for (int i = 0; i < 24; i++) o[22 + i] = loc[(idx[(t * 3 + i) % NT] + i) % 24];
    // 64-bit pointer arithmetic with a subtracted shifted index.
    uint64_t p = (uint64_t)(in + 4 * NT) + ((uint64_t)t << 20);
    for (int i = 0; i < 4; i++) {
        p -= (uint64_t)(idx[(t + i) % NT] + step) << 8;
        uint64_t q = p - (uint64_t)in;  // relative: the allocation address differs per run
        o[46 + 2 * i] = (uint32_t)q;
        o[47 + 2 * i] = (uint32_t)(q >> 32);
    }
    for (int i = 54; i < NOUT; i++) o[i] = r[i % 14] + i;
}
