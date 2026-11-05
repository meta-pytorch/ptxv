/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TF32 HMMA probe (N-HMMA): D[M x N] f32 = A[M x K] * B[N x K]^T through
// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 (SASS HMMA.1688.F32.TF32), one warp per
// 16x8 tile, K/8 steps accumulating in registers (the first from zero). A and B are raw f32
// words passed to the instruction unconverted (the low 13 mantissa bits included), fragments
// loaded straight from global memory in the PTX ISA layout. `ptxv calib-mma probes/tmma`
// scores the MmaModel family (and the tf32 input conversions) against GB300 on it.
//   nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin tmma.cu
#include <stdint.h>
#define M 128
#define N 128
#define K 64
extern "C" __global__ void __launch_bounds__(128)
tmma(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ D) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    int tm = w / (N / 8), tn = w % (N / 8);
    if (tm >= M / 16) return;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
#pragma unroll
    for (int kc = 0; kc < K / 8; kc++) {
        int r0 = tm * 16 + g, k0 = kc * 8 + t;
        uint32_t a0 = A[r0 * K + k0], a1 = A[(r0 + 8) * K + k0];
        uint32_t a2 = A[r0 * K + k0 + 4], a3 = A[(r0 + 8) * K + k0 + 4];
        int n0 = tn * 8 + g;
        uint32_t b0 = B[n0 * K + k0], b1 = B[n0 * K + k0 + 4];
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    int r = tm * 16 + g, col = tn * 8 + 2 * t;
    D[r * N + col] = c0;
    D[r * N + col + 1] = c1;
    D[(r + 8) * N + col] = c2;
    D[(r + 8) * N + col + 1] = c3;
}
