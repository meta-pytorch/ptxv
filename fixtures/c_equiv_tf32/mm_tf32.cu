/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TF32 GEMM pair (N-TF32, docs/equiv-mma.md §6): D[32 x 32] f32 = A[32 x 64] * B[64 x 32], row
// major, as an f32 SIMT loop and on TF32 tensor cores (mma.sync m16n8k8 tf32, SASS
// HMMA.1688.F32.TF32; one warp per 16x8 tile, 8 K steps accumulating in registers):
//   mm_f32_simt  one thread per output, FFMA over k in order (the reference);
//   mm_1xtf32    raw f32 words as operands (the MMA truncates them to TF32);
//   mm_3xtf32    CUTLASS's big + small split (OpMultiplyAddFastF32): big = x & 0xffffe000,
//                small = x - big rounded half away (round_half_ulp_truncate: +0x1000 if
//                finite), then small*big, big*small, big*big per K step;
//   mm_3xtf32_rev  the same, K steps in reverse order;
//   mm_2xtf32    big*small and big*big only (a dropped correction: must not get the 3xTF32
//                bound);
//   mm_3xtf32_sign  rounds small only when small > 0 (a truly data-dependent guard: blocks);
//   mm_3xtf32_bits  rounds small when the float with x's bits shifted left is finite (a
//                finiteness test of a value finite inputs do not make finite: blocks).
//   nvcc -cubin -arch=sm_103a -O3 -o mm_tf32.cubin mm_tf32.cu   (each c_tf32_mm_* case: kernel.cubin)
#include <stdint.h>
#define M 32
#define N 32
#define K 64

__device__ __forceinline__ void mma_tf32(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// SPLIT 0: round small if finite (CUTLASS); 1: if small > 0; 2: if x << 1 is finite.
template <int SPLIT>
__device__ __forceinline__ void split(uint32_t x, uint32_t& big, uint32_t& small) {
    big = x & 0xffffe000u;
    float s = __uint_as_float(x) - __uint_as_float(big);
    uint32_t sb = __float_as_uint(s);
    bool r = SPLIT == 0 ? isfinite(s) : SPLIT == 1 ? s > 0.f : isfinite(__uint_as_float(x << 1));
    if (r) sb += 0x1000u;
    small = sb;
}

// MODE 0: 1xTF32; 1: 3xTF32; 2: 3xTF32, K reversed; 3: 2xTF32 (no small*big).
template <int MODE, int SPLIT = 0>
__device__ __forceinline__ void mm_tc(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B,
                                      float* __restrict__ D) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    int tm = w / (N / 8), tn = w % (N / 8);
    if (tm >= M / 16) return;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kk = 0; kk < K / 8; kk++) {
        int kc = MODE == 2 ? K / 8 - 1 - kk : kk;
        int r0 = tm * 16 + g, k0 = kc * 8 + t, n0 = tn * 8 + g;
        uint32_t a[4] = {A[r0 * K + k0], A[(r0 + 8) * K + k0], A[r0 * K + k0 + 4],
                         A[(r0 + 8) * K + k0 + 4]};
        uint32_t b[2] = {B[k0 * N + n0], B[(k0 + 4) * N + n0]};
        if (MODE == 0) {
            mma_tf32(c, a, b);
        } else {
            uint32_t ab[4], as[4], bb[2], bs[2];
#pragma unroll
            for (int i = 0; i < 4; i++) split<SPLIT>(a[i], ab[i], as[i]);
#pragma unroll
            for (int i = 0; i < 2; i++) split<SPLIT>(b[i], bb[i], bs[i]);
            if (MODE != 3) mma_tf32(c, as, bb);
            mma_tf32(c, ab, bs);
            mma_tf32(c, ab, bb);
        }
    }
    int r = tm * 16 + g, col = tn * 8 + 2 * t;
    D[r * N + col] = c[0];
    D[r * N + col + 1] = c[1];
    D[(r + 8) * N + col] = c[2];
    D[(r + 8) * N + col + 1] = c[3];
}

extern "C" __global__ void __launch_bounds__(256)
mm_1xtf32(const uint32_t* A, const uint32_t* B, float* D) { mm_tc<0>(A, B, D); }
extern "C" __global__ void __launch_bounds__(256)
mm_3xtf32(const uint32_t* A, const uint32_t* B, float* D) { mm_tc<1>(A, B, D); }
extern "C" __global__ void __launch_bounds__(256)
mm_3xtf32_rev(const uint32_t* A, const uint32_t* B, float* D) { mm_tc<2>(A, B, D); }
extern "C" __global__ void __launch_bounds__(256)
mm_2xtf32(const uint32_t* A, const uint32_t* B, float* D) { mm_tc<3>(A, B, D); }
extern "C" __global__ void __launch_bounds__(256)
mm_3xtf32_sign(const uint32_t* A, const uint32_t* B, float* D) { mm_tc<1, 1>(A, B, D); }
extern "C" __global__ void __launch_bounds__(256)
mm_3xtf32_bits(const uint32_t* A, const uint32_t* B, float* D) { mm_tc<1, 2>(A, B, D); }

extern "C" __global__ void __launch_bounds__(128)
mm_f32_simt(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    int r = i / N, col = i % N;
    float s = 0.f;
#pragma unroll 8
    for (int k = 0; k < K; k++) s = fmaf(A[r * K + k], B[k * N + col], s);
    D[i] = s;
}
