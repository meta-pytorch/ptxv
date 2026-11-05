/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Row maxima of X[32 x 128] f32 (N-FMAX, docs/equiv-mma.md §6): Y[r] = max_j X[r][j].
//   rowmax_seq     one thread per row, fmaxf over j in order (the reference);
//   rowmax_atomic  four warps, one per 32 columns, each lane a row: fmaxf over its columns,
//                  then CUTLASS's atomicMaxFloat into shared memory (signed atomicMax for a
//                  value with the sign bit clear, unsigned atomicMin otherwise; the shared
//                  maximum starts at -inf);
//   rowmax_atomic_bad  the same with the choice made by value > 0.5 (wrong for values in
//                  (0, 0.5]: a data-dependent branch that must block).
//   nvcc -cubin -arch=sm_103a -O3 -o fmax.cubin fmax.cu   (each c_fmax_* case: kernel.cubin)
#include <math.h>
#define R 32
#define C 128

__device__ __forceinline__ void atomic_max_float(float* addr, float v) {
    if (!signbit(v)) atomicMax((int*)addr, __float_as_int(v));
    else atomicMin((unsigned int*)addr, __float_as_uint(v));
}

__device__ __forceinline__ void atomic_max_float_bad(float* addr, float v) {
    if (v > 0.5f) atomicMax((int*)addr, __float_as_int(v));
    else atomicMin((unsigned int*)addr, __float_as_uint(v));
}

template <bool BAD>
__device__ __forceinline__ void rowmax_atomic_t(const float* __restrict__ X, float* __restrict__ Y) {
    __shared__ float m[R];
    int tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    if (tid < R) m[tid] = -INFINITY;
    __syncthreads();
    float mx = -INFINITY;
#pragma unroll 8
    for (int j = 0; j < 32; j++) mx = fmaxf(mx, X[lane * C + w * 32 + j]);
    if (BAD) atomic_max_float_bad(&m[lane], mx);
    else atomic_max_float(&m[lane], mx);
    __syncthreads();
    if (tid < R) Y[tid] = m[tid];
}

extern "C" __global__ void __launch_bounds__(128)
rowmax_atomic(const float* X, float* Y) { rowmax_atomic_t<false>(X, Y); }
extern "C" __global__ void __launch_bounds__(128)
rowmax_atomic_bad(const float* X, float* Y) { rowmax_atomic_t<true>(X, Y); }

extern "C" __global__ void __launch_bounds__(32)
rowmax_seq(const float* __restrict__ X, float* __restrict__ Y) {
    int r = threadIdx.x;
    float mx = -INFINITY;
#pragma unroll 8
    for (int j = 0; j < C; j++) mx = fmaxf(mx, X[r * C + j]);
    Y[r] = mx;
}
