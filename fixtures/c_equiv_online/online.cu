/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Online-softmax bookkeeping and exact taint (docs/equiv-mma.md §7): X[32 x 128] f32, four
// warps, warp w the columns 32w..32w+31, lane the row.
//   online_ok        row maximum by CUTLASS's atomicMaxFloat into shared memory (m_prime starts
//                    at -inf, mi the shared maximum), the first rescaling exp2f(m_prime - mi)
//                    under `m_prime < mi`, a causal mask by index (FSEL of -inf), exp2f(x - mi)
//                    (non-ftz: range guards), a guard on the masked value, a shuffle from lanes
//                    holding constants, and a float2 store whose second word is a constant
//                    (each decides whether F is written: input-independent here);
//                    Y[r] = s_prime + the four warps' sums.
//   online_bad_mp    m_prime starts at -1e30 (finite): `m_prime < mi` is data (N-FINITE does
//                    not decide it).
//   online_bad_sel   the mask chosen by x > 0: the selected -inf is input data.
//   online_bad_shfl  even lanes test a value shuffled from odd lanes holding data.
//   online_bad_word  the guard reads the float2's data word.
//   nvcc -cubin -arch=sm_103a -O3 -o online.cubin online.cu   (each c_online_* case)
#include <math.h>
#define R 32
#define C 128

__device__ __forceinline__ void atomic_max_float(float* addr, float v) {
    if (!signbit(v)) atomicMax((int*)addr, __float_as_int(v));
    else atomicMin((unsigned int*)addr, __float_as_uint(v));
}

template <int BAD>
__device__ __forceinline__ void online_t(const float* __restrict__ X, float* __restrict__ Y,
                                         float* __restrict__ F) {
    __shared__ float mi[R], mp[R], sp[R], part[4][R];
    __shared__ float2 pair[128];
    int tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    if (tid < R) {
        mp[tid] = BAD == 1 ? -1e30f : -INFINITY;
        mi[tid] = -INFINITY;
        sp[tid] = 0.f;
    }
    __syncthreads();
    float x[32];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        int col = w * 32 + j;
        float v = X[lane * C + col];
        bool masked = BAD == 2 ? v > 0.f : col > 4 * lane;
        x[j] = masked ? -INFINITY : v;
        mx = fmaxf(mx, x[j]);
    }
    atomic_max_float(&mi[lane], mx);
    __syncthreads();
    if (tid < R) {
        if (mp[tid] < mi[tid]) {
            float r = exp2f(mp[tid] - mi[tid]);
            sp[tid] *= r;
            mp[tid] = mi[tid];
        }
    }
    __syncthreads();
    float m = mi[lane];
    float s = 0.f;
    int flags = 0;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        s += exp2f(x[j] - m);
        if (x[j] == -INFINITY) flags++;
    }
    // A value from the neighbouring lane: data in odd lanes (bad) or constants in odd lanes.
    float give = (lane & 1) ? (BAD == 3 ? s : 1.0f) : s;
    float got = __shfl_xor_sync(0xffffffffu, give, 1);
    if (!(lane & 1) && got > 0.5f) flags += 64;
    pair[tid] = make_float2(s, 2.0f);
    __syncwarp();
    float2 q = pair[tid ^ 1];
    if ((BAD == 4 ? q.x : q.y) > 1.5f) flags += 128;
    if (flags & 0x1c0) F[tid] = 1.0f;
    part[w][lane] = s;
    __syncthreads();
    if (tid < R) Y[tid] = sp[tid] + part[0][tid] + part[1][tid] + part[2][tid] + part[3][tid];
}

extern "C" __global__ void __launch_bounds__(128)
online_ok(const float* X, float* Y, float* F) { online_t<0>(X, Y, F); }
extern "C" __global__ void __launch_bounds__(128)
online_bad_mp(const float* X, float* Y, float* F) { online_t<1>(X, Y, F); }
extern "C" __global__ void __launch_bounds__(128)
online_bad_sel(const float* X, float* Y, float* F) { online_t<2>(X, Y, F); }
extern "C" __global__ void __launch_bounds__(128)
online_bad_shfl(const float* X, float* Y, float* F) { online_t<3>(X, Y, F); }
extern "C" __global__ void __launch_bounds__(128)
online_bad_word(const float* X, float* Y, float* F) { online_t<4>(X, Y, F); }
