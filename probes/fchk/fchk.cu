/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// FCHK probe (N-FCHK). ptxas emits `FCHK P, a, b` only inside its div.rn.f32 sequence, where
// the predicate is not observable. This kernel computes `setp.gtu.f32 p, a, b` for every pair
// of the s_alu2 operands and stores p; patch.py rewrites that FSETP into `FCHK P, a, b` (same
// registers, write scoreboard + a wait on it in the consumer), so the output is FCHK's predicate.
//   nvcc -cubin -arch=sm_103a -O3 -o fchk.cubin fchk.cu && python3 patch.py fchk.cubin out.cubin
//   ptxv hw-worker probes/fchk out.cubin --runs 1 --dump fchk.bin
#include <stdint.h>
#define N 128
extern "C" __global__ void __launch_bounds__(128)
fchk(const uint32_t* sp, const uint32_t* rnd, uint32_t* out) {
    int i = threadIdx.x;
    __shared__ uint32_t v[N];
    v[i] = i < 64 ? sp[i] : rnd[i - 64];
    __syncthreads();
    float x = __uint_as_float(v[i]);
#pragma unroll 1
    for (int j = 0; j < N; j++) {
        uint32_t p;
        asm volatile("{ .reg .pred q; setp.gtu.f32 q, %1, %2; selp.u32 %0, 1, 0, q; }"
                     : "=r"(p) : "f"(x), "f"(__uint_as_float(v[j])));
        out[j * N + i] = p;
    }
}
