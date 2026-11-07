/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Module data globals (`__device__` variables) across launches of one module (S-GLOBAL-INIT).
// `counter` is zero-initialized (.nv.global), `table` initialized (.nv.global.init). The case
// launches bump, bump, readout: each bump adds 1 to `counter` per thread (64) and sets
// table[i] = table[i] * 10 + i (threads 0..7); readout writes out[0] = counter and
// out[1 + i] = table[i]. Kernels reach both through relocated words in constant bank 4.
//   nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin globals.cu
__device__ unsigned counter;
__device__ unsigned table[8] = {3, 1, 4, 1, 5, 9, 2, 6};

extern "C" __global__ void bump(unsigned* out) {
    unsigned t = threadIdx.x;
    atomicAdd(&counter, 1u);
    if (t < 8) {
        table[t] = table[t] * 10 + t;
    }
}

extern "C" __global__ void readout(unsigned* out) {
    unsigned t = threadIdx.x;
    if (t == 0) {
        out[0] = counter;
    } else if (t <= 8) {
        out[t] = table[t - 1];
    }
}
