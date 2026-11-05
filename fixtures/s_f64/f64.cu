/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// f64 ALU forms against GB300 bitwise (s_alu2-style): 128 threads x 128 operands (64 special
// bit patterns + 64 random); thread i, x = v[i]; for every j, y = v[j], z = v[(i + j) % 128],
// 11 64-bit words:
//   DFMA / DFMA.RZ / .RM / .RP (x*y + z), DMUL (x*y) / DMUL.RP, DADD (x + y) / DADD.RM (x - y)
// then a word of DSETP predicates (bit k = comparison k of x, y), fmin, fmax (DSETP.MIN/MAX).
//   nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin f64.cu
#include <stdint.h>
#define N 128
extern "C" __global__ void __launch_bounds__(N)
f64(const unsigned long long* __restrict__ vin, unsigned long long* __restrict__ out) {
    __shared__ double v[N];
    int i = threadIdx.x;
    v[i] = __longlong_as_double(vin[i]);
    __syncthreads();
    double x = v[i];
#pragma unroll 1
    for (int j = 0; j < N; j++) {
        double y = v[j], z = v[(i + j) % N];
        unsigned long long* p = out + ((size_t)i * N + j) * 11;
        p[0] = __double_as_longlong(__fma_rn(x, y, z));
        p[1] = __double_as_longlong(__fma_rz(x, y, z));
        p[2] = __double_as_longlong(__fma_rd(x, -y, z));
        p[3] = __double_as_longlong(__fma_ru(-x, y, -z));
        p[4] = __double_as_longlong(__dmul_rn(x, y));
        p[5] = __double_as_longlong(__dmul_ru(x, y));
        p[6] = __double_as_longlong(__dadd_rn(x, y));
        p[7] = __double_as_longlong(__dadd_rd(x, -y));
        unsigned c = 0;
        c |= (x < y) << 0;
        c |= (x <= y) << 1;
        c |= (x == y) << 2;
        c |= (x != y) << 3;
        c |= (x > y) << 4;
        c |= (x >= y) << 5;
        c |= (!(x >= y)) << 6;
        c |= (!(x < y)) << 7;
        c |= (x != x || y != y) << 8;
        c |= (fabs(x) < fabs(y)) << 9;
        p[8] = c;
        p[9] = __double_as_longlong(fmin(x, y));
        p[10] = __double_as_longlong(fmax(x, y));
    }
}
