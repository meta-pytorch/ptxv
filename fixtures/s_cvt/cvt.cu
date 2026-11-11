/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Conversions of torch's dynamic-cast copy kernels against GB300 bitwise: thread i, operand
// x = v[i] (the s_alu2 operands). Per thread, 16 + 3*128 words:
//   F2F.F64.F32, F2I.S64.TRUNC / F2I.U64.TRUNC (64-bit results), HADD2.F32 of both halves,
//   F2F.F16.F32, F2FP.F16.F32.PACK_AB, I2F.S16 / I2F.S8 / I2F.U16 of the low bits, LDG.E.S8 / S16;
// and for every j: F2F.F32.F64 of the f64 {v[j]:x}, I2F.S64 / I2F.U64 of the 64-bit {v[j]:x}.
//   nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin cvt.cu
#include <stdint.h>
#include <cuda_fp16.h>
#define N 128
#define W (16 + 3 * N)
extern "C" __global__ void __launch_bounds__(N)
cvt(const uint32_t* __restrict__ sp, const uint32_t* __restrict__ rnd, uint32_t* __restrict__ out) {
    __shared__ uint32_t v[N];
    int i = threadIdx.x;
    uint32_t xb = i < 64 ? sp[i] : rnd[i - 64];
    v[i] = xb;
    __syncthreads();
    uint32_t* o = out + i * W;
    float x = __uint_as_float(xb);
    double dx = (double)x;
    o[0] = (uint32_t)__double_as_longlong(dx);
    o[1] = (uint32_t)(__double_as_longlong(dx) >> 32);
    long long s64 = (long long)x;
    o[2] = (uint32_t)s64;
    o[3] = (uint32_t)(s64 >> 32);
    unsigned long long u64 = (unsigned long long)x;
    o[4] = (uint32_t)u64;
    o[5] = (uint32_t)(u64 >> 32);
    o[6] = __float_as_uint(__half2float(__ushort_as_half((unsigned short)(xb & 0xffff))));
    o[7] = __float_as_uint(__half2float(__ushort_as_half((unsigned short)(xb >> 16))));
    o[8] = (uint32_t)__half_as_ushort(__float2half_rn(x));
    o[9] = __float_as_uint((float)(short)(xb & 0xffff));
    o[10] = __float_as_uint((float)(signed char)(xb & 0xff));
    o[11] = __float_as_uint((float)(unsigned short)(xb >> 16));
    const signed char* sb = (const signed char*)sp;
    const short* ss = (const short*)rnd;
    int t12, t13;
    asm volatile("ld.global.s8 %0, [%1];" : "=r"(t12) : "l"(sb + (i * 7) % 256));
    asm volatile("ld.global.s16 %0, [%1];" : "=r"(t13) : "l"(ss + (i * 5) % 128));
    o[12] = (uint32_t)t12;
    o[13] = (uint32_t)t13;
    o[14] = __float_as_uint((float)(int)sb[(i * 3) % 256]);
    __half2 hp = __floats2half2_rn(x, 0.f);
    o[15] = *(uint32_t*)&hp;
#pragma unroll 1
    for (int j = 0; j < N; j++) {
        unsigned long long w = ((unsigned long long)v[j] << 32) | xb;
        o[16 + j] = __float_as_uint((float)__longlong_as_double(w));
        o[16 + N + j] = __float_as_uint((float)(long long)w);
        o[16 + 2 * N + j] = __float_as_uint((float)w);
    }
}
