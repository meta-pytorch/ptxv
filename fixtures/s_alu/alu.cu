/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SASS ALU forms of rung 3a (CUTLASS grouped GEMM), checked against the simulator:
// `ptxv hw fixtures/s_alu` compares every output word with the SASS front-end's run.
// nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin alu.cu
#include <cuda_bf16.h>
#include <stdint.h>

#define NOUT 40

extern "C" __global__ void __launch_bounds__(128) alu(const uint64_t* in, uint32_t* out) {
    int t = threadIdx.x;
    uint64_t x = in[2 * t], y = in[2 * t + 1];
    uint32_t a = (uint32_t)x, b = (uint32_t)(x >> 32), c = (uint32_t)y, d = (uint32_t)(y >> 32);
    uint32_t* o = out + t * NOUT;
    uint64_t q = x / (y | 1);              // u64 division: I2F.U64.RP, IMAD.HI, carries, ISETP.EX
    uint64_t r = x % ((y >> 7) | 1);
    o[0] = (uint32_t)q; o[1] = (uint32_t)(q >> 32);
    o[2] = (uint32_t)r; o[3] = (uint32_t)(r >> 32);
    int64_t s = (int64_t)(int32_t)a * (int32_t)c;   // signed IMAD.WIDE
    o[4] = (uint32_t)s; o[5] = (uint32_t)(s >> 32);
    o[6] = __umulhi(a, c) + b;              // IMAD.HI
    uint64_t h = __umul64hi(x, y);
    o[7] = (uint32_t)h; o[8] = (uint32_t)(h >> 32);
    o[9] = (int64_t)x < (int64_t)y ? 1 : 0;   // ISETP.EX signed
    o[10] = x <= y ? 1 : 0;
    o[11] = __brev(a);
    o[12] = __clz(b) + (__ffs(c) << 8);
    float f = __uint_as_float((a & 0x3fffffff) | 0x30000000), g = __uint_as_float((c & 0x3fffffff) | 0x30000000);
    o[13] = __float_as_uint(__fmaf_rn(f, g, __uint_as_float(d & 0xbfffffff)));
    o[14] = __float_as_uint(f * g);
    __nv_bfloat162 p = __floats2bfloat162_rn(f * 3.0f, g);
    o[15] = *(uint32_t*)&p;
    o[16] = __float_as_uint((float)a);      // I2FP.F32.U32
    o[17] = __float_as_uint((float)(int)b);
    o[18] = __shfl_sync(0xffffffff, a, c & 31);
    o[19] = __shfl_up_sync(0xffffffff, b, 3);
    o[20] = __shfl_down_sync(0xffffffff, d, 5) ^ __shfl_xor_sync(0xffffffff, c, 9);
    o[21] = min(a + b, d);                 // VIADDMNMX
    o[22] = __float_as_uint(__ull2float_rn(x));  // I2F.U64
    o[23] = (uint32_t)__float2ull_rz(f * 1e9f);   // F2I.U64.TRUNC
    // 64-bit compares with equal high words (ISETP.EX deferring to the low-word result).
    uint64_t z = (x & 0xffffffff00000000ull) | (t & 1 ? c : a);
    o[24] = ((int64_t)x < (int64_t)z) | ((x >= z) << 1) | ((x == z) << 2) | ((x != y) << 3);
    int64_t w;                              // signed IMAD.WIDE
    asm("mul.wide.s32 %0, %1, %2;" : "=l"(w) : "r"(a), "r"(c));
    o[25] = (uint32_t)w; o[26] = (uint32_t)(w >> 32);
    uint64_t l1 = y + ((uint64_t)a << 7);   // LEA Pc / LEA.HI.X
    uint64_t l2 = y + ((int64_t)(int32_t)b << 4);   // LEA.HI.X.SX32
    o[27] = (uint32_t)l1; o[28] = (uint32_t)(l1 >> 32);
    o[29] = (uint32_t)l2; o[30] = (uint32_t)(l2 >> 32);
    uint64_t s3 = x + y - (q | ((uint64_t)t << 40));   // IADD3 with two carry-outs
    o[31] = (uint32_t)s3; o[32] = (uint32_t)(s3 >> 32);
    o[33] = __shfl_sync(0xffffffff, c, d & 15, 16);   // segmented shuffles
    o[34] = __shfl_up_sync(0xffffffff, a, 2, 8) + __shfl_down_sync(0xffffffff, b, 3, 8);
    o[35] = __shfl_xor_sync(0xffffffff, d, 6, 4);
    float big = __uint_as_float(c), tiny = __uint_as_float(d & 0x807fffff);  // any bits, denormals
    __nv_bfloat162 p2 = __floats2bfloat162_rn(big, tiny);
    o[36] = *(uint32_t*)&p2;
    o[37] = __float_as_uint(__fmaf_rn(tiny, __uint_as_float(0x3f800001), -tiny));
    o[38] = __float_as_uint(__fmul_rn(big, __uint_as_float(a)));
    o[39] = __float_as_uint(__fmaf_rn(big, __uint_as_float(a), __uint_as_float(b)));
}
