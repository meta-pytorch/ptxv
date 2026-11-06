/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// F2I forms with a rounding other than .TRUNC (libdevice's sinf / cosf range reduction:
// cvt.rni.ftz.s32.f32 = F2I.FTZ.NTZ), against GB300 bitwise: `ptxv hw fixtures/s_f2i`.
// 128 threads x 128 operands: the s_alu2 set (64 special patterns, 64 random) and, per pair,
// x, x * 2^-k and k + 0.5 style ties; each through
//   F2I.NTZ, F2I.FTZ.NTZ, F2I.U32.NTZ, F2I.FTZ.U32.NTZ, F2I.FLOOR.NTZ, F2I.CEIL.NTZ,
//   F2I.FTZ.FLOOR.NTZ, F2I.FTZ.TRUNC.NTZ, F2I.S64 (rni.s64), F2I.U64 (rpi.u64);
// and I2F.F64.S64 / .U64 (RN, RZ, RP, RM) and I2F.F64 (s32) on 64-bit integers of every size.
// nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin f2i.cu
#include <stdint.h>
#define N 128
#define NOUT 24
#define F(x) __uint_as_float(x)
#define CVT32(name, m)                                                                       \
    __device__ __forceinline__ uint32_t name(float x) {                                     \
        uint32_t r;                                                                          \
        asm volatile("cvt." m ".f32 %0, %1;" : "=r"(r) : "f"(x));                            \
        return r;                                                                            \
    }
#define CVT64(name, m)                                                                       \
    __device__ __forceinline__ uint64_t name(float x) {                                     \
        uint64_t r;                                                                          \
        asm volatile("cvt." m ".f32 %0, %1;" : "=l"(r) : "f"(x));                            \
        return r;                                                                            \
    }
CVT32(rni, "rni.s32")
CVT32(rni_ftz, "rni.ftz.s32")
CVT32(rni_u, "rni.u32")
CVT32(rni_ftz_u, "rni.ftz.u32")
CVT32(rmi, "rmi.s32")
CVT32(rpi, "rpi.s32")
CVT32(rmi_ftz, "rmi.ftz.s32")
CVT32(rzi_ftz, "rzi.ftz.s32")
CVT64(rni64, "rni.s64")
CVT64(rpi64u, "rpi.u64")
#define I2D(name, m, T, C)                                                                   \
    __device__ __forceinline__ uint64_t name(T x) {                                         \
        double r;                                                                            \
        asm volatile("cvt." m " %0, %1;" : "=d"(r) : C(x));                                  \
        return __double_as_longlong(r);                                                      \
    }
I2D(i2d_s64, "rn.f64.s64", int64_t, "l")
I2D(i2d_u64, "rn.f64.u64", uint64_t, "l")
I2D(i2d_s64_rz, "rz.f64.s64", int64_t, "l")
I2D(i2d_u64_rp, "rp.f64.u64", uint64_t, "l")
I2D(i2d_s64_rm, "rm.f64.s64", int64_t, "l")
I2D(i2d_s32, "rn.f64.s32", int32_t, "r")

extern "C" __global__ void __launch_bounds__(128)
f2i(const uint32_t* sp, const uint32_t* rnd, uint32_t* out) {
    int i = threadIdx.x;
    __shared__ uint32_t v[N];
    v[i] = i < 64 ? sp[i] : rnd[i - 64];
    __syncthreads();
    for (int j = 0; j < N; j++) {
        uint32_t* o = out + (j * N + i) * NOUT;
        // operand: v[i] scaled by 2^-(j % 32) (brings large values to the tie range), or a
        // tie k + 0.5 (signed) when j is a multiple of 8.
        float x = F(v[i]) * __uint_as_float((uint32_t)(127 - (j & 31)) << 23);
        if ((j & 7) == 0) x = (float)((int)(v[i] & 0xff) - 128) + 0.5f;
        o[0] = rni(x);
        o[1] = rni_ftz(x);
        o[2] = rni_u(x);
        o[3] = rni_ftz_u(x);
        o[4] = rmi(x);
        o[5] = rpi(x);
        o[6] = rmi_ftz(x);
        o[7] = rzi_ftz(x);
        uint64_t a = rni64(x), b = rpi64u(x);
        o[8] = (uint32_t)a;
        o[9] = (uint32_t)(a >> 32);
        o[10] = (uint32_t)b;
        o[11] = (uint32_t)(b >> 32);
        o[12] = rni(F(v[i]));
        o[13] = rmi(F(v[i]));
        o[14] = rpi(F(v[i]));
        o[15] = rni_ftz(F(v[j]));
        // I2F.F64.* (libdevice sinf's large-argument reduction): 64-bit integers of every
        // magnitude (v[i]:v[j] shifted right by j % 64) through each rounding.
        uint64_t w = ((uint64_t)v[i] << 32 | v[j]) >> (j & 63);
        uint64_t r;
        r = i2d_s64((int64_t)w); o[16] = (uint32_t)r; o[17] = (uint32_t)(r >> 32);
        r = i2d_u64(w) ^ i2d_s64_rz((int64_t)w); o[18] = (uint32_t)r; o[19] = (uint32_t)(r >> 32);
        r = i2d_u64_rp(w) ^ (i2d_s64_rm((int64_t)w) << 1); o[20] = (uint32_t)r; o[21] = (uint32_t)(r >> 32);
        r = i2d_s32((int32_t)v[i]); o[22] = (uint32_t)r; o[23] = (uint32_t)(r >> 32);
    }
}
