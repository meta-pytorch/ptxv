/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SASS f32 ALU forms of rung 4 (softmax / norms / reductions / attention), checked against the
// simulator bitwise: `ptxv hw fixtures/s_alu2` compares every output word with the SASS
// front-end's run. Every pair (x, y) of N operand bit patterns (64 special values: zeros,
// subnormals, normal edges, infinities, NaNs with payloads, rounding ties; 64 random) goes
// through each instruction form (inline PTX pins the form; the SASS each becomes is listed in
// case.toml).
// nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin alu2.cu
#include <stdint.h>

#define N 128
#define NOUT 44

#define F(x) __uint_as_float(x)
#define U(x) __float_as_uint(x)

__device__ __forceinline__ float bin(float a, float b, int op) { return a; }

#define OP2(name, ptx)                                                          \
    __device__ __forceinline__ uint32_t name(uint32_t a, uint32_t b) {          \
        float d;                                                                \
        asm volatile(ptx " %0, %1, %2;" : "=f"(d) : "f"(F(a)), "f"(F(b)));      \
        return U(d);                                                            \
    }
#define OP3(name, ptx)                                                                      \
    __device__ __forceinline__ uint32_t name(uint32_t a, uint32_t b, uint32_t c) {          \
        float d;                                                                            \
        asm volatile(ptx " %0, %1, %2, %3;" : "=f"(d) : "f"(F(a)), "f"(F(b)), "f"(F(c)));   \
        return U(d);                                                                        \
    }
#define SETP(name, ptx)                                                         \
    __device__ __forceinline__ uint32_t name(uint32_t a, uint32_t b) {          \
        uint32_t d;                                                             \
        asm volatile("{ .reg .pred p; " ptx " p, %1, %2; selp.u32 %0, 1, 0, p; }" \
                     : "=r"(d) : "f"(F(a)), "f"(F(b)));                          \
        return d;                                                               \
    }

OP2(add_rn, "add.rn.f32")
OP2(sub_rn, "sub.rn.f32")
OP2(add_rz, "add.rz.f32")
OP2(add_rm, "add.rm.f32")
OP2(add_rp, "add.rp.f32")
OP2(add_ftz, "add.rn.ftz.f32")
OP2(sub_ftz, "sub.rn.ftz.f32")
OP2(mul_rn, "mul.rn.f32")
OP2(mul_rz, "mul.rz.f32")
OP2(mul_rm, "mul.rm.f32")
OP2(mul_rp, "mul.rp.f32")
OP2(mul_ftz, "mul.rn.ftz.f32")
OP2(mul_sat, "mul.rn.sat.f32")
OP2(add_sat, "add.rn.sat.f32")
OP2(add_ftz_sat, "add.rn.ftz.sat.f32")
OP3(fma_rn, "fma.rn.f32")
OP3(fma_rz, "fma.rz.f32")
OP3(fma_rm, "fma.rm.f32")
OP3(fma_rp, "fma.rp.f32")
OP3(fma_ftz, "fma.rn.ftz.f32")
OP3(fma_sat, "fma.rn.sat.f32")
OP3(fma_ftz_sat, "fma.rn.ftz.sat.f32")
OP2(min_f, "min.f32")
OP2(max_f, "max.f32")
OP2(min_ftz, "min.ftz.f32")
OP2(max_ftz, "max.ftz.f32")
OP2(min_nan, "min.NaN.f32")
OP2(max_nan, "max.NaN.f32")
OP3(min3, "min.f32")
OP3(max3, "max.f32")
OP2(div_rn, "div.rn.f32")

SETP(s_eq, "setp.eq.f32") SETP(s_ne, "setp.ne.f32") SETP(s_lt, "setp.lt.f32")
SETP(s_le, "setp.le.f32") SETP(s_gt, "setp.gt.f32") SETP(s_ge, "setp.ge.f32")
SETP(s_equ, "setp.equ.f32") SETP(s_neu, "setp.neu.f32") SETP(s_ltu, "setp.ltu.f32")
SETP(s_leu, "setp.leu.f32") SETP(s_gtu, "setp.gtu.f32") SETP(s_geu, "setp.geu.f32")
SETP(s_num, "setp.num.f32") SETP(s_nan, "setp.nan.f32")
SETP(s_eq_ftz, "setp.eq.ftz.f32") SETP(s_lt_ftz, "setp.lt.ftz.f32")
SETP(s_gtu_ftz, "setp.gtu.ftz.f32") SETP(s_neu_ftz, "setp.neu.ftz.f32")

extern "C" __global__ void __launch_bounds__(128)
alu2(const uint32_t* sp, const uint32_t* rnd, uint32_t* out) {
    int i = threadIdx.x;
    __shared__ uint32_t v[N];
    v[i] = i < 64 ? sp[i] : rnd[i - 64];
    __syncthreads();
    uint32_t x = v[i];
    for (int j = 0; j < N; j++) {
        uint32_t y = v[j], z = v[(i + 3 * j + 1) % N];
        uint32_t* o = out + (j * N + i) * NOUT;
        o[0] = add_rn(x, y);
        o[1] = sub_rn(x, y);
        o[2] = add_rz(x, y);
        o[3] = add_rm(x, y);
        o[4] = add_rp(x, y);
        o[5] = add_ftz(x, y);
        o[6] = sub_ftz(x, y);
        o[7] = add_rn(U(fabsf(F(x))), U(-F(y)));     // FADD |R|, -R
        o[8] = mul_rn(x, y);
        o[9] = mul_rz(x, y);
        o[10] = mul_rm(x, y);
        o[11] = mul_rp(x, y);
        o[12] = mul_ftz(x, y);
        o[13] = mul_sat(x, y);
        o[14] = fma_rn(x, y, z);
        o[15] = fma_rz(x, y, z);
        o[16] = fma_rm(x, y, z);
        o[17] = fma_rp(x, y, z);
        o[18] = fma_ftz(x, y, z);
        o[19] = fma_sat(x, y, z);
        o[20] = fma_ftz_sat(x, y, z);
        o[21] = fma_rn(U(-F(x)), y, U(fabsf(F(z))));   // FFMA -R, R, |R|
        o[22] = min_f(x, y);
        o[23] = max_f(x, y);
        o[24] = min_ftz(x, y);
        o[25] = max_ftz(x, y);
        o[26] = min_nan(x, y);
        o[27] = max_nan(x, y);
        o[28] = min3(x, y, z);
        o[29] = max3(U(fabsf(F(x))), y, z);
        o[30] = div_rn(x, y);
        o[31] = s_eq(x, y) | s_ne(x, y) << 1 | s_lt(x, y) << 2 | s_le(x, y) << 3 |
                s_gt(x, y) << 4 | s_ge(x, y) << 5 | s_equ(x, y) << 6 | s_neu(x, y) << 7 |
                s_ltu(x, y) << 8 | s_leu(x, y) << 9 | s_gtu(x, y) << 10 | s_geu(x, y) << 11 |
                s_num(x, y) << 12 | s_nan(x, y) << 13 | s_eq_ftz(x, y) << 14 |
                s_lt_ftz(x, y) << 15 | s_gtu_ftz(x, y) << 16 | s_neu_ftz(x, y) << 17;
        uint32_t p;
        asm volatile("{ .reg .pred q, r; setp.lt.f32 q, %3, 0f00000000; "
                     "setp.gtu.or.f32 r, %1, %2, q; selp.u32 %0, 1, 0, r; }"
                     : "=r"(p) : "f"(F(x)), "f"(F(y)), "f"(F(z)));
        uint32_t p2;
        asm volatile("{ .reg .pred q, r; setp.lt.f32 q, %3, 0f00000000; "
                     "setp.le.xor.f32 r, %1, %2, q; selp.u32 %0, 1, 0, r; }"
                     : "=r"(p2) : "f"(F(x)), "f"(F(y)), "f"(F(z)));
        o[32] = p | p2 << 1;
        o[33] = (F(x) > F(y)) ? x : z;                  // FSEL / SEL
        uint64_t a2 = (uint64_t)y << 32 | x, b2 = (uint64_t)z << 32 | y, c2 = (uint64_t)x << 32 | z, d2;
        asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d2) : "l"(a2), "l"(b2), "l"(c2));
        o[34] = (uint32_t)d2;
        o[35] = (uint32_t)(d2 >> 32);
        asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(d2) : "l"(a2), "l"(b2));
        o[36] = (uint32_t)d2 ^ (uint32_t)(d2 >> 32);
        o[37] = min(x, y) ^ (uint32_t)max((int)y, (int)z);   // VIMNMX
        uint32_t lt;
        asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lt));
        o[38] = __popc(x ^ z) | (lt << 8);
        // Sub-word loads (LDG.E.U16 / U8) of the operand buffers.
        o[39] = (uint32_t)((const uint16_t*)rnd)[(i + j) & 127] |
                (uint32_t)((const uint8_t*)sp)[(3 * i + j) & 255] << 16;
        o[40] = U((F(x) > F(z)) ? fabsf(F(y)) : -F(z));     // FSEL |R| / -R
        o[41] = add_sat(x, y);
        o[42] = add_ftz_sat(x, z);
        // Sub-word stores (STG.E.U16 / STG.E.U8).
        ((uint16_t*)&o[43])[0] = (uint16_t)(x ^ y);
        ((uint8_t*)&o[43])[2] = (uint8_t)z;
        ((uint8_t*)&o[43])[3] = (uint8_t)(y >> 8);
    }
}
