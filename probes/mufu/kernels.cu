/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// MUFU / PTX approx-op kernels for the exhaustive sweep (see README.md in this directory).
// nvcc -cubin -arch=sm_103a -O3 -o mufu.cubin kernels.cu
// Each f32 kernel evaluates one op on the inputs base, base+1, ... (bit patterns) and writes
// the output bits; f16/bf16 kernels evaluate one op on all 2^16 bit patterns of each half.
#include <stdint.h>

#define F32OP(name, asmstr)                                                              \
    extern "C" __global__ void name(uint32_t base, uint32_t n, uint32_t* out) {          \
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                  \
             i += gridDim.x * blockDim.x) {                                              \
            float x = __uint_as_float(base + i), y;                                      \
            asm(asmstr : "=f"(y) : "f"(x));                                              \
            out[i] = __float_as_uint(y);                                                 \
        }                                                                                \
    }
F32OP(k_ex2, "ex2.approx.f32 %0, %1;")
F32OP(k_ex2_ftz, "ex2.approx.ftz.f32 %0, %1;")
F32OP(k_rcp, "rcp.approx.f32 %0, %1;")
F32OP(k_rcp_ftz, "rcp.approx.ftz.f32 %0, %1;")
F32OP(k_rsq, "rsqrt.approx.f32 %0, %1;")
F32OP(k_rsq_ftz, "rsqrt.approx.ftz.f32 %0, %1;")
F32OP(k_lg2, "lg2.approx.f32 %0, %1;")
F32OP(k_lg2_ftz, "lg2.approx.ftz.f32 %0, %1;")
// sin/cos: ptxas emits FMUL.RZ x, 1/(2 pi) then MUFU.SIN/COS (the same code with and
// without .ftz). In mufu_raw.cubin, patch.py replaces the constant by 1.0, which leaves the
// bare MUFU.SIN/COS of x (FMUL.RZ x, 1.0 is exact, denormals included).
F32OP(k_sin, "sin.approx.f32 %0, %1;")
F32OP(k_cos, "cos.approx.f32 %0, %1;")
F32OP(k_tanh, "tanh.approx.f32 %0, %1;")
F32OP(k_sqrt, "sqrt.approx.f32 %0, %1;")
F32OP(k_sqrt_ftz, "sqrt.approx.ftz.f32 %0, %1;")

// 16-bit forms: input word = (h1 << 16) | h0, both lanes are evaluated.
#define X2OP(name, asmstr)                                                               \
    extern "C" __global__ void name(uint32_t base, uint32_t n, uint32_t* out) {          \
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                  \
             i += gridDim.x * blockDim.x) {                                              \
            uint32_t x = base + i, y;                                                    \
            asm(asmstr : "=r"(y) : "r"(x));                                              \
            out[i] = y;                                                                  \
        }                                                                                \
    }
X2OP(h_ex2_f16x2, "ex2.approx.f16x2 %0, %1;")
X2OP(h_ex2_bf16x2, "ex2.approx.ftz.bf16x2 %0, %1;")
X2OP(h_tanh_f16x2, "tanh.approx.f16x2 %0, %1;")
X2OP(h_tanh_bf16x2, "tanh.approx.bf16x2 %0, %1;")
// Scalar 16-bit forms (MUFU.EX2.F16 etc. on the low half).
#define H1OP(name, asmstr)                                                               \
    extern "C" __global__ void name(uint32_t base, uint32_t n, uint32_t* out) {          \
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                  \
             i += gridDim.x * blockDim.x) {                                              \
            uint16_t x = (uint16_t)(base + i), y;                                        \
            asm(asmstr : "=h"(y) : "h"(x));                                              \
            out[i] = y;                                                                  \
        }                                                                                \
    }
H1OP(h_ex2_f16, "ex2.approx.f16 %0, %1;")
H1OP(h_ex2_bf16, "ex2.approx.ftz.bf16 %0, %1;")
H1OP(h_tanh_f16, "tanh.approx.f16 %0, %1;")
H1OP(h_tanh_bf16, "tanh.approx.bf16 %0, %1;")

// f64 forms: rcp.approx.ftz.f64 / rsqrt.approx.ftz.f64 are MUFU.RCP64H / MUFU.RSQ64H of the
// input's high word, result low word 0. Input i: high word base + i, low word a hash of i
// (it must not matter); output: the result's high word (d_*) or low word (d_*_lo).
#define D64OP(name, asmstr, lo)                                                          \
    extern "C" __global__ void name(uint32_t base, uint32_t n, uint32_t* out) {          \
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                  \
             i += gridDim.x * blockDim.x) {                                              \
            uint32_t xl = (i * 2654435761u) ^ 0x9e3779b9u;                               \
            double x = __hiloint2double((int)(base + i), (int)xl), y;                    \
            asm(asmstr : "=d"(y) : "d"(x));                                              \
            out[i] = lo ? (uint32_t)__double2loint(y) : (uint32_t)__double2hiint(y);     \
        }                                                                                \
    }
D64OP(d_rcp64h, "rcp.approx.ftz.f64 %0, %1;", 0)
D64OP(d_rsq64h, "rsqrt.approx.ftz.f64 %0, %1;", 0)
D64OP(d_rcp64h_lo, "rcp.approx.ftz.f64 %0, %1;", 1)
D64OP(d_rsq64h_lo, "rsqrt.approx.ftz.f64 %0, %1;", 1)
