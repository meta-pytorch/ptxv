/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Signed-zero probe for the tensor-core MMA models (N-MMA, N-HMMA, N-MMA-F8): which sign does
// GB300 give an MMA result that is zero, either exactly (all products +-0, exact
// cancellation) or after underflow (a nonzero sum below the f32 range), with C disabled or
// C = +-0? Each entry computes one small tile per CTA from global A / B / C, writes D.
//   tc_f16 / tc_f8 / tc_tf32: tcgen05.mma.cta_group::1 kind::f16 / f8f6f4 / tf32, one MMA
//     (32 bytes of K per row) per CTA of 128 threads. A tile 128 x 32 B, B tile 16 x 32 B,
//     K-major, no swizzle, stored to shared memory with plain stores (+ fence.proxy.async).
//     C (128 TMEM lanes x 16 f32 columns) is written with tcgen05.st, the MMA runs with
//     enable_input_d = `en` and instruction descriptor `idesc` (M128 or M64, N16), D is
//     read back with tcgen05.ld.32x32b (D[tile][lane][col]).
//   hm / hm0 / hmh: mma.sync.m16n8k16 bf16 (f16 for hmh) -> f32, one warp per 16x8 tile, C
//     from memory (hm, hmh) or the constant 0 (hm0: SASS HMMA ... RZ).
//   tm / tm0: mma.sync.m16n8k8 tf32 -> f32 on raw f32 words, C from memory / 0.
// Generated inputs and the comparison with the model: run.py. README.md has the results.
//   nvcc -ptx -arch=sm_103a -O3 -o kernel.ptx zero.cu
#include <stdint.h>

#define TC_KERNEL(NAME, KIND)                                                                    \
    extern "C" __global__ void __launch_bounds__(128)                                            \
        NAME(const uint4* __restrict__ A, const uint4* __restrict__ B,                           \
             const float* __restrict__ C, float* __restrict__ D, unsigned idesc, unsigned en) {  \
        __shared__ __align__(1024) unsigned char sA[4096];                                       \
        __shared__ __align__(1024) unsigned char sB[512];                                        \
        __shared__ __align__(8) unsigned long long bar;                                          \
        __shared__ unsigned tslot;                                                               \
        int tid = threadIdx.x, w = tid / 32, t = blockIdx.x;                                     \
        const uint4* a = A + (t * 128 + tid) * 2;                                                \
        *(uint4*)(sA + (tid / 8) * 128 + (tid % 8) * 16) = a[0];                                 \
        *(uint4*)(sA + 2048 + (tid / 8) * 128 + (tid % 8) * 16) = a[1];                          \
        if (tid < 16) {                                                                          \
            const uint4* b = B + (t * 16 + tid) * 2;                                             \
            *(uint4*)(sB + (tid / 8) * 128 + (tid % 8) * 16) = b[0];                             \
            *(uint4*)(sB + 256 + (tid / 8) * 128 + (tid % 8) * 16) = b[1];                       \
        }                                                                                        \
        const float* c = C + (t * 128 + tid) * 16;                                               \
        unsigned cv[16];                                                                         \
        _Pragma("unroll") for (int j = 0; j < 16; j++) cv[j] = __float_as_uint(c[j]);            \
        unsigned bara = (unsigned)__cvta_generic_to_shared(&bar);                                \
        unsigned slot = (unsigned)__cvta_generic_to_shared(&tslot);                              \
        if (w == 0) {                                                                            \
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"     \
                         ::"r"(slot));                                                           \
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");          \
        }                                                                                        \
        if (tid == 0) {                                                                          \
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bara));                  \
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");                   \
        }                                                                                        \
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");                             \
        asm volatile("tcgen05.fence::before_thread_sync;");                                      \
        __syncthreads();                                                                         \
        asm volatile("tcgen05.fence::after_thread_sync;");                                       \
        unsigned tm = *(volatile unsigned*)&tslot;                                               \
        unsigned ta = tm + ((unsigned)(32 * w) << 16);                                           \
        asm volatile(                                                                            \
            "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, " \
            "%10, %11, %12, %13, %14, %15, %16};" ::"r"(ta),                                     \
            "r"(cv[0]), "r"(cv[1]), "r"(cv[2]), "r"(cv[3]), "r"(cv[4]), "r"(cv[5]), "r"(cv[6]),  \
            "r"(cv[7]), "r"(cv[8]), "r"(cv[9]), "r"(cv[10]), "r"(cv[11]), "r"(cv[12]),           \
            "r"(cv[13]), "r"(cv[14]), "r"(cv[15]) : "memory");                                   \
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");                             \
        asm volatile("tcgen05.fence::before_thread_sync;");                                      \
        __syncthreads();                                                                         \
        asm volatile("tcgen05.fence::after_thread_sync;");                                       \
        if (tid == 0) {                                                                          \
            unsigned long long ad =                                                              \
                (unsigned long long)((unsigned)__cvta_generic_to_shared(sA) >> 4) |              \
                0x0000400800800000ull; /* LBO 2048, SBO 128 */                                   \
            unsigned long long bd =                                                              \
                (unsigned long long)((unsigned)__cvta_generic_to_shared(sB) >> 4) |              \
                0x0000400800100000ull; /* LBO 256, SBO 128 */                                    \
            asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %4, 0;\n"                           \
                         " tcgen05.mma.cta_group::1.kind::" KIND " [%0], %1, %2, %3, p;\n}"     \
                         ::"r"(tm), "l"(ad), "l"(bd), "r"(idesc), "r"(en) : "memory");           \
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 " \
                         "[%0];" ::"r"(bara) : "memory");                                        \
        }                                                                                        \
        asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], " \
                     "0;\n @!p bra W;\n}" ::"r"(bara) : "memory");                               \
        asm volatile("tcgen05.fence::after_thread_sync;");                                       \
        unsigned r[16];                                                                          \
        asm volatile(                                                                            \
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, "   \
            "%10, %11, %12, %13, %14, %15}, [%16];"                                              \
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]),            \
              "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),          \
              "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])                                 \
            : "r"(ta) : "memory");                                                               \
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");                             \
        float* d = D + (t * 128 + tid) * 16;                                                     \
        _Pragma("unroll") for (int j = 0; j < 16; j++) d[j] = __uint_as_float(r[j]);             \
        asm volatile("tcgen05.fence::before_thread_sync;");                                      \
        __syncthreads();                                                                         \
        if (w == 0) {                                                                            \
            asm volatile("tcgen05.fence::after_thread_sync;");                                   \
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" ::"r"(tm));     \
        }                                                                                        \
    }

TC_KERNEL(tc_f16, "f16")
TC_KERNEL(tc_f8, "f8f6f4")
TC_KERNEL(tc_tf32, "tf32")

// mma.sync m16n8k16, A[16][16] row-major, B stored [n][k] (8 x 16), C / D [16][8] per tile.
#define HM_KERNEL(NAME, TY, CZERO)                                                               \
    extern "C" __global__ void __launch_bounds__(32)                                             \
        NAME(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B,                     \
             const float* __restrict__ C, float* __restrict__ D) {                               \
        int lane = threadIdx.x, g = lane / 4, t = lane % 4, tile = blockIdx.x;                   \
        const uint32_t* a = A + tile * 128;                                                      \
        const uint32_t* b = B + tile * 64;                                                       \
        const float* c = C + tile * 128;                                                         \
        float* d = D + tile * 128;                                                               \
        uint32_t a0 = a[(g * 16 + 2 * t) / 2], a1 = a[((g + 8) * 16 + 2 * t) / 2];               \
        uint32_t a2 = a[(g * 16 + 2 * t + 8) / 2], a3 = a[((g + 8) * 16 + 2 * t + 8) / 2];       \
        uint32_t b0 = b[(g * 16 + 2 * t) / 2], b1 = b[(g * 16 + 2 * t + 8) / 2];                 \
        float c0, c1, c2, c3;                                                                    \
        if (CZERO) {                                                                             \
            c0 = c1 = c2 = c3 = 0.f;                                                             \
        } else {                                                                                 \
            c0 = c[g * 8 + 2 * t];                                                               \
            c1 = c[g * 8 + 2 * t + 1];                                                           \
            c2 = c[(g + 8) * 8 + 2 * t];                                                         \
            c3 = c[(g + 8) * 8 + 2 * t + 1];                                                     \
        }                                                                                        \
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32." TY "." TY                          \
                     ".f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"                \
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)                                    \
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));                    \
        d[g * 8 + 2 * t] = c0;                                                                   \
        d[g * 8 + 2 * t + 1] = c1;                                                               \
        d[(g + 8) * 8 + 2 * t] = c2;                                                             \
        d[(g + 8) * 8 + 2 * t + 1] = c3;                                                         \
    }

HM_KERNEL(hm, "bf16", 0)
HM_KERNEL(hm0, "bf16", 1)
HM_KERNEL(hmh, "f16", 0)

// mma.sync m16n8k8 tf32, A[16][8] f32 words, B stored [n][k] (8 x 8), C / D [16][8] per tile.
#define TM_KERNEL(NAME, CZERO)                                                                   \
    extern "C" __global__ void __launch_bounds__(32)                                             \
        NAME(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B,                     \
             const float* __restrict__ C, float* __restrict__ D) {                               \
        int lane = threadIdx.x, g = lane / 4, t = lane % 4, tile = blockIdx.x;                   \
        const uint32_t* a = A + tile * 128;                                                      \
        const uint32_t* b = B + tile * 64;                                                       \
        const float* c = C + tile * 128;                                                         \
        float* d = D + tile * 128;                                                               \
        uint32_t a0 = a[g * 8 + t], a1 = a[(g + 8) * 8 + t];                                     \
        uint32_t a2 = a[g * 8 + t + 4], a3 = a[(g + 8) * 8 + t + 4];                             \
        uint32_t b0 = b[g * 8 + t], b1 = b[g * 8 + t + 4];                                       \
        float c0, c1, c2, c3;                                                                    \
        if (CZERO) {                                                                             \
            c0 = c1 = c2 = c3 = 0.f;                                                             \
        } else {                                                                                 \
            c0 = c[g * 8 + 2 * t];                                                               \
            c1 = c[g * 8 + 2 * t + 1];                                                           \
            c2 = c[(g + 8) * 8 + 2 * t];                                                         \
            c3 = c[(g + 8) * 8 + 2 * t + 1];                                                     \
        }                                                                                        \
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, "        \
                     "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"                                    \
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)                                    \
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));                    \
        d[g * 8 + 2 * t] = c0;                                                                   \
        d[g * 8 + 2 * t + 1] = c1;                                                               \
        d[(g + 8) * 8 + 2 * t] = c2;                                                             \
        d[(g + 8) * 8 + 2 * t + 1] = c3;                                                         \
    }

TM_KERNEL(tm, 0)
TM_KERNEL(tm0, 1)
