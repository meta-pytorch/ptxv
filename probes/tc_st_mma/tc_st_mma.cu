/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Same-thread tcgen05.st -> tcgen05.wait::st -> tcgen05.mma without tcgen05.fence (ptxv
// S-TCGEN05; mslk mx8mx4 grouped GEMM, PTX front). Does the MMA a thread issues right after
// its own wait::st read the TMEM words that thread just stored?
// Per iteration, every warp first stores "old" words and synchronizes properly (wait::st,
// fence::before_thread_sync, bar.sync, fence::after_thread_sync). Then warp 0 stores "new"
// words (thread 0: TMEM lane 0, the lane checked), waits with tcgen05.wait::st, and thread 0
// issues one tcgen05.mma.cta_group::1.kind::f16 (bf16, M128 N16 K16) at once, with or
// without the fences in between. The MMA reads the stored words as
//   mode 0 / 1: C (enable_input_d, A = B = 0 from shared memory: D = C exactly);
//   mode 2 / 3: A from TMEM (B = identity from shared memory, C disabled: D row 0 = A row 0).
// Odd modes put tcgen05.fence::before_thread_sync / __syncwarp / after_thread_sync between
// wait::st and the MMA (the control). After commit -> mbarrier, warp 0 loads D and thread 0
// counts lane-0 words that do not hold the new value (old value: stale; anything else:
// other). Lanes 1..31 (other threads of the warp, not ordered by the ISA without
// synchronization) are counted separately. Modes 4 / 5 drop the wait::st too: a control for
// the probe's sensitivity (the MMA may then read old words).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ unsigned long long g_cnt[6][4];  // [mode][lane0 stale, lane0 other, lanes1-31 bad, checked]

__device__ __forceinline__ unsigned bf(float x) { return __float_as_uint(x) >> 16; }  // exact small ints

extern "C" __global__ void __launch_bounds__(128) probe(int mode, int iters) {
    __shared__ __align__(1024) unsigned char sA[4096];
    __shared__ __align__(1024) unsigned char sB[512];
    __shared__ __align__(8) unsigned long long bar;
    __shared__ unsigned tslot;
    int tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    // Modes 4 / 5: C read / A in TMEM with no wait::st either (sensitivity control).
    bool a_tmem = mode == 2 || mode == 3 || mode == 5, fenced = mode < 4 && (mode & 1), nowait = mode >= 4;
    // sA: zeros (128 rows x 32 B, K-major, no swizzle: row r at (r/8)*128 + (r%8)*16, two
    // 16-byte K halves 2048 apart). sB: 16 x 16 bf16, identity for the A-in-TMEM modes, else 0.
    for (int i = tid; i < 1024; i += 128) ((unsigned*)sA)[i] = 0;
    if (tid < 16) {
        for (int h = 0; h < 2; h++) {
            unsigned v[4];
            for (int j = 0; j < 4; j++) {
                int k0 = h * 8 + 2 * j;
                unsigned lo = (a_tmem && k0 == tid) ? 0x3f80u : 0u;
                unsigned hi = (a_tmem && k0 + 1 == tid) ? 0x3f80u : 0u;
                v[j] = lo | hi << 16;
            }
            *(uint4*)(sB + h * 256 + (tid / 8) * 128 + (tid % 8) * 16) = make_uint4(v[0], v[1], v[2], v[3]);
        }
    }
    unsigned bara = (unsigned)__cvta_generic_to_shared(&bar);
    unsigned slot = (unsigned)__cvta_generic_to_shared(&tslot);
    if (w == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;" ::"r"(slot));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bara));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    unsigned tm = *(volatile unsigned*)&tslot;
    unsigned tD = tm + ((unsigned)(32 * w) << 16);          // D / C: columns 0..15
    unsigned tA = tm + 32 + ((unsigned)(32 * w) << 16);     // A: columns 32..39 (16 bf16)
    unsigned long long ad = (unsigned long long)((unsigned)__cvta_generic_to_shared(sA) >> 4) |
                            0x0000400800800000ull;  // LBO 2048, SBO 128
    unsigned long long bd = (unsigned long long)((unsigned)__cvta_generic_to_shared(sB) >> 4) |
                            0x0000400800100000ull;  // LBO 256, SBO 128
    const unsigned idesc = (8u << 24) | (2u << 17) | 0x490u;  // bf16 x bf16 -> f32, M128 N16
    unsigned long long stale = 0, other = 0, bad = 0, checked = 0;
    for (int it = 0; it < iters; it++) {
        int seed = (blockIdx.x * 977 + it * 131) & 1023;
        // Old words (every warp), properly synchronized.
        unsigned o[16], n[16];
        for (int j = 0; j < 16; j++) {
            float ov = (float)((seed + lane * 3 + j) & 127) + 1.0f;
            float nv = ov + 256.0f;
            if (a_tmem) {
                // 8 columns of A, two bf16 each (k = 2j, 2j + 1).
                o[j] = j < 8 ? (bf((float)((seed + lane + 2 * j) & 63) + 1.0f) |
                                bf((float)((seed + lane + 2 * j + 1) & 63) + 1.0f) << 16) : 0;
                n[j] = j < 8 ? (bf((float)((seed + lane + 2 * j) & 63) + 65.0f) |
                                bf((float)((seed + lane + 2 * j + 1) & 63) + 65.0f) << 16) : 0;
            } else {
                o[j] = __float_as_uint(ov);
                n[j] = __float_as_uint(nv);
            }
        }
        unsigned ta = a_tmem ? tA : tD;
#define ST16(addr, v)                                                                            \
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, " \
                 "%9, %10, %11, %12, %13, %14, %15, %16};" ::"r"(addr),                          \
                 "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]),    \
                 "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]),            \
                 "r"(v[13]), "r"(v[14]), "r"(v[15]) : "memory")
        ST16(ta, o);
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (w == 0) {
            ST16(ta, n);
            if (!nowait) asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            if (fenced) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            if (tid == 0) {
                if (a_tmem)
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, 0;"
                                 ::"r"(tm), "r"(tm + 32), "l"(bd), "r"(idesc) : "memory");
                else
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;"
                                 ::"r"(tm), "l"(ad), "l"(bd), "r"(idesc) : "memory");
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                             ::"r"(bara) : "memory");
            }
        }
        asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                     " @!p bra W;\n}" ::"r"(bara), "r"(it & 1) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (w == 0) {
            unsigned r[16];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, "
                         "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]),
                           "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
                           "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                         : "r"(tD) : "memory");
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            for (int j = 0; j < 16; j++) {
                unsigned want, old;
                if (a_tmem) {
                    // D[row][n] = A[row][n] (B identity): bf16 k = n of the stored words.
                    unsigned wn = n[j / 2] >> (16 * (j & 1)) & 0xffff, wo = o[j / 2] >> (16 * (j & 1)) & 0xffff;
                    want = wn << 16; old = wo << 16;
                } else {
                    want = n[j]; old = o[j];
                }
                if (lane == 0) {
                    checked++;
                    if (r[j] != want) { if (r[j] == old) stale++; else other++; }
                } else if (r[j] != want) {
                    bad++;
                }
            }
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;");
    }
    if (lane == 0 && w == 0) {
        atomicAdd(&g_cnt[mode][0], stale); atomicAdd(&g_cnt[mode][1], other); atomicAdd(&g_cnt[mode][3], checked);
    }
    if (w == 0 && bad) atomicAdd(&g_cnt[mode][2], bad);
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (w == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" ::"r"(tm));
    }
}

int main(int argc, char** argv) {
    int ctas = 296, iters = argc > 1 ? atoi(argv[1]) : 2000;
    const char* name[6] = {"C read, no fence", "C read, fenced (control)", "A in TMEM, no fence",
                           "A in TMEM, fenced (control)", "C read, no wait::st", "A in TMEM, no wait::st"};
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("%s; %d CTAs x %d iterations per mode\n", p.name, ctas, iters);
    for (int m = 0; m < 6; m++) {
        probe<<<ctas, 128>>>(m, iters);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("mode %d: %s\n", m, cudaGetErrorString(e)); return 1; }
    }
    unsigned long long c[6][4];
    cudaMemcpyFromSymbol(c, g_cnt, sizeof c);
    for (int m = 0; m < 6; m++)
        printf("mode %d %-28s lane 0: %llu words checked, %llu stale, %llu other; lanes 1-31 wrong: %llu\n",
               m, name[m], c[m][3], c[m][0], c[m][1], c[m][2]);
    return 0;
}
