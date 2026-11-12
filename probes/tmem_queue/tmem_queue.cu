/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// E-TMEM-QUEUE (ptxv ASSUMPTIONS): is a thread's tcgen05.st (SASS STTM) ordered after its own
// earlier tcgen05.ld (LDTM) of the same columns, with no tcgen05.wait::ld between them? Per
// iteration every warp stores "old" words to 16 columns (240..255) of its TMEM lanes (st; wait::st) and
// synchronizes. Then each warp
//   mode 0: ld the 16 columns, st "new" words to them at once (no wait::ld), then wait::ld;
//   mode 1: ld, wait::ld, st (control: ordered by the wait);
//   mode 2: st "new" first, ld at once (no wait::st), wait::ld (the reverse order; recorded,
//           not assumed by the model);
//   mode 4: st "a", st "new" at once (no wait::st between), wait::st, ld (E-TMEM-QUEUE's
//           second order: two un-waited stores of one thread to the same words land in issue
//           order);
// and every thread counts loaded words that are not the expected value (modes 0 / 1: old;
// modes 2 / 4: new), split into "the other value" (mode 4: a) and anything else.
//   mode 3 (a timing check, not a reordering control): warp 4 (TMEM lanes 0..31, as warp 0)
// announces through a shared flag that it loads, sleeps 2 us, and loads warp 0's 16 columns;
// warp 0 stores "new" to them once it sees the flag. Loads that see "new" (counted as "other
// value") show only that the check reads the other value when the store lands first in real
// time (about 2 us first); they give modes 0, 2 and 4 no power.
// Warps 0..3 run modes 0, 1, 2 and 4; warp 4 only mode 3.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ unsigned long long g_cnt[5][3];  // [mode][other value, else, checked]

extern "C" __global__ void __launch_bounds__(160) probe(int mode, int iters) {
    __shared__ __align__(1024) unsigned char sA[4096];
    __shared__ __align__(1024) unsigned char sB[8192];
    __shared__ __align__(8) unsigned long long bar;
    __shared__ unsigned tslot;
    __shared__ unsigned flag;
    int tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    for (int i = tid; i < 1024; i += 160) ((unsigned*)sA)[i] = 0;
    for (int i = tid; i < 2048; i += 160) ((unsigned*)sB)[i] = 0;
    unsigned bara = (unsigned)__cvta_generic_to_shared(&bar);
    unsigned slot = (unsigned)__cvta_generic_to_shared(&tslot);
    unsigned flaga = (unsigned)__cvta_generic_to_shared(&flag);
    if (tid == 0) flag = 0;
    if (w == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::"r"(slot));
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
    unsigned tD = tm + ((unsigned)(32 * (w % 4)) << 16);  // warp 4 shares warp 0's lanes
    unsigned long long ad = (unsigned long long)((unsigned)__cvta_generic_to_shared(sA) >> 4) |
                            0x0000400800800000ull;
    unsigned long long bd = (unsigned long long)((unsigned)__cvta_generic_to_shared(sB) >> 4) |
                            0x0000400801000000ull;  // LBO 4096, SBO 128 (N 256, zeros)
    const unsigned idesc = (8u << 24) | (32u << 17) | 0x490u;  // bf16 x bf16 -> f32, M128 N256
    // Modes 0-2 use columns 240..255 too (mode 3's last columns).
    tD += 240;
    unsigned long long oth = 0, els = 0, checked = 0;
    unsigned phase = 0;
#define ST16(addr, v)                                                                            \
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, " \
                 "%9, %10, %11, %12, %13, %14, %15, %16};" ::"r"(addr),                          \
                 "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]),    \
                 "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]),            \
                 "r"(v[13]), "r"(v[14]), "r"(v[15]) : "memory")
#define LD16(addr, r)                                                                            \
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, "      \
                 "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];"                                 \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]),       \
                   "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),     \
                   "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])                            \
                 : "r"(addr) : "memory")
    for (int it = 0; it < iters; it++) {
        int seed = (blockIdx.x * 977 + it * 131 + (w % 4) * 37) & 1023;  // warp 4 = warp 0's words
        unsigned o[16], n[16], r[16];
        for (int j = 0; j < 16; j++) {
            o[j] = 0x40000000u | ((seed + lane * 3 + j) & 0xffff) << 4 | 1;   // never zero
            n[j] = o[j] ^ 0x00f00000u;                                      // differs from o
        }
        if (w < 4) {
            ST16(tD, o);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;");
        bool check = w < 4;
        if (w == 4 && mode != 3) {
            // idle: shares warp 0's lanes
        } else if (mode == 0) {
            LD16(tD, r);
            ST16(tD, n);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        } else if (mode == 1) {
            LD16(tD, r);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            ST16(tD, n);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        } else if (mode == 2) {
            ST16(tD, n);
            LD16(tD, r);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        } else if (mode == 4) {
            // Two un-waited stores of one thread to the same words: old (already there) is
            // rewritten as a, then n, with no wait between; then wait::st and a load.
            unsigned a[16];
            for (int j = 0; j < 16; j++) a[j] = o[j] ^ 0x0f000000u;
            ST16(tD, a);
            ST16(tD, n);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            LD16(tD, r);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            for (int j = 0; j < 16; j++) o[j] = a[j];  // the "other value" is the first store's
        } else {
            // Positive control: warp 4 announces its load, sleeps, then loads warp 0's lanes;
            // warp 0 stores "new" there as soon as it sees the announcement. Nothing orders
            // the two warps' accesses: loads that see "new" show that the probe observes TMEM
            // accesses landing out of their (announced) order.
            check = w == 4;
            if (w == 4) {
                if (lane == 0)
                    asm volatile("st.release.cta.shared.u32 [%0], %1;" ::"r"(flaga), "r"(it + 1) : "memory");
                __syncwarp();
                __nanosleep(2000);
                LD16(tD, r);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            } else if (w == 0) {
                unsigned f;
                do {
                    asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(f) : "r"(flaga) : "memory");
                } while (f != (unsigned)(it + 1));
                ST16(tD, n);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
        if (check) {
            for (int j = 0; j < 16; j++) {
                unsigned want = mode == 2 || mode == 4 ? n[j] : o[j];
                unsigned alt = mode == 2 || mode == 4 ? o[j] : n[j];
                checked++;
                if (r[j] != want) { if (r[j] == alt) oth++; else els++; }
            }
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;");
    }
    atomicAdd(&g_cnt[mode][0], oth); atomicAdd(&g_cnt[mode][1], els); atomicAdd(&g_cnt[mode][2], checked);
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (w == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tm));
    }
}

int main(int argc, char** argv) {
    int ctas = 296, iters = argc > 1 ? atoi(argv[1]) : 2000, reps = argc > 2 ? atoi(argv[2]) : 1;
    const char* name[5] = {"ld; st (no wait::ld)", "ld; wait::ld; st (control)",
                           "st; ld (no wait::st)", "other warp st, ld 2 us later",
                           "st a; st b (no wait); wait; ld"};
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("%s; %d CTAs x 160 threads x %d iterations x %d reps per mode\n", p.name, ctas, iters, reps);
    for (int m = 0; m < 5; m++)
        for (int k = 0; k < reps; k++) {
            probe<<<ctas, 160>>>(m, iters);
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) { printf("mode %d: %s\n", m, cudaGetErrorString(e)); return 1; }
        }
    unsigned long long c[5][3];
    cudaMemcpyFromSymbol(c, g_cnt, sizeof c);
    for (int m = 0; m < 5; m++)
        printf("mode %d %-34s %llu words checked, %llu other value, %llu else\n", m, name[m], c[m][2], c[m][0], c[m][1]);
    return 0;
}
