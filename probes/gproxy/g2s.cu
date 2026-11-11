/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// S-GPROXY litmus: generic-proxy global stores read back by an async-proxy bulk copy.
// Per round k, thread 0 of each CTA stores k into W words of its own global slot (st.global,
// generic proxy), runs the variant's fence, then copies the slot into shared memory with
// cp.async.bulk.shared::cta.global (async proxy, mbarrier completion) and counts words != k.
// Variants (argument `v`):
//   0 NOF : no fence between the stores and the copy
//   1 GLB : fence.proxy.async.global
//   2 ALL : fence.proxy.async
//   3 SH  : fence.proxy.async.shared::cta (shared memory only)
//   4 MB  : fence.acq_rel.gpu (no proxy fence)
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o g2s g2s.cu
// Run:   ./g2s v rounds W launches
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define MAXW 1024

__global__ void __launch_bounds__(32) g2s(int v, int rounds, int W, unsigned* slots,
                                          unsigned long long* bad, unsigned* first) {
    __shared__ alignas(128) unsigned buf[MAXW];
    __shared__ alignas(8) unsigned long long mb;
    if (threadIdx.x != 0) return;
    unsigned* slot = slots + (size_t)blockIdx.x * MAXW;
    unsigned sb = (unsigned)__cvta_generic_to_shared(buf);
    unsigned smb = (unsigned)__cvta_generic_to_shared(&mb);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smb));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    unsigned long long nbad = 0;
    unsigned fr = 0xffffffffu;
    for (int k = 1; k <= rounds; k++) {
        for (int i = 0; i < W; i++)
            asm volatile("st.global.u32 [%0], %1;" ::"l"(slot + i), "r"((unsigned)k) : "memory");
        switch (v) {
            case 1: asm volatile("fence.proxy.async.global;" ::: "memory"); break;
            case 2: asm volatile("fence.proxy.async;" ::: "memory"); break;
            case 3: asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); break;
            case 4: asm volatile("fence.acq_rel.gpu;" ::: "memory"); break;
            default: break;
        }
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smb), "r"(W * 4)
                     : "memory");
        asm volatile(
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sb),
            "l"(slot), "r"(W * 4), "r"(smb)
            : "memory");
        unsigned ok = 0;
        while (!ok)
            asm volatile(
                "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                : "=r"(ok)
                : "r"(smb), "r"((unsigned)((k - 1) & 1))
                : "memory");
        for (int i = 0; i < W; i++) {
            unsigned x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(sb + 4 * i) : "memory");
            if (x != (unsigned)k) {
                nbad++;
                if (fr == 0xffffffffu) fr = k;
            }
        }
    }
    bad[blockIdx.x] = nbad;
    first[blockIdx.x] = fr;
}

int main(int argc, char** argv) {
    int v = argc > 1 ? atoi(argv[1]) : 0, rounds = argc > 2 ? atoi(argv[2]) : 1000;
    int W = argc > 3 ? atoi(argv[3]) : 64, launches = argc > 4 ? atoi(argv[4]) : 10;
    if (W < 4 || W > MAXW || W % 4) return 2;
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int nb = sms * 4;
    unsigned* slots;
    unsigned long long* bad;
    unsigned* first;
    cudaMalloc(&slots, (size_t)nb * MAXW * 4);
    cudaMalloc(&bad, nb * 8);
    cudaMalloc(&first, nb * 4);
    unsigned long long tot = 0, ctas = 0, words = 0;
    for (int it = 0; it < launches; it++) {
        cudaMemset(slots, 0xff, (size_t)nb * MAXW * 4);
        g2s<<<nb, 32>>>(v, rounds, W, slots, bad, first);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("error %s\n", cudaGetErrorString(e));
            return 1;
        }
        unsigned long long hb[4096];
        cudaMemcpy(hb, bad, nb * 8, cudaMemcpyDeviceToHost);
        for (int c = 0; c < nb; c++) {
            tot += hb[c];
            ctas += hb[c] > 0;
        }
        words += (unsigned long long)nb * rounds * W;
    }
    printf("v=%d rounds=%d W=%d launches=%d ctas/launch=%d: words read %llu, stale %llu, CTA-launches with a stale word %llu\n",
           v, rounds, W, launches, nb, words, tot, ctas);
    return 0;
}
