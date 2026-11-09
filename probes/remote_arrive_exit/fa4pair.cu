/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// FA4 2-CTA teardown shape (L1) with one binary for every variant: the targets are runtime
// values, so the SASS and its timing are the same in all runs.
// Cluster (2,1,1), one CTA per SM, 512 threads. Rank 0 initializes E (count 512, never waited)
// and D (count 1); its thread 0 waits on D, then rank 0 exits.
// Rank 1, warps 0-7: K st.shared::cluster per thread into the CTA `st_tgt` (0: rank 0, 1:
// itself), then `fence.proxy.async.shared::cta` (SASS MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S, as
// FA4 before 0xa4b0) and `mbarrier.arrive.release.cta.shared::cluster` on E of CTA `a_tgt`
// (A); then `bar.sync 1, 288`. Warp 8 waits at the bar.sync and arrives on rank 0's D (B).
//   a_tgt 1 is the control in which A cannot fault (it targets the live issuer); st_tgt 1
//   keeps the stores off the peer link.
//   late = 1: A is issued 5 us after the bar.sync instead (positive control: late issue).
// arv: warps 0-7 use bar.arrive instead of bar.sync (FA4's 0xb2f0 -> BAR.ARV 0x2, 0x1a0 -> EXIT).
// same: thread 0 issues B itself right after its A arrivals (one thread, two mbarriers, no
// barrier in between: the FusedRMSNormBwd epilogue shape); warp 8 then issues nothing.
// arep: A is issued arep times in a row (arrival traffic to rank 0 ahead of the bar.sync).
// Noise: every other cluster hammers DSMEM to its peer for 200 us instead.
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o fa4pair fa4pair.cu
// Run:   CUDA_VISIBLE_DEVICES=<gpu> ./fa4pair K st_tgt a_tgt late launches noise arep arv same
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(512)
k(int K, unsigned st_tgt, unsigned a_tgt, int late, int noise, int arep, int arv, int same) {
    extern __shared__ __align__(1024) unsigned s[];
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    unsigned base = (unsigned)__cvta_generic_to_shared(s);
    unsigned E = base + 0x490, D = base + 0x4a0, R = base + 0x1000;
    unsigned cl = blockIdx.x / 2;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 512;" :: "r"(E));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(D));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.release;\n barrier.cluster.wait.acquire;" ::: "memory");
    unsigned w = threadIdx.x / 32;
    if (noise && (cl & 1)) {
        unsigned r;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(R + 4 * threadIdx.x), "r"(rank ^ 1));
        unsigned long long t0 = gtime();
        while (gtime() - t0 < 200000)
            for (int i = 0; i < 64; i++)
                asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(r + 2048 * (i & 15)), "r"(i) : "memory");
        asm volatile("barrier.cluster.arrive.release;\n barrier.cluster.wait.acquire;" ::: "memory");
        return;
    }
    if (rank == 0) {
        if (threadIdx.x == 0)
            asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra W;\n}"
                         :: "r"(D) : "memory");
        return;
    }
    if (w < 8) {
        unsigned rE, rR;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rE) : "r"(E), "r"(a_tgt));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rR) : "r"(R + 4 * threadIdx.x), "r"(st_tgt));
        for (int i = 0; i < K; i++)
            asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(rR + 1024 * (i & 63)), "r"(i) : "memory");
        if (!late) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            for (int j = 0; j < arep; j++)
                asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(rE) : "memory");
            if (same && threadIdx.x == 0) {  // B from the same thread, no barrier in between
                unsigned rD;
                asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rD) : "r"(D));
                asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(rD) : "memory");
            }
        }
        if (arv) asm volatile("bar.arrive 1, 288;" ::: "memory");
        else asm volatile("bar.sync 1, 288;" ::: "memory");
        if (late) {
            unsigned long long t0 = gtime();
            while (gtime() - t0 < 5000) {}
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(rE) : "memory");
        }
    } else if (w == 8) {
        asm volatile("bar.sync 1, 288;" ::: "memory");
        if (threadIdx.x == 256 && !same) {
            unsigned rD;
            asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rD) : "r"(D));
            asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(rD) : "memory");
        }
    }
}

int main(int argc, char** argv) {
    int K = atoi(argv[1]), st = atoi(argv[2]), at = atoi(argv[3]), late = atoi(argv[4]), n = atoi(argv[5]);
    int noise = argc > 6 ? atoi(argv[6]) : 0;
    int arep = argc > 7 ? atoi(argv[7]) : 1;
    int arv = argc > 8 ? atoi(argv[8]) : 0;
    int same = argc > 9 ? atoi(argv[9]) : 0;
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int ncl = sms / 2, smem = 200 * 1024;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int done = 0;
    cudaError_t e = cudaSuccess;
    for (; done < n; done++) {
        k<<<ncl * 2, 512, smem>>>(K, st, at, late, noise, arep, arv, same);
        e = cudaDeviceSynchronize();
        if (e) break;
    }
    printf("K=%d st_tgt=%d a_tgt=%d late=%d noise=%d arep=%d arv=%d same=%d launches_ok=%d test_clusters_per_launch=%d error=%s\n", K, st, at,
           late, noise, arep, arv, same, done, noise ? ncl / 2 : ncl, e ? cudaGetErrorString(e) : "none");
    return e ? 1 : 0;
}
