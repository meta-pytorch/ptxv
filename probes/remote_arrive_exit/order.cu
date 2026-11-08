/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Order probe for E-REMOTE-FIFO: do remote mbarrier arrivals from one CTA to one CTA land in
// happens-before issue order? Cluster of C CTAs (runtime C in 2..16, non-portable sizes
// allowed), one CTA per SM, 512 threads. Rank 0 is the target, rank `src` the source.
// Target thread 0: waits on Y (try_wait loop, phase 0), then at once test_wait's X phase 0
// (non-blocking); X not complete = "B seen, A not" (a violation of the order). The other
// target threads idle.
// Source:
//   shape 0 (bar):  warps 0-7, every lane: `arep` remote arrivals on X (A), [spin D ns],
//                   bar.sync 1, 288; warp 8 lane 0: remote arrive on Y (B)
//   shape 1 (same): warp 0 lane 0: `arep` remote arrivals on X, [spin D ns], arrive on Y
//   shape 2 (bar, stores; positive-control shape): warps 0-7, every lane: `arep` remote
//                   st.shared::cluster to its own word of the target, the last one writing
//                   the run tag, [spin D ns], bar.sync; warp 8 lane 0 arrives on Y. The
//                   target checks the 256 words after Y: any word without the tag = "B seen,
//                   A not" for stores
//   shape 3 (same, stores): warp 0 lane 0: `arep` stores, the last with the tag, then Y
// Every arrive is `mbarrier.arrive.release.cta.shared::cluster` after
// `fence.proxy.async.shared::cta` (FA4's SASS sequence). X expects all A arrivals of a run.
// Noise: when noise != 0, ranks other than 0 and src hammer DSMEM to their neighbour
// (r ^ 1 when that is not 0 / src, else themselves) for the whole run.
// Placement: the target logs its %smid and the source's; the host buckets violations by
// "same TPC" (smid / 2 equal) and by smid distance.
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o order order.cu
// Run:   CUDA_VISIBLE_DEVICES=<gpu> ./order C src shape arep D_ns launches noise
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned smid() {
    unsigned r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}
__device__ __forceinline__ void arrive_remote(unsigned a) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(a) : "memory");
}

struct Log { unsigned viol, sm_t, sm_s, pad; };

__global__ void __launch_bounds__(512) k(int src, int shape, int arep, unsigned dns, int noise, unsigned tag, Log* L) {
    extern __shared__ __align__(1024) unsigned s[];
    unsigned rank, csize;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(csize));
    unsigned cl = blockIdx.x / csize;
    unsigned base = (unsigned)__cvta_generic_to_shared(s);
    unsigned X = base, Y = base + 8, W = base + 1024, R = base + 8192, SM = base + 16;
    bool stores = shape >= 2, same = shape & 1;
    unsigned nA = same ? arep : 256 * arep;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(X), "r"(stores ? 1u : nA));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(Y));
    }
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s[256 + i] = 0;  // W words
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release;\n barrier.cluster.wait.acquire;" ::: "memory");
    unsigned w = threadIdx.x / 32;
    if (rank == 0) {
        if (threadIdx.x == 0) {
            asm volatile("{\n .reg .pred p;\n WT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra WT;\n}"
                         :: "r"(Y) : "memory");
            unsigned v = 0;
            if (!stores) {
                asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], 0;\n"
                             " selp.u32 %0, 0, 1, p;\n}" : "=r"(v) : "r"(X) : "memory");
            } else {
                int n = same ? 1 : 256;
                for (int i = 0; i < n; i++) {
                    unsigned x;
                    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(x) : "r"(W + 4 * i) : "memory");
                    v += x != tag;
                }
            }
            unsigned ss;
            asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(ss) : "r"(SM) : "memory");
            L[cl].viol = v;
            L[cl].sm_t = smid();
            L[cl].sm_s = ss;
        }
    } else if (rank == (unsigned)src) {
        unsigned rX, rY, rW, rSM;
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rX) : "r"(X));
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rY) : "r"(Y));
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rW) : "r"(W + 4 * (threadIdx.x & 255)));
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rSM) : "r"(SM));
        if (threadIdx.x == 0)
            asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(rSM), "r"(smid()) : "memory");
        bool issuer = same ? threadIdx.x == 0 : w < 8;
        if (issuer) {
            for (int j = 0; j < arep; j++) {
                if (stores)
                    asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(rW), "r"(j == arep - 1 ? tag : j) : "memory");
                else
                    arrive_remote(rX);
            }
            if (dns) {
                unsigned long long t0 = gtime();
                while (gtime() - t0 < dns) {}
            }
        }
        if (same) {
            if (threadIdx.x == 0) arrive_remote(rY);
        } else if (w <= 8) {
            asm volatile("bar.sync 1, 288;" ::: "memory");
            if (threadIdx.x == 256) arrive_remote(rY);
        }
    } else if (noise) {
        unsigned nb = rank ^ 1;
        if (nb == 0 || nb == (unsigned)src || nb >= csize) nb = rank;
        unsigned r;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(R + 4 * threadIdx.x), "r"(nb));
        unsigned long long t0 = gtime();
        while (gtime() - t0 < 30000)
            for (int i = 0; i < 32; i++)
                asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(r + 2048 * (i & 15)), "r"(i) : "memory");
    }
    // Everyone (incl. the target) stays until the cluster is done: no exit-related effects.
    asm volatile("barrier.cluster.arrive.release;\n barrier.cluster.wait.acquire;" ::: "memory");
}

int main(int argc, char** argv) {
    int C = atoi(argv[1]), src = atoi(argv[2]), shape = atoi(argv[3]), arep = atoi(argv[4]);
    unsigned dns = atoi(argv[5]);
    int n = atoi(argv[6]), noise = argc > 7 ? atoi(argv[7]) : 0;
    int smem = 200 * 1024;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = smem;
    int maxcl = 0;
    cudaOccupancyMaxActiveClusters(&maxcl, (void*)k, &cfg);
    int ncl = maxcl > 0 ? maxcl : sms / C;
    cfg.gridDim = dim3(ncl * C);
    Log* L;
    cudaMalloc(&L, ncl * sizeof(Log));
    std::vector<Log> h(ncl);
    long viol = 0, viol_cl = 0, tot = 0, same_tpc = 0, same_tpc_viol = 0;
    for (int it = 0; it < n; it++) {
        cudaMemset(L, 0xff, ncl * sizeof(Log));
        unsigned tag = 0x5a000000u | (unsigned)it;
        cudaLaunchKernelEx(&cfg, k, src, shape, arep, dns, noise, tag, L);
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { printf("error %s at launch %d\n", cudaGetErrorString(e), it); return 1; }
        cudaMemcpy(h.data(), L, ncl * sizeof(Log), cudaMemcpyDeviceToHost);
        for (auto& l : h) {
            tot++;
            viol += l.viol;
            viol_cl += l.viol > 0;
            bool st = l.sm_t / 2 == l.sm_s / 2;
            same_tpc += st;
            same_tpc_viol += st && l.viol > 0;
        }
    }
    printf("C=%d src=%d shape=%d arep=%d D=%uns noise=%d: clusters %ld (same TPC %ld), clusters with B-before-A %ld "
           "(same TPC %ld), violating %s %ld\n",
           C, src, shape, arep, dns, noise, tot, same_tpc, viol_cl, same_tpc_viol,
           shape >= 2 ? "words" : "tests", viol);
    return 0;
}
