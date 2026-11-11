/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// S-MBAR-SCOPE litmus: a 2-CTA tile-id hand-off through a remote mbarrier arrive, many rounds
// per cluster on one slot. Rank 0 (producer, thread 0) per round k: optionally P weak remote
// stores to scratch words of rank 1 (queue pressure), the weak remote store of k into rank 1's
// slot, the variant's fence, a remote arrive on rank 1's `full` barrier; then it waits on its
// own `empty` barrier (rank 1's give-back) before the next round. Rank 1 (consumer, thread 0)
// per round: waits on `full` (phase k & 1), optionally spins S ns, reads the slot, records a
// stale read (value != k), and gives the slot back with a remote arrive on rank 0's `empty`.
// The give-back uses the variant's release too (the read must be done before the next store).
// Variants (argument `v`):
//   0 PFC   : st.shared::cluster; fence.proxy.async.shared::cluster; arrive.release.cta;
//             try_wait.acquire.cta
//   1 NOF   : st.shared::cluster; arrive.release.cta; try_wait.acquire.cta
//   2 PCTA  : st.shared::cluster; fence.proxy.async.shared::cta; arrive.release.cta
//   3 CLU   : st.shared::cluster; arrive.release.cluster; try_wait.acquire.cluster (control)
//   4 STA   : arrive.expect_tx.relaxed.cluster(4) + st.async complete_tx; acquire.cluster
//             (control)
//   5 FEN   : st.shared::cluster; fence.acq_rel.cluster; arrive.release.cta; acquire.cta
// Noise (argument `noise`): threads 32..(32*W) of both CTAs hammer weak remote stores into the
// other CTA's scratch area for the whole run.
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o handoff handoff.cu
// Run:   ./handoff v rounds P S_ns noise launches
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define SCRATCH 1024  // words of scratch per CTA (noise and queue-pressure stores)

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned mapa(unsigned a, unsigned r) {
    unsigned d;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(r));
    return d;
}
__device__ __forceinline__ void st_remote(unsigned a, unsigned v) {
    asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void arrive_remote(unsigned bar, int v) {
    if (v == 3) {
        asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(bar) : "memory");
    } else {
        asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(bar) : "memory");
    }
}
__device__ __forceinline__ bool try_wait(unsigned bar, unsigned parity, int v) {
    unsigned ok;
    if (v == 3 || v == 4) {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    } else {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    }
    return ok;
}

struct Log { unsigned stale, future, rounds, sm0, sm1, first_round, first_got, pad; };

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256)
k(int v, int rounds, int P, unsigned sns, int noise, unsigned tag, Log* L, volatile int* stop) {
    __shared__ __align__(8) unsigned long long full, empty;
    __shared__ unsigned slot;
    __shared__ unsigned scratch[SCRATCH];
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    unsigned cl = blockIdx.x / 2;
    unsigned fullA = (unsigned)__cvta_generic_to_shared(&full);
    unsigned emptyA = (unsigned)__cvta_generic_to_shared(&empty);
    unsigned slotA = (unsigned)__cvta_generic_to_shared(&slot);
    unsigned scrA = (unsigned)__cvta_generic_to_shared(scratch);
    unsigned peer = rank ^ 1;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(fullA));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(emptyA));
        slot = 0xffffffffu;
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned rslot = mapa(slotA, peer), rfull = mapa(fullA, peer), rempty = mapa(emptyA, peer);
    unsigned rscr = mapa(scrA, peer);
    if (threadIdx.x >= 32) {
        if (noise) {
            unsigned i = threadIdx.x;
            while (!*stop) {
                st_remote(rscr + 4 * (i % SCRATCH), i);
                i += 224;
            }
        }
    } else if (threadIdx.x == 0 && rank == 0) {
        unsigned smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        L[cl].sm0 = smid;
        for (int r = 0; r < rounds; r++) {
            unsigned val = tag + r;
            for (int p = 0; p < P; p++) st_remote(rscr + 4 * (p % SCRATCH), val);
            if (v == 4) {
                asm volatile("mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [%0], 4;" :: "r"(rfull) : "memory");
                asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" :: "r"(rslot), "r"(val), "r"(rfull) : "memory");
            } else {
                st_remote(rslot, val);
                if (v == 0) asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
                if (v == 2) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (v == 5) asm volatile("fence.acq_rel.cluster;" ::: "memory");
                arrive_remote(rfull, v);
            }
            while (!try_wait(emptyA, r & 1, v)) {}
        }
        *stop = 1;
    } else if (threadIdx.x == 0 && rank == 1) {
        unsigned smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        L[cl].sm1 = smid;
        unsigned stale = 0, future = 0;
        for (int r = 0; r < rounds; r++) {
            while (!try_wait(fullA, r & 1, v)) {}
            if (sns) {
                unsigned long long t0 = gtime();
                while (gtime() - t0 < sns) {}
            }
            unsigned got = *(volatile unsigned*)&slot;
            unsigned want = tag + r;
            if (got != want) {
                if (stale + future == 0) {
                    L[cl].first_round = r;
                    L[cl].first_got = got;
                }
                if ((int)(got - want) > 0 && got != 0xffffffffu) future++; else stale++;
            }
            if (v == 4) {
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" :: "r"(rempty) : "memory");
            } else {
                if (v == 5) asm volatile("fence.acq_rel.cluster;" ::: "memory");
                arrive_remote(rempty, v);
            }
        }
        L[cl].stale = stale;
        L[cl].future = future;
        L[cl].rounds = rounds;
    }
    // Remote stores (noise) and arrivals must land before either CTA exits.
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s v rounds P S_ns noise launches\n", argv[0]);
        return 2;
    }
    int v = atoi(argv[1]), rounds = atoi(argv[2]), P = atoi(argv[3]);
    unsigned sns = atoi(argv[4]);
    int noise = atoi(argv[5]), launches = atoi(argv[6]);
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int clusters = sms / 2;
    Log* L;
    int* stop;
    cudaMalloc(&L, clusters * sizeof(Log));
    cudaMalloc(&stop, sizeof(int));
    Log* h = (Log*)malloc(clusters * sizeof(Log));
    unsigned long long stale = 0, future = 0, total = 0, cl_stale = 0, round0 = 0;
    for (int it = 0; it < launches; it++) {
        cudaMemset(L, 0, clusters * sizeof(Log));
        cudaMemset(stop, 0, sizeof(int));
        unsigned tag = 1 + (unsigned)it * (unsigned)rounds;
        k<<<clusters * 2, 256>>>(v, rounds, P, sns, noise, tag, L, stop);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("launch %d: %s\n", it, cudaGetErrorString(e));
            return 1;
        }
        cudaMemcpy(h, L, clusters * sizeof(Log), cudaMemcpyDeviceToHost);
        for (int c = 0; c < clusters; c++) {
            stale += h[c].stale;
            future += h[c].future;
            total += h[c].rounds;
            cl_stale += h[c].stale + h[c].future > 0;
            if (h[c].stale + h[c].future > 0 && h[c].first_round == 0) round0++;
            if (h[c].stale + h[c].future > 0 && cl_stale <= 3)
                printf("  bad read: launch %d cluster %d first at round %u (got %#x, want %#x)\n", it, c,
                       h[c].first_round, h[c].first_got, tag + h[c].first_round);
        }
    }
    printf("v=%d rounds=%d P=%d S=%uns noise=%d launches=%d clusters/launch=%d: hand-offs %llu, stale %llu, future %llu, clusters with a bad read %llu (first at round 0: %llu)\n",
           v, rounds, P, sns, noise, launches, clusters, total, stale, future, cl_stale, round0);
    return 0;
}
