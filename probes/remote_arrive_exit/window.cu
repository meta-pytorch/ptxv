/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Timing window of the rae.cu fault: cluster (2,1,1), one CTA per SM. Rank 0 initializes an
// mbarrier (count 2), both ranks agree on a time T (globaltimer, rank 0's clock + 20 us,
// broadcast before a cluster barrier). Rank 1 issues `mbarrier.arrive.release.cta.shared::cluster`
// on rank 0's mbarrier at T; rank 0 exits at T + Y ns (Y may be negative: exit first).
// One launch = 76 clusters; the program prints one line per launch: ok or the CUDA error.
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o window window.cu
// Run:   CUDA_VISIBLE_DEVICES=<gpu> ./window Y_ns [clusters]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void __cluster_dims__(2, 1, 1) k(long long y, unsigned long long* lag) {
    extern __shared__ __align__(1024) unsigned long long s[];
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    unsigned base = (unsigned)__cvta_generic_to_shared(s);
    if (threadIdx.x == 0 && rank == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" :: "r"(base + 64));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        unsigned long long T = gtime() + 20000;
        s[0] = T;
        unsigned r;
        asm volatile("mapa.shared::cluster.u32 %0, %1, 1;" : "=r"(r) : "r"(base));
        asm volatile("st.shared::cluster.u64 [%0], %1;" :: "r"(r), "l"(T) : "memory");
    }
    asm volatile("barrier.cluster.arrive.release;\n barrier.cluster.wait.acquire;" ::: "memory");
    unsigned long long T = s[0];
    if (threadIdx.x != 0) return;
    if (rank == 0) {
        unsigned long long e = T + y;
        while (gtime() < e) {}
        return;
    }
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(r) : "r"(base + 64));
    unsigned long long t;
    while ((t = gtime()) < T) {}
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
    lag[blockIdx.x / 2] = t - T;
}

int main(int argc, char** argv) {
    long long y = atoll(argv[1]);
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int ncl = argc > 2 ? atoi(argv[2]) : sms / 2;
    int smem = 200 * 1024;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    unsigned long long* lag;
    cudaMalloc(&lag, ncl * 8);
    k<<<ncl * 2, 32, smem>>>(y, lag);
    cudaError_t e = cudaDeviceSynchronize();
    printf("Y=%lld clusters=%d %s\n", y, ncl, e ? cudaGetErrorString(e) : "ok");
    return e ? 1 : 0;
}
