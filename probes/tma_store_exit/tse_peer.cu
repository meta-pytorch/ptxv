/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Stronger E-BULK-EXIT test (index L4, DeepEP k_dispatch / k_combine_reduce: UBLKCP.G.S, then
// exit with no wait_group.read). tma_store_exit.cu stores to local memory; DeepEP's bulk
// stores go to peer GPUs over NVLink, where the bulk engine may read the source later.
// Same shape as tma_store_exit.cu (32 waves of CTAs, ~100 KiB smem, each CTA refills the smem
// the previous CTA on the SM stored from), with:
//   dst 0: local memory, dst 1: the peer GPU's memory (peer access)
//   noise 1: co-tenant kernels on both GPUs (other streams, 16 CTAs each) write 16-byte
//            vectors to the other GPU's memory for the whole run (NVLink congestion)
//   mode 0: exit right after commit_group (DeepEP)
//   mode 1: cp.async.bulk.wait_group.read 0 before exit (the PTX contract)
//   mode 2: positive control: the CTA overwrites its own source right after the commit
//   mode 3: positive control with a delay: overwrite D ns after the commit (how late the
//           bulk engine still reads the source)
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o tse_peer tse_peer.cu
// Run:   CUDA_VISIBLE_DEVICES=<gpu>,<peer gpu> ./tse_peer runs dst noise [D_ns]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int N = 8192;  // words per CTA (32 KiB)

__global__ void k(int mode, unsigned* out, unsigned dns) {
    extern __shared__ __align__(128) unsigned s[];
    for (int i = threadIdx.x; i < N; i += blockDim.x) s[i] = (blockIdx.x << 13) | i;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned sa = (unsigned)__cvta_generic_to_shared(s);
        unsigned long long ga = (unsigned long long)(out + (size_t)blockIdx.x * N);
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(ga), "r"(sa), "r"(N * 4) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        if (mode == 1) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    if (mode == 3) {
        unsigned long long t0, t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < dns);
    }
    if (mode >= 2) {
        __syncthreads();
        for (int i = threadIdx.x; i < N; i += blockDim.x) s[i] = 0xdeadbeef;
    }
}

__global__ void noise_k(uint4* dst, size_t n, volatile int* stop) {
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
    uint4 v = make_uint4(1, 2, 3, 4);
    while (!*stop)
        for (size_t i = gid; i < n; i += st) dst[i] = v;
}

int main(int argc, char** argv) {
    int runs = atoi(argv[1]), dst = atoi(argv[2]), noise = atoi(argv[3]);
    unsigned dns = argc > 4 ? atoi(argv[4]) : 0;
    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int ctas = sms * 32, smem = 100 * 1024;
    CK(cudaSetDevice(1));
    CK(cudaDeviceEnablePeerAccess(0, 0));
    CK(cudaSetDevice(0));
    CK(cudaDeviceEnablePeerAccess(1, 0));
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    size_t words = (size_t)ctas * N;
    unsigned* d;
    if (dst) { CK(cudaSetDevice(1)); CK(cudaMalloc(&d, words * 4)); CK(cudaSetDevice(0)); }
    else CK(cudaMalloc(&d, words * 4));
    size_t nn = (256u << 20) / 16;
    uint4 *n0 = 0, *n1 = 0;
    int* stop;
    CK(cudaMallocHost(&stop, 4));
    cudaStream_t s0, s1;
    if (noise) {
        CK(cudaSetDevice(1)); CK(cudaMalloc(&n1, nn * 16)); CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
        CK(cudaSetDevice(0)); CK(cudaMalloc(&n0, nn * 16)); CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    }
    std::vector<unsigned> h(words);
    int m0 = argc > 5 ? atoi(argv[5]) : 0, m1 = argc > 6 ? atoi(argv[6]) : 3;
    for (int mode = m0; mode <= m1; mode++) {
        long bad_runs = 0, bad_words = 0;
        for (int r = 0; r < runs; r++) {
            CK(cudaMemset(d, 0xff, words * 4));
            CK(cudaDeviceSynchronize());
            if (noise) {
                *stop = 0;
                CK(cudaSetDevice(1)); noise_k<<<16, 1024, 0, s1>>>(n0, nn, stop);  // GPU 1 -> GPU 0
                CK(cudaSetDevice(0)); noise_k<<<16, 1024, 0, s0>>>(n1, nn, stop);  // GPU 0 -> GPU 1
            }
            k<<<ctas, 256, smem>>>(mode, d, dns);
            CK(cudaStreamSynchronize(0));
            if (noise) {
                *stop = 1;
                CK(cudaDeviceSynchronize());
                CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize()); CK(cudaSetDevice(0));
            }
            CK(cudaMemcpy(h.data(), d, words * 4, cudaMemcpyDefault));
            long bad = 0;
            for (size_t i = 0; i < words; i++)
                if (h[i] != ((unsigned)(i / N) << 13 | (unsigned)(i % N))) bad++;
            bad_runs += bad > 0;
            bad_words += bad;
        }
        printf("dst %s noise %d mode %d D %uns: %d CTAs x %d runs: %ld runs with wrong words (%ld words)\n",
               dst ? "peer" : "local", noise, mode, dns, ctas, runs, bad_runs, bad_words);
    }
    return 0;
}
