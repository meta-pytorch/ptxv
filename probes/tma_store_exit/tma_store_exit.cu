/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Probe for async-after-exit with bulk (TMA) stores: may a CTA exit while its
// cp.async.bulk shared->global store is still reading shared memory? cuBLAS's nvjet GEMM
// (fixtures/r2_cublas_gemm) commits its epilogue TMA store and exits without
// cp.async.bulk.wait_group.read; the PTX ISA requires the source to stay valid until the
// read completes. Each CTA fills 32 KiB of shared memory with a CTA-specific pattern, one
// thread stores it with cp.async.bulk and the CTA exits; with ~100 KiB of dynamic shared
// memory per CTA and 16 waves of CTAs, the next CTA on the SM refills the same shared memory
// right away. A store that read freed memory shows up as a wrong word in the output.
//   mode 0: exit right after commit (cuBLAS)
//   mode 1: cp.async.bulk.wait_group.read 0 before exit (the PTX contract)
//   mode 2: positive control: the CTA itself overwrites its shared memory right after commit
// Build: nvcc -arch=sm_103a -O2 -o tma_store_exit tma_store_exit.cu
// Run:   CUDA_VISIBLE_DEVICES=<free gpu> ./tma_store_exit [runs]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int N = 8192;  // words per CTA (32 KiB)

__global__ void k(int mode, unsigned* out) {
    extern __shared__ __align__(128) unsigned s[];
    for (int i = threadIdx.x; i < N; i += blockDim.x) s[i] = (blockIdx.x << 13) | i;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned sa = (unsigned)__cvta_generic_to_shared(s);
        unsigned long long ga = (unsigned long long)(out + (size_t)blockIdx.x * N);
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                     :: "l"(ga), "r"(sa), "r"(N * 4) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        if (mode == 1) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    if (mode == 2) {
        __syncthreads();
        for (int i = threadIdx.x; i < N; i += blockDim.x) s[i] = 0xdeadbeef;
    }
}

int main(int argc, char** argv) {
    int runs = argc > 1 ? atoi(argv[1]) : 200;
    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int ctas = sms * 32;
    int smem = 100 * 1024;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    unsigned* d;
    size_t words = (size_t)ctas * N;
    CK(cudaMalloc(&d, words * 4));
    std::vector<unsigned> h(words);
    for (int mode = 0; mode < 3; mode++) {
        long bad_runs = 0, bad_words = 0;
        for (int r = 0; r < runs; r++) {
            CK(cudaMemset(d, 0xff, words * 4));
            k<<<ctas, 256, smem>>>(mode, d);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(h.data(), d, words * 4, cudaMemcpyDeviceToHost));
            long bad = 0;
            for (size_t i = 0; i < words; i++)
                if (h[i] != ((unsigned)(i / N) << 13 | (unsigned)(i % N))) bad++;
            bad_runs += bad > 0;
            bad_words += bad;
        }
        printf("mode %d: %d CTAs x %d runs: %ld runs with wrong words (%ld words)\n", mode, ctas,
               runs, bad_runs, bad_words);
    }
    return 0;
}
