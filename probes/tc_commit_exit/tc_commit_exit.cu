/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Probe for E-TC-EXIT: may a tcgen05.commit arrive on a CTA's shared-memory mbarrier after
// the CTA exited, i.e. into the next CTA on the SM? flashinfer's trtllm-gen batched GEMM
// (fixtures/r3b_trtllm_bmm) ends its MMA warp with commits to load-stage "empty" barriers
// that no thread waits for, then exits. One CTA per SM (200 KiB of dynamic shared memory),
// 32 waves. Each CTA first initializes mbarrier V at shared offset 0 (count 1), sleeps ~20 us
// and tests phase 0: a completed phase is an arrival from the previous CTA. Then
// warp 0 allocates 256 TMEM columns and thread 0 issues tcgen05.mma (bf16, 128 x 256 x 16,
// zero operands) and commits:
//   mode 0 (trtllm-gen): 8 MMAs; commit F; commit V; everyone waits for F; dealloc; exit
//   mode 1: 8 MMAs; commit F; everyone waits for F; 64 more MMAs; commit V; dealloc; exit
//           (MMAs still in flight at dealloc and exit: outside the PTX contract, it only
//           widens the window)
//   mode 2: positive control of the detector: the CTA arrives on V itself before the test
//           (every CTA counts)
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o tc_commit_exit tc_commit_exit.cu
// Run:   CUDA_VISIBLE_DEVICES=<free gpu> ./tc_commit_exit [runs]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ bool done0(unsigned a) {
    unsigned v;
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], 0;\n"
                 " selp.u32 %0, 1, 0, p;\n}" : "=r"(v) : "r"(a) : "memory");
    return v;
}

__global__ void k(int mode, unsigned* bad) {
    extern __shared__ __align__(1024) unsigned char s[];
    unsigned base = (unsigned)__cvta_generic_to_shared(s);
    unsigned V = base, F = base + 8, tslot = base + 16, A = base + 1024, B = base + 8192;
    for (int i = threadIdx.x; i < 32768 / 4; i += blockDim.x) ((unsigned*)s)[i + 256] = 0;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(V));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(F));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        if (mode == 2)
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(V) : "memory");
        for (int i = 0; i < 20; i++) __nanosleep(1000);
        if (done0(V)) atomicAdd(bad, 1);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" :: "r"(tslot));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    unsigned t = *(volatile unsigned*)(s + 16);
    unsigned long long ad = (unsigned long long)(A >> 4) | 0x0000400800800000ull;
    unsigned long long bd = (unsigned long long)(B >> 4) | 0x0000400810000000ull;
    unsigned idesc = 0x08400490;  // D f32, A/B bf16, K-major, N = 256, M = 128
    if (threadIdx.x == 0) {
        for (int i = 0; i < 8; i++)
            asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %4, 0;\n"
                         " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                         :: "r"(t), "l"(ad), "l"(bd), "r"(idesc), "r"(i));
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(F) : "memory");
        if (mode == 0)
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(V) : "memory");
    }
    asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra W;\n}" :: "r"(F) : "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (mode == 1 && threadIdx.x == 0) {
        for (int i = 0; i < 64; i++)
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;"
                         :: "r"(t), "l"(ad), "l"(bd), "r"(idesc));
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(V) : "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (threadIdx.x < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(t));
}

int main(int argc, char** argv) {
    int runs = argc > 1 ? atoi(argv[1]) : 200;
    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int ctas = sms * 32, smem = 200 * 1024;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    unsigned* d;
    CK(cudaMalloc(&d, 4));
    for (int mode = 0; mode < 3; mode++) {
        long bad_runs = 0, bad_ctas = 0;
        for (int r = 0; r < runs; r++) {
            CK(cudaMemset(d, 0, 4));
            k<<<ctas, 128, smem>>>(mode, d);
            CK(cudaDeviceSynchronize());
            unsigned h;
            CK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
            bad_runs += h > 0;
            bad_ctas += h;
        }
        printf("mode %d: %d CTAs x %d runs: %ld runs with an early phase (%ld CTAs)\n", mode,
               ctas, runs, bad_runs, bad_ctas);
    }
    return 0;
}
