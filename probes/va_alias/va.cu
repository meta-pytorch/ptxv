// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Virtual aliases within one process (S-VA-ALIAS): one cuMemCreate allocation mapped at two
// virtual addresses A and B with cuMemMap. Litmus shapes, each without and with
// fence.proxy.alias, over RUNS launches:
//
//   raw   same thread: ld.global [B] (warms L1 under B), st.global [A], (fence),
//         ld.global [B]. Stale: the second load returns the old value.
//   mp    message passing across SMs: block 0 writes N data words through A, (fence), then
//         st.release.gpu flag through A. Every other block warms its data through B, polls
//         the flag through B with ld.acquire.gpu, (fence), reads the data through B.
//   Controls: raw-same (both loads through A: coherence, never stale), raw-nc (second load
//   ld.global.nc through B: the non-coherent path, shows the detector sees a stale line),
//   mp-relaxed (relaxed flag, through B: shows the detector sees late data).
//
//   nvcc -O2 -arch=sm_103a -o va va.cu -lcuda
//   ./va RUNS                     one line per shape / variant: runs, stale words, runs with any
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x)                                                                       \
    do {                                                                            \
        CUresult r_ = (x);                                                          \
        if (r_ != CUDA_SUCCESS) {                                                   \
            const char *s_ = 0;                                                     \
            cuGetErrorString(r_, &s_);                                              \
            fprintf(stderr, "%s:%d %s -> %d %s\n", __FILE__, __LINE__, #x, r_, s_); \
            exit(1);                                                                \
        }                                                                           \
    } while (0)
#define RK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t r_ = (x);                                                                    \
        if (r_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(r_)); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

// mode bit 0: fence.proxy.alias; raw: bit 1 second load through A, bit 2 second load .nc.
__global__ void k_raw(uint32_t *A, uint32_t *B, uint32_t *stale, uint32_t v, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t r0, r1;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(r0) : "l"(B + i) : "memory");
    // Keep the warming load (ptxas deletes a dead ld.global, inline asm or not).
    if (r0 == 0xffffffffu) atomicAdd(stale + 1, 1);
    asm volatile("st.global.u32 [%0], %1;" ::"l"(A + i), "r"(v) : "memory");
    if (mode & 1) asm volatile("fence.proxy.alias;" ::: "memory");
    if (mode & 2)
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(r1) : "l"(A + i) : "memory");
    else if (mode & 4)
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(r1) : "l"(B + i) : "memory");
    else
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(r1) : "l"(B + i) : "memory");
    if (r1 != v) atomicAdd(stale, 1);
}

// CTA shape: every thread warms word j = (i + 32) % n through B, writes word i through A,
// bar.sync, reads word j through B. mode bit 0: writer fence after its store, bit 1: reader
// fence after bar.sync, bit 2: both through A (control: same mapping).
__global__ void k_cta(uint32_t *A, uint32_t *B, uint32_t *stale, uint32_t v, int mode) {
    int n = blockDim.x, i = threadIdx.x, j = (i + 32) % n, base = blockIdx.x * n;
    uint32_t *R = (mode & 4) ? A : B;
    uint32_t r0, r1;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(r0) : "l"(R + base + j) : "memory");
    if (r0 == 0xffffffffu) atomicAdd(stale + 1, 1);
    asm volatile("st.global.u32 [%0], %1;" ::"l"(A + base + i), "r"(v) : "memory");
    if (mode & 1) asm volatile("fence.proxy.alias;" ::: "memory");
    __syncthreads();
    if (mode & 2) asm volatile("fence.proxy.alias;" ::: "memory");
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(r1) : "l"(R + base + j) : "memory");
    if (r1 != v) atomicAdd(stale, 1);
}

// mode bit 0: producer fence, bit 1: consumer fence, bit 2: relaxed flag (control).
#define NDATA 256
__global__ void k_mp(uint32_t *A, uint32_t *B, uint32_t *stale, uint32_t v, int mode) {
    uint32_t *flagA = A + 4096, *flagB = B + 4096;
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            // Let consumers warm their lines first.
            uint64_t t0 = clock64();
            while (clock64() - t0 < 2000000) {
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < NDATA; i += blockDim.x)
            asm volatile("st.global.u32 [%0], %1;" ::"l"(A + i), "r"(v) : "memory");
        __syncthreads();
        if (threadIdx.x == 0) {
            asm volatile("fence.acq_rel.gpu;" ::: "memory");  // block's stores before the flag
            if (mode & 1) asm volatile("fence.proxy.alias;" ::: "memory");
            if (mode & 4)
                asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(flagA), "r"(v) : "memory");
            else
                asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(flagA), "r"(v) : "memory");
        }
        return;
    }
    __shared__ uint32_t warm;
    uint32_t acc = 0;
    uint32_t *R = (mode & 8) ? A : B;  // bit 3: the consumer reads through A (same mapping)
    for (int i = threadIdx.x; i < NDATA; i += blockDim.x) {
        uint32_t x;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(R + i) : "memory");
        acc += x;
    }
    if (threadIdx.x == 0) {
        uint32_t f;
        do {
            if (mode & 4)
                asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(flagB) : "memory");
            else
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(flagB) : "memory");
        } while (f != v);
        if (mode & 2) asm volatile("fence.proxy.alias;" ::: "memory");
        warm = acc;
    }
    if (acc == 0xffffffffu) atomicAdd(stale + 1, 1);  // keep the warming loads
    __syncthreads();
    if (mode & 4) {
    } else {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");  // thread 0's acquire for the block
    }
    for (int i = threadIdx.x; i < NDATA; i += blockDim.x) {
        uint32_t x;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(R + i) : "memory");
        if (x != v) atomicAdd(stale, 1);
    }
}

int main(int argc, char **argv) {
    int runs = argc > 1 ? atoi(argv[1]) : 100;
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    size_t gran = 0;
    CK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t size = gran;
    CUmemGenericAllocationHandle h;
    CK(cuMemCreate(&h, size, &prop, 0));
    CUdeviceptr a, b;
    CK(cuMemAddressReserve(&a, size, 0, 0, 0));
    CK(cuMemAddressReserve(&b, size, 0, 0, 0));
    CK(cuMemMap(a, size, 0, h, 0));
    CK(cuMemMap(b, size, 0, h, 0));
    CUmemAccessDesc acc = {};
    acc.location = prop.location;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CK(cuMemSetAccess(a, size, &acc, 1));
    CK(cuMemSetAccess(b, size, &acc, 1));
    uint32_t *A = (uint32_t *)a, *B = (uint32_t *)b, *stale;
    if (argc > 2 && !strcmp(argv[2], "plain")) {
        // Detector check: A = B = one cudaMalloc buffer (no aliasing).
        RK(cudaMalloc(&A, size));
        B = A;
    }
    RK(cudaMalloc(&stale, 8));
    // Sanity: a write through A is read through B after a kernel boundary.
    RK(cudaMemset(A, 0x5a, 4));
    uint32_t chk = 0;
    RK(cudaMemcpy(&chk, B, 4, cudaMemcpyDeviceToHost));
    if (chk != 0x5a5a5a5a) {
        fprintf(stderr, "A and B are not one allocation (%#x)\n", chk);
        return 1;
    }
    cudaDeviceProp dp;
    RK(cudaGetDeviceProperties(&dp, 0));
    printf("device %s, %d SMs, granularity %zu, A %p B %p, runs %d\n", dp.name,
           dp.multiProcessorCount, gran, (void *)A, (void *)B, runs);
    struct S {
        const char *name;
        int kind, mode;
    } shapes[] = {
        {"raw nofence", 0, 0},       {"raw fence", 0, 1},          {"raw-same (control)", 0, 2},
        {"raw-nc (control)", 0, 4},  {"mp nofence", 1, 0},         {"mp producer-fence", 1, 1},
        {"mp consumer-fence", 1, 2}, {"mp both-fence", 1, 3},      {"mp-relaxed (control)", 1, 4},
        {"mp-relaxed-same (control)", 1, 12},
        {"cta nofence", 2, 0},       {"cta writer-fence", 2, 1},   {"cta reader-fence", 2, 2},
        {"cta-same (control)", 2, 4},
    };
    int threads = 256, blocks = 4 * dp.multiProcessorCount;
    for (auto &s : shapes) {
        unsigned long long total = 0;
        int bad_runs = 0;
        for (int r = 0; r < runs; r++) {
            RK(cudaMemset(A, 0, size));
            RK(cudaMemset(stale, 0, 8));
            uint32_t v = 0x1000 + r;
            if (s.kind == 0)
                k_raw<<<blocks, threads>>>(A, B, stale, v, s.mode);
            else if (s.kind == 2)
                k_cta<<<blocks, threads>>>(A, B, stale, v, s.mode);
            else
                k_mp<<<dp.multiProcessorCount, 128>>>(A, B, stale, v, s.mode);
            RK(cudaGetLastError());
            RK(cudaDeviceSynchronize());
            uint32_t n = 0;
            RK(cudaMemcpy(&n, stale, 4, cudaMemcpyDeviceToHost));
            total += n;
            bad_runs += n != 0;
        }
        unsigned long long per = s.kind != 1 ? (unsigned long long)blocks * threads
                                             : (unsigned long long)(dp.multiProcessorCount - 1) * NDATA;
        printf("%-22s runs %d  words/run %llu  stale words %llu  runs with stale %d\n", s.name, runs,
               per, total, bad_runs);
    }
    return 0;
}
