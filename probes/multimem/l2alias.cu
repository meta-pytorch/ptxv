// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Sensitive reverse-shape litmus for E-MC-ALIAS-HW (index L2: dist_gemm
// Sm100GemmRsMultimemKernel: TMA unicast store, wait_group, release flag / acquire poll,
// multimem.ld_reduce of the same bytes with no fence.proxy.alias).
// mm litmus2 saw 0 stale lanes in its controls too (4 KiB blocks, an all-member poll). Here:
// one writer (member 0) per run, 64 writer CTAs each storing 128 KiB (8 bulk copies of 16 KiB)
// to its own unicast buffer, then +1 on the CTA's multicast flag; 64 reader CTAs on member R
// (R = 0: the writer's GPU, other streams; R > 0: a peer) each poll their own unicast flag
// copy for one writer CTA and immediately multimem.ld_reduce that CTA's 128 KiB, last
// bulk copy first. Every other member's bytes are 0 (written before the run, kernel
// boundary), so each reduced word must equal the writer's pattern word.
//   var 0: wait_group 0; red.release.sys / ld.acquire.sys        (L2's shape, no alias fence)
//   var 1: wait_group.read 0; red.relaxed / ld.relaxed           (positive control: the
//          writes may land after the flag)
//   var 2: var 0 + fence.proxy.alias before the red and after the poll (strict PTX)
//   var 3: st.global.v4 from all threads, bar.sync, red.release / ld.acquire (generic proxy)
//   var 4: wait_group.read 0; red.release / ld.acquire          (control: release without
//          waiting for the bulk writes)
//   var 5: red.relaxed BEFORE the bulk copies; ld.relaxed       (positive control of the
//          detector: the flag precedes the data)
// noise != 0: a co-tenant kernel on every member streams a 1 GiB RMW (other buffers) during
// the run.
// Build: nvcc -O2 -gencode arch=compute_103a,code=sm_103a -o l2alias l2alias.cu -lcuda
// Run:   CUDA_VISIBLE_DEVICES=0,1[,3] ./l2alias R runs var noise [spin_ns (var 5: wait between flag and data)]
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

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
#define RK(x)                                                                               \
    do {                                                                                    \
        cudaError_t e_ = (x);                                                               \
        if (e_ != cudaSuccess) {                                                            \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)

constexpr int G = 64;                 // writer / reader CTAs
constexpr int CH = 128 * 1024;        // bytes per writer CTA
constexpr int CHW = CH / 4;
constexpr size_t FLAGS = (size_t)G * CH;  // flag offset

__device__ __forceinline__ uint32_t pat(uint32_t tag, int c, int i) { return tag ^ (uint32_t)(c * CHW + i) * 2654435761u; }

__global__ void k_writer(uint32_t *data_uc, uint32_t *flag_mc, uint32_t tag, int var, unsigned spin) {
    extern __shared__ __align__(128) uint32_t sb[];
    int c = blockIdx.x;
    if (var == 3) {
        uint4 *d = (uint4 *)(data_uc + (size_t)c * CHW);
        for (int i = threadIdx.x; i < CHW / 4; i += blockDim.x) {
            uint4 v = make_uint4(pat(tag, c, 4 * i), pat(tag, c, 4 * i + 1), pat(tag, c, 4 * i + 2), pat(tag, c, 4 * i + 3));
            asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(d + i), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
        }
        __syncthreads();
    } else {
        for (int i = threadIdx.x; i < CHW; i += blockDim.x) sb[i] = pat(tag, c, i);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned sa = (unsigned)__cvta_generic_to_shared(sb);
            if (var == 5)  // flag first: the reader may reduce before any byte landed
                asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], 1;" ::"l"(flag_mc + c) : "memory");
            if (var == 5 && spin) {
                unsigned long long t0, t;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
                do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < spin);
            }
            for (int k = 0; k < 8; k++)
                asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 16384;" ::"l"(data_uc + (size_t)c * CHW + k * 4096),
                             "r"(sa + k * 16384) : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            if (var == 1 || var == 4 || var == 5)
                asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            else
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
    }
    if (threadIdx.x == 0 && var != 5) {
        if (var == 2) asm volatile("fence.proxy.alias;" ::: "memory");
        if (var == 1)
            asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], 1;" ::"l"(flag_mc + c) : "memory");
        else
            asm volatile("multimem.red.release.sys.global.add.u32 [%0], 1;" ::"l"(flag_mc + c) : "memory");
    }
}

__global__ void k_reader(const uint32_t *data_mc, const uint32_t *flag_uc, uint32_t tag, int var, unsigned *bad) {
    int c = blockIdx.x;
    __shared__ int ok;
    if (threadIdx.x == 0) {
        uint32_t f = 0;
        long spins = 0;
        while (f < 1 && spins < (1L << 28)) {
            if (var == 1 || var == 5)
                asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(f) : "l"(flag_uc + c) : "memory");
            else
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(f) : "l"(flag_uc + c) : "memory");
            spins++;
        }
        if (var == 2) asm volatile("fence.proxy.alias;" ::: "memory");
        ok = f >= 1;
        if (!ok) atomicAdd(bad + 1, 1u);
    }
    __syncthreads();
    if (!ok) return;
    unsigned nb = 0;
    for (int i = CHW - 1 - threadIdx.x; i >= 0; i -= blockDim.x) {  // last bulk copy first
        uint32_t got;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];" : "=r"(got) : "l"(data_mc + (size_t)c * CHW + i) : "memory");
        nb += got != pat(tag, c, i);
    }
    __shared__ unsigned cta_bad;
    if (threadIdx.x == 0) cta_bad = 0;
    __syncthreads();
    if (nb) {
        atomicAdd(bad, nb);
        cta_bad = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && cta_bad) atomicAdd(bad + 2, 1u);
}

__global__ void k_noise(uint32_t *buf, size_t n, volatile int *stop) {
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
    for (int it = 0; it < 1000 && !*stop; it++)
        for (size_t i = gid; i < n; i += st * 8) buf[i] += 1;
}

int main(int argc, char **argv) {
    int R = atoi(argv[1]), runs = atoi(argv[2]), var = atoi(argv[3]), noise = argc > 4 ? atoi(argv[4]) : 0;
    unsigned spin = argc > 5 ? atoi(argv[5]) : 0;
    CK(cuInit(0));
    int nm = 0;
    CK(cuDeviceGetCount(&nm));
    std::vector<CUcontext> ctx(nm);
    for (int d = 0; d < nm; d++) CK(cuDevicePrimaryCtxRetain(&ctx[d], d));
    CUmulticastObjectProp p = {};
    p.numDevices = nm;
    p.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t want = FLAGS + (2u << 20), gran = 0;
    p.size = want;
    CK(cuCtxSetCurrent(ctx[0]));
    CK(cuMulticastGetGranularity(&gran, &p, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    size_t size = (want + gran - 1) / gran * gran;
    p.size = size;
    CUmemGenericAllocationHandle mc;
    CK(cuMulticastCreate(&mc, &p));
    for (int d = 0; d < nm; d++) CK(cuMulticastAddDevice(mc, d));
    std::vector<CUdeviceptr> uc(nm), mv(nm);
    for (int d = 0; d < nm; d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        CUmemAllocationProp ap = {};
        ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ap.location.id = d;
        ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        CUmemGenericAllocationHandle h;
        CK(cuMemCreate(&h, size, &ap, 0));
        CK(cuMulticastBindMem(mc, 0, h, 0, size, 0));
        CUmemAccessDesc ad = {};
        ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ad.location.id = d;
        ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CK(cuMemAddressReserve(&uc[d], size, 0, 0, 0));
        CK(cuMemMap(uc[d], size, 0, h, 0));
        CK(cuMemSetAccess(uc[d], size, &ad, 1));
        CK(cuMemAddressReserve(&mv[d], size, 0, 0, 0));
        CK(cuMemMap(mv[d], size, 0, mc, 0));
        CK(cuMemSetAccess(mv[d], size, &ad, 1));
    }
    std::vector<cudaStream_t> sw(nm), sr(nm), sn(nm);
    std::vector<uint32_t *> nbuf(nm);
    std::vector<int *> stop(nm);
    unsigned *bad;
    for (int d = 0; d < nm; d++) {
        RK(cudaSetDevice(d));
        CK(cuCtxSetCurrent(ctx[d]));
        RK(cudaStreamCreateWithFlags(&sw[d], cudaStreamNonBlocking));
        RK(cudaStreamCreateWithFlags(&sr[d], cudaStreamNonBlocking));
        RK(cudaStreamCreateWithFlags(&sn[d], cudaStreamNonBlocking));
        if (noise) {
            RK(cudaMalloc(&nbuf[d], 1u << 30));
            RK(cudaMallocManaged(&stop[d], 4));
        }
    }
    RK(cudaSetDevice(R));
    CK(cuCtxSetCurrent(ctx[R]));
    RK(cudaMalloc(&bad, 12));
    RK(cudaSetDevice(0));
    CK(cuCtxSetCurrent(ctx[0]));
    RK(cudaFuncSetAttribute(k_writer, cudaFuncAttributeMaxDynamicSharedMemorySize, CH));
    unsigned long tot = 0, ctas_bad = 0, runs_bad = 0, to = 0;
    for (int run = 0; run < runs; run++) {
        uint32_t tag = 0x7e000000u ^ ((uint32_t)run << 4) ^ (uint32_t)var;
        for (int d = 0; d < nm; d++) {
            CK(cuCtxSetCurrent(ctx[d]));
            CK(cuMemsetD8(uc[d], 0, size));
            CK(cuCtxSynchronize());
        }
        RK(cudaSetDevice(R));
        CK(cuCtxSetCurrent(ctx[R]));
        RK(cudaMemset(bad, 0, 12));
        RK(cudaDeviceSynchronize());
        if (noise)
            for (int d = 0; d < nm; d++) {
                RK(cudaSetDevice(d));
                CK(cuCtxSetCurrent(ctx[d]));
                *stop[d] = 0;
                k_noise<<<24, 512, 0, sn[d]>>>(nbuf[d], (1u << 30) / 4, stop[d]);
            }
        RK(cudaSetDevice(R));
        CK(cuCtxSetCurrent(ctx[R]));
        k_reader<<<G, 512, 0, sr[R]>>>((const uint32_t *)mv[R], (const uint32_t *)(uc[R] + FLAGS), tag, var, bad);
        RK(cudaSetDevice(0));
        CK(cuCtxSetCurrent(ctx[0]));
        k_writer<<<G, 512, CH, sw[0]>>>((uint32_t *)uc[0], (uint32_t *)(mv[0] + FLAGS), tag, var, spin);
        RK(cudaSetDevice(R));
        CK(cuCtxSetCurrent(ctx[R]));
        RK(cudaStreamSynchronize(sr[R]));
        RK(cudaSetDevice(0));
        CK(cuCtxSetCurrent(ctx[0]));
        RK(cudaStreamSynchronize(sw[0]));
        if (noise)
            for (int d = 0; d < nm; d++) {
                *stop[d] = 1;
                RK(cudaSetDevice(d));
                CK(cuCtxSetCurrent(ctx[d]));
                RK(cudaStreamSynchronize(sn[d]));
            }
        unsigned h[3];
        RK(cudaSetDevice(R));
        CK(cuCtxSetCurrent(ctx[R]));
        RK(cudaMemcpy(h, bad, 12, cudaMemcpyDeviceToHost));
        tot += h[0];
        to += h[1];
        ctas_bad += h[2];
        runs_bad += h[0] > 0;
    }
    printf("members %d reader %d var %d noise %d spin %uns: runs %d, words/run %d: stale words %lu, CTAs with stale %lu / %lu, runs with stale %lu, timeouts %lu\n",
           nm, R, var, noise, spin, runs, G * CHW, tot, ctas_bad, (unsigned long)runs * G, runs_bad, to);
    return 0;
}
