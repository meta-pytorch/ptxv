/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// E-DESC-VERSION probe: does GB300 decode a tcgen05 shared-memory matrix descriptor the same
// way whatever its bits 46-48 hold? The PTX ISA fixes them at 0b001; Triton 3.5 leaves them 0.
// One CTA of 128 threads per config runs one tcgen05.mma.cta_group::1 (M = 128, N = 64, one
// K step) from shared-memory images A and B (32 KiB each, random finite values) through the
// descriptors of the config, and dumps D (128 TMEM lanes x 64 f32 columns).
//
// A config is (kind, swizzle layout, A / B major, LBO, SBO). Every config runs with A's bits
// 46-48 = va and B's = vb (command line); run.sh compares the dumps of every (va, vb) with the
// dump of (1, 1) bitwise.
//   nvcc -O3 -gencode arch=compute_103a,code=sm_103a -o desc_version desc_version.cu
//   desc_version <va> <vb> <out.bin>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#define NCOL 64

template <int KIND>
__global__ void __launch_bounds__(128)
    mma(const uint4* __restrict__ A, const uint4* __restrict__ B, float* __restrict__ D,
        unsigned long long ahi, unsigned long long bhi, unsigned idesc) {
    extern __shared__ __align__(1024) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sB = smem + 32768;
    __shared__ __align__(8) unsigned long long bar;
    __shared__ unsigned tslot;
    int tid = threadIdx.x, w = tid / 32;
    for (int i = tid; i < 2048; i += 128) {
        ((uint4*)sA)[i] = A[i];
        ((uint4*)sB)[i] = B[i];
    }
    unsigned bara = (unsigned)__cvta_generic_to_shared(&bar);
    unsigned slot = (unsigned)__cvta_generic_to_shared(&tslot);
    if (w == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;" ::"r"(slot));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bara));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    unsigned tm = *(volatile unsigned*)&tslot;
    if (tid == 0) {
        unsigned long long ad = (unsigned long long)((unsigned)__cvta_generic_to_shared(sA) >> 4) | ahi;
        unsigned long long bd = (unsigned long long)((unsigned)__cvta_generic_to_shared(sB) >> 4) | bhi;
        if (KIND == 0)
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;" ::"r"(tm),
                         "l"(ad), "l"(bd), "r"(idesc)
                         : "memory");
        else
            asm volatile("tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 0;" ::"r"(tm),
                         "l"(ad), "l"(bd), "r"(idesc)
                         : "memory");
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bara)
                     : "memory");
    }
    asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra W;\n}" ::"r"(bara)
                 : "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    unsigned ta = tm + ((unsigned)(32 * w) << 16);
    float* d = D + tid * NCOL;
#pragma unroll
    for (int c = 0; c < NCOL; c += 16) {
        unsigned r[16];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, "
            "%11, %12, %13, %14, %15}, [%16];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
              "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]),
              "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
            : "r"(ta + c)
            : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
        for (int j = 0; j < 16; j++) d[c + j] = __uint_as_float(r[j]);
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (w == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" ::"r"(tm));
    }
}

struct Cfg {
    const char* kind;
    int k;        // 0 kind::f16 (bf16), 1 kind::f8f6f4 (e4m3)
    int layout;   // bits 61-63: 0 none, 1 128B base 32B, 2 128B, 4 64B, 6 32B
    int amaj, bmaj;
    unsigned lbo, sbo;
};

static unsigned long long hi(const Cfg& c, unsigned v) {
    return ((unsigned long long)(c.lbo >> 4) << 16) | ((unsigned long long)(c.sbo >> 4) << 32) |
           ((unsigned long long)(v & 7) << 46) | ((unsigned long long)c.layout << 61);
}

static uint16_t bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: desc_version <va> <vb> <out.bin> [list]\n");
        return 2;
    }
    unsigned va = atoi(argv[1]), vb = atoi(argv[2]);
    std::vector<Cfg> cfgs;
    const int layouts[] = {0, 1, 2, 4, 6};
    // (LBO, SBO): the canonical value for the layout and a second, odd one.
    for (int k = 0; k < 2; k++)
        for (int lay : layouts)
            for (int am = 0; am < 2; am++)
                for (int bm = 0; bm < 2; bm++) {
                    unsigned span = lay == 0 ? 16 : lay == 6 ? 32 : lay == 4 ? 64 : 128;
                    cfgs.push_back({k ? "f8" : "bf16", k, lay, am, bm, 128, 8 * span});
                    cfgs.push_back({k ? "f8" : "bf16", k, lay, am, bm, 1024, 2 * 8 * span});
                }
    if (argc > 4) {
        for (size_t i = 0; i < cfgs.size(); i++)
            printf("%zu %s layout=%d amaj=%d bmaj=%d lbo=%u sbo=%u\n", i, cfgs[i].kind,
                   cfgs[i].layout, cfgs[i].amaj, cfgs[i].bmaj, cfgs[i].lbo, cfgs[i].sbo);
        return 0;
    }
    // Inputs: bf16 normal(0,1)-ish values; e4m3 codes with the NaN codes avoided.
    std::vector<uint16_t> abf(16384), bbf(16384);
    std::vector<uint8_t> a8(32768), b8(32768);
    uint64_t s = 0x243f6a8885a308d3ull;
    auto rnd = [&]() {
        s ^= s << 13, s ^= s >> 7, s ^= s << 17;
        return s;
    };
    for (auto& x : abf) x = bf16(((int)(rnd() % 2001) - 1000) / 400.0f);
    for (auto& x : bbf) x = bf16(((int)(rnd() % 2001) - 1000) / 400.0f);
    for (auto& x : a8) x = (uint8_t)(rnd() % 0x7f) | (uint8_t)(rnd() & 0x80);
    for (auto& x : b8) x = (uint8_t)(rnd() % 0x7f) | (uint8_t)(rnd() & 0x80);
    uint4 *dA, *dB;
    float* dD;
    cudaMalloc(&dA, 32768);
    cudaMalloc(&dB, 32768);
    cudaMalloc(&dD, 128 * NCOL * 4);
    cudaFuncSetAttribute(mma<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    cudaFuncSetAttribute(mma<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    FILE* out = fopen(argv[3], "wb");
    std::vector<float> h(128 * NCOL);
    for (size_t i = 0; i < cfgs.size(); i++) {
        const Cfg& c = cfgs[i];
        cudaMemcpy(dA, c.k ? (void*)a8.data() : (void*)abf.data(), 32768, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, c.k ? (void*)b8.data() : (void*)bbf.data(), 32768, cudaMemcpyHostToDevice);
        cudaMemset(dD, 0xff, 128 * NCOL * 4);
        // idesc: D f32 (bit 4); kind::f16 A / B bf16 (bits 7, 10); majors (15, 16); N >> 3
        // at 17; M >> 4 at 24.
        unsigned idesc = (1u << 4) | (c.k ? 0 : (1u << 7) | (1u << 10)) | (c.amaj << 15) |
                         (c.bmaj << 16) | ((64 >> 3) << 17) | ((128 >> 4) << 24);
        if (c.k)
            mma<1><<<1, 128, 65536>>>(dA, dB, dD, hi(c, va), hi(c, vb), idesc);
        else
            mma<0><<<1, 128, 65536>>>(dA, dB, dD, hi(c, va), hi(c, vb), idesc);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("cfg %zu: %s\n", i, cudaGetErrorString(e));
            return 1;
        }
        cudaMemcpy(h.data(), dD, h.size() * 4, cudaMemcpyDeviceToHost);
        fwrite(h.data(), 4, h.size(), out);
    }
    fclose(out);
    printf("va=%u vb=%u: %zu configs written\n", va, vb, cfgs.size());
    return 0;
}
