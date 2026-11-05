/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Probe for S-TMA: swizzled TMA at shared-memory destinations that are not aligned to the
// swizzle repeat (8 rows x span), as multicast slices land (cuBLAS
// nvjet_sm103_tst_64x8_64x16_2x4_h_bz_TNT: cluster of 8, each rank loads a 4-row slice of a
// 128B-swizzled tile, rank 1's at stage base 0x2400 + 0x200, multicast to the cluster).
// For every swizzle span S (32/64/128 B) and start offset D (multiples of 128 B, 0..0x780):
//   load:  cluster of 4 CTAs; rank r loads rows [4r, 4r+4) of G (u16 G[q][c] = q * 64 + c,
//          64 x 64), box (S/2 cols x 4 rows), with .multicast::cluster mask 0xf to
//          base + D + r * 4S (base 1024-aligned, smem pre-filled 0xffff). Every CTA's image is
//          compared with two models of where box element (q, c) of slice r lands:
//            abs:  swz(D + 4rS + qS + 2c)    swizzle phase from the shared-memory address
//            rel:  D + 4rS + swz(qS + 2c)    swizzle phase from the box row
//   store: one CTA, smem[i] = i (u16), TMA store of a 4-row box from base + D to G rows 0..3,
//          compared with the same two models.
// Misaligned (not 128-byte) starts: `./tma_mcast_swizzle <span> <D>` runs one load config
// alone (the launch may fault).
// Build: nvcc -arch=sm_103a -O2 -o tma_mcast_swizzle tma_mcast_swizzle.cu -lcuda
// Run:   CUDA_VISIBLE_DEVICES=0 ./tma_mcast_swizzle
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int W = 64, H = 64, ROWS = 4, NCTA = 4, IMG = 4096, SMEM = IMG + 2048;

__device__ unsigned aligned_base(uint16_t* s) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    return (a + 1023) & ~1023u;
}

__global__ void __cluster_dims__(NCTA, 1, 1) k_load(const __grid_constant__ CUtensorMap tm, int span, int d, uint16_t* out) {
    extern __shared__ uint16_t s[];
    __shared__ __align__(8) uint64_t mbar;
    unsigned base = aligned_base(s);
    uint16_t* img = s + (base - (unsigned)__cvta_generic_to_shared(s)) / 2;
    for (int i = threadIdx.x; i < IMG / 2; i += blockDim.x) img[i] = 0xffff;
    unsigned ma = (unsigned)__cvta_generic_to_shared(&mbar);
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(ma));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(ma), "r"(NCTA * ROWS * span) : "memory");
        unsigned dst = base + d + rank * ROWS * span;
        int c0 = 0, r0 = rank * ROWS;
        unsigned short mask = (1 << NCTA) - 1;
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                     :: "r"(dst), "l"(&tm), "r"(c0), "r"(r0), "r"(ma), "h"(mask) : "memory");
    }
    asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra W;\n}" :: "r"(ma) : "memory");
    for (int i = threadIdx.x; i < IMG / 2; i += blockDim.x) out[rank * IMG / 2 + i] = img[i];
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__global__ void k_store(const __grid_constant__ CUtensorMap tm, int d) {
    extern __shared__ uint16_t s[];
    unsigned base = aligned_base(s);
    uint16_t* img = s + (base - (unsigned)__cvta_generic_to_shared(s)) / 2;
    for (int i = threadIdx.x; i < IMG / 2; i += blockDim.x) img[i] = (uint16_t)i;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
        int c0 = 0, r0 = 0;
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                     :: "l"(&tm), "r"(c0), "r"(r0), "r"(base + d) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
}

static uint32_t swz(uint32_t a, int span) {
    uint32_t b = span == 128 ? 7 : span == 64 ? 3 : span == 32 ? 1 : 0;
    return a ^ (((a >> 7) & b) << 4);
}
// smem byte offset (from the 1024-aligned base) of element (q, c) of a box starting at `start`.
static uint32_t soff(int model, uint32_t start, int q, int c, int span) {
    uint32_t in = q * span + 2 * c;
    return model == 0 ? swz(start + in, span) : start + swz(in, span);
}

static CUtensorMap encode(uint16_t* g, int span) {
    CUtensorMap tm;
    cuuint64_t dims[2] = {W, H};
    cuuint64_t strides[1] = {W * 2};
    cuuint32_t box[2] = {(cuuint32_t)(span / 2), ROWS};
    cuuint32_t es[2] = {1, 1};
    CUtensorMapSwizzle sw = span == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : span == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_32B;
    CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, g, dims, strides, box, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r) { printf("encode span %d: error %d\n", span, (int)r); exit(1); }
    return tm;
}

// One load config; returns a bitmask of matching models (1 = abs, 2 = rel), -1 on error.
static int run_load(uint16_t* g, uint16_t* o, int span, int d, bool verbose) {
    CUtensorMap tm = encode(g, span);
    std::vector<uint16_t> hg(W * H), ho(NCTA * IMG / 2);
    for (int i = 0; i < W * H; i++) hg[i] = (uint16_t)((i / W) * 64 + i % W);
    CK(cudaMemcpy(g, hg.data(), W * H * 2, cudaMemcpyHostToDevice));
    k_load<<<NCTA, 128, SMEM>>>(tm, span, d, o);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("swizzle %3d D %#06x: load launch failed: %s\n", span, d, cudaGetErrorString(e)); return -1; }
    CK(cudaMemcpy(ho.data(), o, NCTA * IMG, cudaMemcpyDeviceToHost));
    int res = 0;
    for (int m = 0; m < 2; m++) {
        std::vector<uint16_t> want(IMG / 2, 0xffff);
        for (int r = 0; r < NCTA; r++)
            for (int q = 0; q < ROWS; q++)
                for (int c = 0; c < span / 2; c++)
                    want[soff(m, d + r * ROWS * span, q, c, span) / 2] = (uint16_t)((r * ROWS + q) * 64 + c);
        bool ok = true;
        for (int cta = 0; cta < NCTA; cta++)
            for (int i = 0; i < IMG / 2; i++)
                if (ho[cta * IMG / 2 + i] != want[i]) {
                    if (verbose && ok && m == 0)
                        printf("  abs first diff: cta %d byte %#x got %#06x want %#06x\n", cta, 2 * i, ho[cta * IMG / 2 + i], want[i]);
                    ok = false;
                }
        if (ok) res |= 1 << m;
    }
    return res;
}

static int run_store(uint16_t* g, int span, int d) {
    CUtensorMap tm = encode(g, span);
    std::vector<uint16_t> hg(W * H);
    CK(cudaMemset(g, 0, W * H * 2));
    k_store<<<1, 128, SMEM>>>(tm, d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("swizzle %3d D %#06x: store launch failed: %s\n", span, d, cudaGetErrorString(e)); return -1; }
    CK(cudaMemcpy(hg.data(), g, W * H * 2, cudaMemcpyDeviceToHost));
    int res = 0;
    for (int m = 0; m < 2; m++) {
        std::vector<uint16_t> want(W * H, 0);
        for (int q = 0; q < ROWS; q++)
            for (int c = 0; c < span / 2; c++) want[q * W + c] = (uint16_t)(soff(m, d, q, c, span) / 2);
        if (want == hg) res |= 1 << m;
    }
    return res;
}

static const char* name(int r) {
    return r < 0 ? "FAULT" : r == 3 ? "abs+rel" : r == 1 ? "abs" : r == 2 ? "rel" : "none";
}

int main(int argc, char** argv) {
    uint16_t* g;
    uint16_t* o;
    CK(cudaMalloc(&g, W * H * 2));
    CK(cudaMalloc(&o, NCTA * IMG));
    CK(cudaFuncSetAttribute(k_load, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    CK(cudaFuncSetAttribute(k_store, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    if (argc == 3) {
        int span = atoi(argv[1]), d = (int)strtol(argv[2], 0, 0);
        int r = run_load(g, o, span, d, true);
        printf("swizzle %3d D %#06x: load %s\n", span, d, name(r));
        return 0;
    }
    int fails = 0, n = 0;
    for (int span = 32; span <= 128; span *= 2)
        for (int d = 0; d < 0x800; d += 0x80) {
            int l = run_load(g, o, span, d, true);
            int s = run_store(g, span, d);
            printf("swizzle %3d D %#06x: multicast load %-7s | store %-7s\n", span, d, name(l), name(s));
            n++;
            if (l != 1 && l != 3) fails++;
            if (s != 1 && s != 3) fails++;
        }
    printf("%d configs: %s\n", n, fails ? "abs model FAILS somewhere" : "abs model matches every config");
    return 0;
}
