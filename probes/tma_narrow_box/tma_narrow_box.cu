/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Probe for S-TMA: the shared-memory layout of a swizzled TMA box whose inner dimension is
// narrower than the swizzle span (cuBLAS nvjet_sm103_tst_8x64_64x16_4x1_v_bz_TTN stores an
// 8 x 16 bf16 box, 16-byte rows, with SWIZZLE_64B).
// For every swizzle (none/32B/64B/128B) and inner box width (16/32/64/128 bytes, at most the
// span) with 16 rows of u16:
//   load:  global G[r][c] = r * 256 + c (64 x 256 u16), smem pre-filled with 0xffff, one TMA
//          load of the box at (c0, r0) = (8, 3); the smem image (4 KiB) is compared with
//          two models of where box element (r, c) lands, before the address swizzle:
//            dense:  r * inner_bytes + 2c
//            padded: r * max(span, inner_bytes) + 2c      (rows padded to the swizzle span)
//   store: smem[i] = i (u16 index), TMA store of the box to a zeroed G at (8, 3); global is
//          compared with both models (which smem u16 each in-box global element came from).
// Build: nvcc -arch=sm_103a -O2 -o tma_narrow_box tma_narrow_box.cu -lcuda
// Run:   CUDA_VISIBLE_DEVICES=3 ./tma_narrow_box
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int W = 256, H = 64, ROWS = 16, SMEM = 4096, C0 = 8, R0 = 3;

__global__ void k_load(const __grid_constant__ CUtensorMap tm, int bytes, uint16_t* out) {
    extern __shared__ __align__(1024) uint16_t s[];
    __shared__ __align__(8) uint64_t mbar;
    for (int i = threadIdx.x; i < SMEM / 2; i += blockDim.x) s[i] = 0xffff;
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    unsigned ma = (unsigned)__cvta_generic_to_shared(&mbar);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(ma));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(ma), "r"(bytes) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                     :: "r"(sa), "l"(&tm), "r"(C0), "r"(R0), "r"(ma) : "memory");
    }
    asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra W;\n}" :: "r"(ma) : "memory");
    for (int i = threadIdx.x; i < SMEM / 2; i += blockDim.x) out[i] = s[i];
}

__global__ void k_store(const __grid_constant__ CUtensorMap tm) {
    extern __shared__ __align__(1024) uint16_t s[];
    for (int i = threadIdx.x; i < SMEM / 2; i += blockDim.x) s[i] = (uint16_t)i;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned sa = (unsigned)__cvta_generic_to_shared(s);
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                     :: "l"(&tm), "r"(C0), "r"(R0), "r"(sa) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
}

static uint32_t swz(uint32_t a, int span) {
    uint32_t b = span == 128 ? 7 : span == 64 ? 3 : span == 32 ? 1 : 0;
    return a ^ (((a >> 7) & b) << 4);
}
// smem byte offset of box element (r, c) under a model.
static uint32_t soff(int model, int r, int c, int inner, int span) {
    int pitch = model == 0 ? inner : (span > inner ? span : inner);
    return swz(r * pitch + 2 * c, span);
}

int main() {
    uint16_t* g;
    uint16_t* o;
    CK(cudaMalloc(&g, W * H * 2));
    CK(cudaMalloc(&o, SMEM));
    std::vector<uint16_t> hg(W * H), ho(SMEM / 2);
    CK(cudaFuncSetAttribute(k_load, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    CK(cudaFuncSetAttribute(k_store, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    const int spans[4] = {0, 32, 64, 128};
    const CUtensorMapSwizzle sw[4] = {CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
                                      CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_SWIZZLE_128B};
    int fails = 0;
    for (int si = 0; si < 4; si++)
        for (int inner = 16; inner <= 128; inner *= 2) {
            int span = spans[si];
            if (span && inner > span) continue;
            CUtensorMap tm;
            cuuint64_t dims[2] = {W, H};
            cuuint64_t strides[1] = {W * 2};
            cuuint32_t box[2] = {(cuuint32_t)(inner / 2), ROWS};
            cuuint32_t es[2] = {1, 1};
            CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, g, dims, strides, box, es,
                                                CU_TENSOR_MAP_INTERLEAVE_NONE, sw[si], CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (r) { printf("encode span %d inner %d: error %d\n", span, inner, (int)r); fails++; continue; }
            // load
            for (int i = 0; i < W * H; i++) hg[i] = (uint16_t)((i / W) * 256 + i % W);
            CK(cudaMemcpy(g, hg.data(), W * H * 2, cudaMemcpyHostToDevice));
            k_load<<<1, 128, SMEM>>>(tm, inner * ROWS, o);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(ho.data(), o, SMEM, cudaMemcpyDeviceToHost));
            bool lm[2];
            for (int m = 0; m < 2; m++) {
                std::vector<uint16_t> want(SMEM / 2, 0xffff);
                for (int rr = 0; rr < ROWS; rr++)
                    for (int c = 0; c < inner / 2; c++)
                        want[soff(m, rr, c, inner, span) / 2] = (uint16_t)((R0 + rr) * 256 + C0 + c);
                lm[m] = want == ho;
            }
            // store
            CK(cudaMemset(g, 0, W * H * 2));
            k_store<<<1, 128, SMEM>>>(tm);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hg.data(), g, W * H * 2, cudaMemcpyDeviceToHost));
            bool sm[2];
            for (int m = 0; m < 2; m++) {
                std::vector<uint16_t> want(W * H, 0);
                for (int rr = 0; rr < ROWS; rr++)
                    for (int c = 0; c < inner / 2; c++)
                        want[(R0 + rr) * W + C0 + c] = (uint16_t)(soff(m, rr, c, inner, span) / 2);
                sm[m] = want == hg;
            }
            printf("swizzle %3d inner %3d B: load dense %s padded %s | store dense %s padded %s\n", span, inner,
                   lm[0] ? "MATCH" : "no", lm[1] ? "MATCH" : "no", sm[0] ? "MATCH" : "no", sm[1] ? "MATCH" : "no");
            if (!(lm[1] && sm[1])) fails++;
        }
    printf("%s\n", fails ? "padded model FAILS somewhere" : "padded model matches every config");
    return 0;
}
