/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// S-TMAP / S-INIT probe: does a TMA load depend on bytes 0x40..0x7f of a 128-byte
// CUtensorMap in global memory? msl_ops dist_moe copies only the 64-byte template and
// publishes the rest of the map uninitialized. For each config the host encodes a map with
// cuTensorMapEncodeTiled, writes it to global memory with bytes 0x40..0x7f as encoded, zero,
// all-ones or random, and one CTA loads a box through it (fence.proxy.tensormap acquire, then
// cp.async.bulk.tensor.2d) and dumps the shared tile; the dumps are compared bitwise.
//   nvcc -O3 -gencode arch=compute_103a,code=sm_103a -o tmap_tail tmap_tail.cu -lcuda
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 16384

__global__ void load(const void* tmap, uint32_t* out, int x0, int y0, unsigned bytes) {
    __shared__ __align__(1024) unsigned char tile[TILE];
    __shared__ __align__(8) unsigned long long bar;
    unsigned ba = (unsigned)__cvta_generic_to_shared(&bar);
    unsigned ta = (unsigned)__cvta_generic_to_shared(tile);
    for (int i = threadIdx.x; i < TILE / 4; i += blockDim.x) ((uint32_t*)tile)[i] = 0xdeadbeef;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(ba));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(tmap) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(ba), "r"(bytes) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
            ::"r"(ta), "l"(tmap), "r"(x0), "r"(y0), "r"(ba) : "memory");
    }
    asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra W;\n}" ::"r"(ba) : "memory");
    for (int i = threadIdx.x; i < TILE / 4; i += blockDim.x) out[i] = ((uint32_t*)tile)[i];
}

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(r_, &s); printf("%s: %s\n", #x, s); exit(1); } } while (0)

int main() {
    const int W = 200, H = 70;  // 4-byte elements; boxes reach past both edges (OOB fill)
    std::vector<uint32_t> src(W * H);
    for (int i = 0; i < W * H; i++) src[i] = 0x10000000u + i * 2654435761u;
    void* dsrc;
    cudaMalloc(&dsrc, src.size() * 4);
    cudaMemcpy(dsrc, src.data(), src.size() * 4, cudaMemcpyHostToDevice);
    void* dmap;
    cudaMalloc(&dmap, 128);
    uint32_t* dout;
    cudaMalloc(&dout, TILE);
    struct Cfg { CUtensorMapDataType dt; int es; unsigned bx, by; CUtensorMapSwizzle sw; CUtensorMapFloatOOBfill fill; int x0, y0; };
    std::vector<Cfg> cfgs;
    CUtensorMapSwizzle sws[] = {CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_SWIZZLE_128B};
    unsigned spans[] = {256, 32, 64, 128};
    for (int s = 0; s < 4; s++)
        for (int fill = 0; fill < 2; fill++)
            for (int pos = 0; pos < 3; pos++) {
                unsigned bx = spans[s] / 4 > 64 ? 64 : spans[s] / 4;  // inner box bytes <= span
                if (s == 0) bx = 32;
                int x0 = pos == 0 ? 0 : pos == 1 ? 40 : W - bx / 2;
                int y0 = pos == 2 ? H - 4 : 3;
                cfgs.push_back({fill ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, bx, 16, sws[s],
                                fill ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE, x0, y0});
            }
    const char* names[] = {"encoded", "zero", "ones", "random"};
    unsigned long long total = 0, diff[4] = {0, 0, 0, 0};
    int tail_nonzero = 0;
    uint64_t rs = 0x9e3779b97f4a7c15ull;
    for (size_t c = 0; c < cfgs.size(); c++) {
        Cfg& g = cfgs[c];
        CUtensorMap m;
        cuuint64_t dims[2] = {(cuuint64_t)W, (cuuint64_t)H};
        cuuint64_t strides[1] = {(cuuint64_t)W * 4};
        cuuint32_t box[2] = {g.bx, g.by};
        cuuint32_t es[2] = {1, 1};
        CK(cuTensorMapEncodeTiled(&m, g.dt, 2, dsrc, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, g.sw,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE, g.fill));
        unsigned char base[128];
        memcpy(base, &m, 128);
        for (int i = 64; i < 128; i++) tail_nonzero += base[i] != 0;
        std::vector<uint32_t> ref;
        for (int v = 0; v < 4; v++) {
            for (int rep = 0; rep < 3; rep++) {
                unsigned char b[128];
                memcpy(b, base, 128);
                for (int i = 64; i < 128; i++) {
                    if (v == 1) b[i] = 0;
                    if (v == 2) b[i] = 0xff;
                    if (v == 3) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; b[i] = (unsigned char)rs; }
                }
                cudaMemcpy(dmap, b, 128, cudaMemcpyHostToDevice);
                load<<<1, 128>>>(dmap, dout, g.x0, g.y0, g.bx * g.by * g.es);
                cudaError_t e = cudaDeviceSynchronize();
                if (e != cudaSuccess) { printf("cfg %zu variant %s: %s\n", c, names[v], cudaGetErrorString(e)); return 1; }
                std::vector<uint32_t> o(TILE / 4);
                cudaMemcpy(o.data(), dout, TILE, cudaMemcpyDeviceToHost);
                if (v == 0 && rep == 0) ref = o;
                else if (o != ref) diff[v]++;
                total++;
            }
        }
        // Sanity: the reference tile holds loaded data (not the 0xdeadbeef fill everywhere).
        int loaded = 0;
        for (auto w : ref) loaded += w != 0xdeadbeef;
        printf("cfg %2zu swizzle %d fill %d box %ux%u at (%d,%d): %d words loaded\n", c, (int)g.sw, (int)g.fill, g.bx, g.by, g.x0, g.y0, loaded);
    }
    printf("encoded maps: %d nonzero bytes in 0x40..0x7f over %zu maps\n", tail_nonzero, cfgs.size());
    for (int v = 0; v < 4; v++) printf("%-8s tail: %llu of %zu loads differ from the encoded map's\n", names[v], diff[v], cfgs.size() * 3 - (v == 0 ? cfgs.size() : 0));
    printf("%llu loads\n", total);
    return 0;
}
