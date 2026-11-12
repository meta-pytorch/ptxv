/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// E-CPASYNC-PROXY discriminating probe: are cp.async writes to shared memory visible to an
// async-proxy read (TMA store / bulk copy from shared memory) without fence.proxy.async after
// the copies completed?
//
// Per round k = 1..R, every thread of a CTA writes its 16-byte chunks of one shared buffer
// (W words) with round-k values, the variant's wait and fence run, and thread 0 stores the
// buffer to the round's global slot with an async-proxy read, waits for its source reads
// (cp.async.bulk.wait_group.read 0), and a barrier frees the buffer for the next round. The
// buffer starts as a poison pattern published with fence.proxy.async, so a stale read returns
// round k-1's values or poison. A separate kernel counts words != val(k, i).
//
// writer: 0 CPW  cp.async.cg (LDGSTS) + cp.async.commit_group + cp.async.wait_group 0, bar.sync
//         1 CPM  cp.async.cg + cp.async.mbarrier.arrive.noinc; every thread waits for the
//                mbarrier phase (all copies of the CTA landed); thread 0 reads at once
//         2 ST   st.shared.v4 (the writer class of rung 1b B04), bar.sync
// fence:  0 none; 1 before the wait (CPW: after commit_group, CPM: after the arrive);
//         2 after the wait (CPW: after wait_group, CPM: after the phase wait, ST: after the
//         stores). fence = fence.proxy.async.shared::cta, executed by every thread.
// reader: 0 bulk copy cp.async.bulk.global.shared::cta (UBLKCP.G.S);
//         1 tensor store cp.async.bulk.tensor.2d.global.shared::cta (UTMASTG), 64-word rows.
// delay:  0, or a per-(round, warp, CTA) pseudo-random spin of up to `delay` cycles before
//         the writes (the last warp writes right before the others' read).
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o cpx cpx.cu -lcuda
// Run:   ./cpx writer fence reader rounds W ctas_per_sm delay launches
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

__host__ __device__ __forceinline__ unsigned val(unsigned k, unsigned i) {
    return (k * 0x9E3779B1u) ^ (i * 0x85EBCA77u) ^ 0x5bd1e995u;
}

__device__ __forceinline__ void fence_proxy() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__global__ void fill_src(unsigned* src, int rounds, int W) {
    size_t n = (size_t)(rounds + 1) * W;
    for (size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x; j < n; j += (size_t)gridDim.x * blockDim.x)
        src[j] = val((unsigned)(j / W), (unsigned)(j % W));
}

__global__ void __launch_bounds__(256) cpx(int writer, int fence, int reader, int rounds, int W, int delay,
                                           const unsigned* __restrict__ src, unsigned* dst,
                                           const __grid_constant__ CUtensorMap tmap) {
    extern __shared__ __align__(1024) unsigned char smem[];
    unsigned* buf = (unsigned*)(smem + 1024);
    const unsigned sbuf = (unsigned)__cvta_generic_to_shared(buf);
    const unsigned smb = (unsigned)__cvta_generic_to_shared(smem);
    const int tid = threadIdx.x, nt = blockDim.x, chunks = W / 4;
    for (int i = tid; i < W; i += nt) buf[i] = val(0, i);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smb), "r"(nt) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    fence_proxy();
    __syncthreads();
    for (int k = 1; k <= rounds; k++) {
        if (delay) {
            unsigned h = val(k, (tid >> 5) * 131 + blockIdx.x * 7919);
            long long d = h % (unsigned)delay, t0 = clock64();
            while (clock64() - t0 < d) {
            }
        }
        if (writer == 2) {
            for (int c = tid; c < chunks; c += nt) {
                unsigned i = 4 * c;
                asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(sbuf + 16 * c), "r"(val(k, i)),
                             "r"(val(k, i + 1)), "r"(val(k, i + 2)), "r"(val(k, i + 3))
                             : "memory");
            }
            if (fence) fence_proxy();
            __syncthreads();
        } else {
            const unsigned* s = src + (size_t)k * W;
            for (int c = tid; c < chunks; c += nt)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sbuf + 16 * c), "l"(s + 4 * c)
                             : "memory");
            if (writer == 0) {
                asm volatile("cp.async.commit_group;" ::: "memory");
                if (fence == 1) fence_proxy();
                asm volatile("cp.async.wait_group 0;" ::: "memory");
                if (fence == 2) fence_proxy();
                __syncthreads();
            } else {
                asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smb) : "memory");
                if (fence == 1) fence_proxy();
                unsigned ok = 0;
                while (!ok)
                    asm volatile(
                        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                        : "=r"(ok)
                        : "r"(smb), "r"((unsigned)((k - 1) & 1))
                        : "memory");
                if (fence == 2) fence_proxy();
            }
        }
        if (tid == 0) {
            size_t slot = (size_t)blockIdx.x * rounds + (k - 1);
            if (reader == 0) {
                asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst + slot * W),
                             "r"(sbuf), "r"(W * 4)
                             : "memory");
            } else {
                int row = (int)(slot * (W / 64));
                asm volatile(
                    "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(&tmap),
                    "r"(0), "r"(row), "r"(sbuf)
                    : "memory");
            }
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        }
        __syncthreads();
    }
    if (tid == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// Per launch: bad words, bad (CTA, round) tiles, tiles holding the previous round's / poison
// values in some word.
__global__ void verify(const unsigned* dst, int rounds, int W, size_t tiles, unsigned long long* out) {
    for (size_t t = blockIdx.x; t < tiles; t += gridDim.x) {
        unsigned k = (unsigned)(t % rounds) + 1;
        unsigned nb = 0, prev = 0, poison = 0;
        for (int i = threadIdx.x; i < W; i += blockDim.x) {
            unsigned x = dst[t * W + i];
            if (x != val(k, i)) {
                nb++;
                prev += x == val(k - 1, i) && k > 1;
                poison += x == val(0, i);
            }
        }
        for (int o = 16; o; o >>= 1) {
            nb += __shfl_xor_sync(0xffffffffu, nb, o);
            prev += __shfl_xor_sync(0xffffffffu, prev, o);
            poison += __shfl_xor_sync(0xffffffffu, poison, o);
        }
        __shared__ unsigned s_nb[32], s_prev[32], s_poison[32];
        if ((threadIdx.x & 31) == 0) {
            s_nb[threadIdx.x >> 5] = nb;
            s_prev[threadIdx.x >> 5] = prev;
            s_poison[threadIdx.x >> 5] = poison;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned a = 0, b = 0, c = 0;
            for (int w = 0; w < (int)(blockDim.x >> 5); w++) {
                a += s_nb[w];
                b += s_prev[w];
                c += s_poison[w];
            }
            if (a) {
                atomicAdd(&out[0], a);
                atomicAdd(&out[1], 1ull);
                if (b) atomicAdd(&out[2], 1ull);
                if (c) atomicAdd(&out[3], 1ull);
            }
        }
        __syncthreads();
    }
}

#define CK(x)                                                                    \
    do {                                                                         \
        cudaError_t e_ = (x);                                                    \
        if (e_ != cudaSuccess) {                                                 \
            printf("error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 9) {
        printf("usage: cpx writer fence reader rounds W ctas_per_sm delay launches\n");
        return 2;
    }
    int writer = atoi(argv[1]), fence = atoi(argv[2]), reader = atoi(argv[3]), rounds = atoi(argv[4]);
    int W = atoi(argv[5]), per_sm = atoi(argv[6]), delay = atoi(argv[7]), launches = atoi(argv[8]);
    if (W < 1024 || W > 16384 || W % 1024 || writer < 0 || writer > 2 || fence < 0 || fence > 2 ||
        (writer == 2 && fence == 1))
        return 2;
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int grid = sms * per_sm, nt = 256;
    size_t smem = 1024 + (size_t)W * 4;
    CK(cudaFuncSetAttribute(cpx, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, cpx, nt, smem));
    size_t tiles = (size_t)grid * rounds;
    unsigned *src, *dst;
    unsigned long long* cnt;
    CK(cudaMalloc(&src, (size_t)(rounds + 1) * W * 4));
    CK(cudaMalloc(&dst, tiles * W * 4));
    CK(cudaMalloc(&cnt, 4 * 8));
    fill_src<<<1024, 256>>>(src, rounds, W);
    CK(cudaDeviceSynchronize());
    CUtensorMap tm;
    memset(&tm, 0, sizeof tm);
    cuuint64_t dims[2] = {64, (cuuint64_t)tiles * (W / 64)};
    cuuint64_t strides[1] = {64 * 4};
    cuuint32_t box[2] = {64, (cuuint32_t)(W / 64)};
    cuuint32_t es[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, dst, dims, strides, box, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        printf("cuTensorMapEncodeTiled failed: %d\n", (int)r);
        return 1;
    }
    unsigned long long fail_launches = 0, bad_words = 0, bad_tiles = 0, prev_tiles = 0, poison_tiles = 0;
    for (int it = 0; it < launches; it++) {
        CK(cudaMemsetAsync(dst, 0xff, tiles * W * 4));
        CK(cudaMemsetAsync(cnt, 0, 4 * 8));
        cpx<<<grid, nt, smem>>>(writer, fence, reader, rounds, W, delay, src, dst, tm);
        verify<<<1024, 256>>>(dst, rounds, W, tiles, cnt);
        unsigned long long h[4];
        CK(cudaMemcpy(h, cnt, sizeof h, cudaMemcpyDeviceToHost));
        fail_launches += h[1] > 0;
        bad_words += h[0];
        bad_tiles += h[1];
        prev_tiles += h[2];
        poison_tiles += h[3];
    }
    printf("RESULT writer=%d fence=%d reader=%d rounds=%d W=%d ctas_per_sm=%d occupancy=%d sms=%d grid=%d delay=%d "
           "launches=%d failing_launches=%llu tiles=%llu bad_tiles=%llu prev_tiles=%llu poison_tiles=%llu "
           "bad_words=%llu\n",
           writer, fence, reader, rounds, W, per_sm, occ, sms, grid, delay, launches, fail_launches,
           (unsigned long long)tiles * launches, bad_tiles, prev_tiles, poison_tiles, bad_words);
    return 0;
}
