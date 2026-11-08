/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Message passing across NVLink with a *weak* flag store (index L5: msl DeepEP
// nvlink_barrier_wo_local_sync signals peers with a plain `ST.E` after `MEMBAR.SC.SYS`; the
// peer polls with `LD.E.STRONG.SYS` and ends with `MEMBAR.SC.SYS; CCTL.IVALL`). PTX: the weak
// store and the strong poll do not synchronize (data race). GPU 0 (writer) and GPU 1
// (reader), S slots, one warp per slot on each side:
//   writer: 32 lanes store 8 words each of the slot's payload into GPU 1's memory (weak
//           st.global), lane 0 then [fence] and stores the flag (weak or relaxed.sys) into
//           GPU 1's memory
//   reader: lane 0 polls the flag (ld.volatile, as DeepEP), `fence.sc.sys` (MEMBAR.SC.SYS +
//           CCTL.IVALL), __syncwarp, every lane reads its 8 payload words (weak ld.global)
//           and counts words != the run tag
//   var 0: fence.sc.sys + weak flag store   (DeepEP's shape)
//   var 1: fence.sc.sys + st.relaxed.sys flag (the PTX-legal form)
//   var 2: no fence + weak flag store        (positive control: payload may land after flag)
//   var 3: var 0 with the reader's fence removed (positive control for L1: stale lines)
// The reader also warms its L1 with the payload lines before the run (weak loads of the old
// values), so a missing invalidation shows.
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o weak_flag weak_flag.cu
// Run:   CUDA_VISIBLE_DEVICES=<a>,<b> ./weak_flag runs var
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
constexpr int S = 4096, W = 256;  // slots, payload words per slot

__global__ void writer(unsigned* pay, unsigned* flag, unsigned tag, int var) {
    int slot = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (slot >= S) return;
    unsigned* p = pay + (size_t)slot * W;
    for (int i = 0; i < W / 32; i++) asm volatile("st.global.u32 [%0], %1;" :: "l"(p + i * 32 + lane), "r"(tag) : "memory");
    __syncwarp();
    if (lane == 0) {
        if (var != 2) asm volatile("fence.sc.sys;" ::: "memory");
        if (var == 1) asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(flag + slot), "r"(tag) : "memory");
        else asm volatile("st.global.u32 [%0], %1;" :: "l"(flag + slot), "r"(tag) : "memory");
    }
}

__global__ void reader(const unsigned* pay, const unsigned* flag, unsigned tag, int var, unsigned* bad, unsigned zero) {
    int slot = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (slot >= S) return;
    const unsigned* p = pay + (size_t)slot * W;
    unsigned warm = 0;
    for (int i = 0; i < W / 32; i++) {  // fill L1 with the old payload lines
        unsigned x;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(p + i * 32 + lane) : "memory");
        warm += x;
    }
    if (lane == 0) {
        unsigned f = 0;
        long n = 0;
        while (f != tag && n < (1L << 26)) {
            asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(f) : "l"(flag + slot + (zero & n)) : "memory");
            n++;
        }
        if (f != tag) atomicAdd(bad + 1, 1u);
        if (var != 3) asm volatile("fence.sc.sys;" ::: "memory");
    }
    __syncwarp();
    unsigned nb = 0;
    for (int i = 0; i < W / 32; i++) {
        unsigned x;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(p + i * 32 + lane + (zero & warm)) : "memory");
        nb += x != tag;
    }
    if (nb) atomicAdd(bad, nb);
}

int main(int argc, char** argv) {
    int runs = atoi(argv[1]), var = atoi(argv[2]);
    CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0));
    CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0, 0));
    unsigned *pay, *flag, *bad;
    CK(cudaMalloc(&pay, (size_t)S * W * 4)); CK(cudaMalloc(&flag, S * 4)); CK(cudaMalloc(&bad, 8));
    cudaStream_t s0, s1;
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CK(cudaSetDevice(0)); CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    unsigned long tot = 0, runs_bad = 0, to = 0;
    for (int r = 0; r < runs; r++) {
        unsigned tag = 0x6b000000u | (unsigned)(r * 4 + var + 1);
        CK(cudaSetDevice(1));
        CK(cudaMemset(pay, 0, (size_t)S * W * 4)); CK(cudaMemset(flag, 0, S * 4)); CK(cudaMemset(bad, 0, 8));
        CK(cudaDeviceSynchronize());
        reader<<<S / 16, 512, 0, s1>>>(pay, flag, tag, var, bad, 0);
        CK(cudaSetDevice(0));
        writer<<<S / 16, 512, 0, s0>>>(pay, flag, tag, var);
        CK(cudaStreamSynchronize(s0));
        CK(cudaSetDevice(1));
        CK(cudaStreamSynchronize(s1));
        unsigned h[2];
        CK(cudaMemcpy(h, bad, 8, cudaMemcpyDeviceToHost));
        tot += h[0]; to += h[1]; runs_bad += h[0] > 0;
    }
    printf("var %d: runs %d x %d slots x %d words: stale words %lu in %lu runs, timeouts %lu\n", var, runs, S, W, tot, runs_bad, to);
    return 0;
}
