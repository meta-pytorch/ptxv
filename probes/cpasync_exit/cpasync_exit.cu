/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// cp.async (LDGSTS) still in flight when the issuing CTA exits: can it land in shared
// memory after the SM hands that shared memory to the next CTA?
// mode 0 (test): each CTA fills its smem with its own pattern, spins, checks it, then every
//   thread issues K 16-byte cp.async.cg from a large cold gmem buffer filled with MARK into
//   its smem slots and exits without cp.async.wait_group. A later CTA on the same SM that
//   sees MARK in its own (pattern-filled, never cp.async-targeted) smem caught a late write.
// mode 1 (positive control): same, but after issuing the copies the CTA itself overwrites
//   the targeted slots with its pattern without waiting, spins, and checks: MARK there means
//   the copies landed after the plain stores (the detector can see late cp.async writes).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#define MARK 0xDEADBEEFu
__device__ __forceinline__ void spin(long long cyc) { long long t = clock64(); while (clock64() - t < cyc) {} }
__global__ void k(const uint4* __restrict__ src, size_t src_n, unsigned long long* cnt, int mode, int K, long long cyc, unsigned seed) {
  extern __shared__ uint32_t sm[];
  const int words = blockDim.x * K * 4;               // smem size in words
  const uint32_t pat = 0x5A000000u | (blockIdx.x & 0xFFFFFF);
  for (int i = threadIdx.x; i < words; i += blockDim.x) sm[i] = pat ^ i;
  __syncthreads();
  spin(cyc);
  __syncthreads();
  unsigned long long bad = 0, mark = 0;
  for (int i = threadIdx.x; i < words; i += blockDim.x) { uint32_t v = sm[i]; if (v != (pat ^ i)) { bad++; if (v == MARK) mark++; } }
  __syncthreads();
  // issue K cold copies per thread into this thread's slots
  unsigned h = seed ^ (blockIdx.x * 2654435761u) ^ (threadIdx.x * 40503u);
  for (int j = 0; j < K; j++) {
    h = h * 1664525u + 1013904223u;
    size_t si = ((size_t)h * 131u + (size_t)j * 9973u * blockDim.x) % src_n;
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(&sm[(j * blockDim.x + threadIdx.x) * 4]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src + si) : "memory");
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  if (mode == 1) {
    for (int j = 0; j < K; j++) for (int w = 0; w < 4; w++) { int i = (j * blockDim.x + threadIdx.x) * 4 + w; sm[i] = pat ^ i; }
    spin(cyc);
    unsigned long long cb = 0;
    for (int j = 0; j < K; j++) for (int w = 0; w < 4; w++) { int i = (j * blockDim.x + threadIdx.x) * 4 + w; if (sm[i] == MARK) cb++; }
    atomicAdd(&cnt[2], cb);
    asm volatile("cp.async.wait_all;\n" ::: "memory");
  }
  if (bad) atomicAdd(&cnt[0], bad);
  if (mark) atomicAdd(&cnt[1], mark);
  // mode 0: exit with the copies in flight
}
int main(int argc, char** argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 0, runs = argc > 2 ? atoi(argv[2]) : 50;
  int K = argc > 3 ? atoi(argv[3]) : 24; long long cyc = argc > 4 ? atoll(argv[4]) : 20000;
  int threads = 256; size_t smem = (size_t)threads * K * 16;
  int dev; cudaGetDevice(&dev); cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int per_sm = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, threads, smem);
  int grid = p.multiProcessorCount * per_sm * 32;    // 32 waves
  size_t src_n = (size_t)1 << 26;                     // 64M x 16 B = 1 GiB, cold
  uint4* src; cudaMalloc(&src, src_n * 16); cudaMemset(src, 0xEF, src_n * 16);
  { uint32_t* h = (uint32_t*)malloc(16); for (int i = 0; i < 4; i++) h[i] = MARK; // fill MARK
    // memset with a 4-byte pattern via a kernel-free path: copy one uint4 then double
    cudaMemcpy(src, h, 16, cudaMemcpyHostToDevice);
    for (size_t n = 1; n < src_n; n *= 2) cudaMemcpy(src + n, src, (n < src_n - n ? n : src_n - n) * 16, cudaMemcpyDeviceToDevice);
    free(h); }
  unsigned long long* cnt; cudaMalloc(&cnt, 3 * 8);
  unsigned long long tot[3] = {0, 0, 0}; int runs_bad = 0;
  for (int r = 0; r < runs; r++) {
    cudaMemset(cnt, 0, 24);
    k<<<grid, threads, smem>>>(src, src_n, cnt, mode, K, cyc, 0x9e3779b9u * (r + 1));
    cudaError_t e = cudaDeviceSynchronize(); if (e) { printf("error %s\n", cudaGetErrorString(e)); return 1; }
    unsigned long long c[3]; cudaMemcpy(c, cnt, 24, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; i++) tot[i] += c[i]; if (c[1] || c[2]) runs_bad++;
  }
  printf("%s mode=%d runs=%d grid=%d (%d SMs x %d CTAs/SM x 32 waves) threads=%d K=%d smem=%zu spin=%lld: "
         "smem words not own pattern %llu, of them MARK %llu; control MARK after own stores %llu; runs with MARK %d\n",
         p.name, mode, runs, grid, p.multiProcessorCount, per_sm, threads, K, smem, cyc, tot[0], tot[1], tot[2], runs_bad);
  return 0;
}
