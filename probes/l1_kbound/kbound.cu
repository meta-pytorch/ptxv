/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Does a GB300 SM's L1 keep lines across a kernel boundary?
//
// Build: nvcc -O3 -arch=sm_103a -std=c++17 -o kbound kbound.cu
// Run:   ./kbound mode rd [rdev=0 wdev=1 iters=N grid=G lines=L]   (refuses GPUs in $PTXV_EXCLUDE_GPUS)
//
// Every iteration k (reader GPU rdev, writer GPU wdev; wdev == rdev: writer on another SM of
// the same GPU):
//   K1 (reader stream): every CTA weak-loads the same L payload lines (value k-1), stores the
//      sum, then signals "loaded" with a relaxed atomic that depends on the loaded values.
//   writer (persistent kernel on wdev): waits for all K1 CTAs, writes payload = k (st.global,
//      over NVLink when wdev != rdev), fence.acq_rel.sys, st.release.sys flag = k.
//   K2: lane 0 of warp 0 polls the flag with ld.volatile (no fence), __syncthreads, then every
//      thread weak-reads the payload. Any word < k is stale.
// mode 0  K1 and K2 are separate kernels on one stream (the kernel boundary under test)
// mode 1  one kernel does K1's loads, signals, polls, reads (positive control: no boundary)
// mode 2  as mode 0, plus fence.acq_rel.sys after the poll (negative control)
// mode 3  K2 on a second stream that waits on an event recorded after K1 (cross-stream)
// mode 4  as mode 0 with an empty kernel between K1 and K2
// rd 0 generic ld (LD.E), 1 ld.global (LDG.E), 2 ld.global.nc (LDG.E.CONSTANT)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>

// True if GPU d is listed in $PTXV_EXCLUDE_GPUS (comma-separated indices).
static bool excluded_gpu(int d) {
  const char* e = getenv("PTXV_EXCLUDE_GPUS");
  if (!e) return false;
  std::string s(e);
  size_t i = 0;
  while (i <= s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    if (j > i && atoi(s.substr(i, j - i).c_str()) == d) return true;
    i = j + 1;
  }
  return false;
}
typedef uint32_t u32;
typedef unsigned long long u64;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2);} } while (0)

__device__ __forceinline__ u32 ld_volatile(const u32* a) {
  u32 v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ void st_release_sys(u32* a, u32 v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory"); }
template <int RD> __device__ __forceinline__ u32 rd(const u32* a) {
  u32 v;
  if constexpr (RD == 0) asm volatile("ld.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  else if constexpr (RD == 1) asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  else asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ u32 smid() { u32 s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }

struct P { u32 *pay, *flag, *cnt, *sink; u64* ctr; u32 words, k, grid; };
enum { C_READS = 0, C_STALE, C_WAITED, C_N };

template <int RD> __device__ void load_phase(const P& p) {
  u32 acc = 0;
  for (u32 i = threadIdx.x; i < p.words; i += blockDim.x) acc += rd<RD>(p.pay + i);
  p.sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;  // waits for the loads
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd_system(p.cnt, 1u + (acc == 0x9e3779b9u ? 1u : 0u));
}
template <int RD, bool FENCE> __device__ void read_phase(const P& p) {
  __shared__ u32 waited;
  if (threadIdx.x == 0) {
    u32 w = 0;
    while (ld_volatile(p.flag) < p.k) w = 1;
    waited = w;
  }
  __syncthreads();
  if constexpr (FENCE) fence_sys();
  u32 stale = 0, n = 0;
  for (u32 i = threadIdx.x; i < p.words; i += blockDim.x) { stale += rd<RD>(p.pay + i) < p.k; n++; }
  atomicAdd(p.ctr + C_READS, (u64)n);
  if (stale) atomicAdd(p.ctr + C_STALE, (u64)stale);
  if (threadIdx.x == 0 && waited) atomicAdd(p.ctr + C_WAITED, 1ull);
}
template <int RD> __global__ void __launch_bounds__(256, 1) k1(P p) { load_phase<RD>(p); }
template <int RD, bool F> __global__ void __launch_bounds__(256, 1) k2(P p) { read_phase<RD, F>(p); }
template <int RD> __global__ void __launch_bounds__(256, 1) k12(P p) { load_phase<RD>(p); read_phase<RD, false>(p); }
__global__ void kempty() {}

// persistent writer: one warp
__global__ void writer(u32* pay, u32* flag, u32* cnt, u32 words, u32 iters, u32 grid, u32* abort_) {
  for (u32 k = 1; k <= iters; k++) {
    if (threadIdx.x == 0) { while (ld_volatile(cnt) < k * grid) { if (ld_volatile(abort_)) break; } }
    __syncwarp();
    for (u32 i = threadIdx.x; i < words; i += 32) pay[i] = k;
    __syncwarp();
    fence_sys();
    if (threadIdx.x == 0) st_release_sys(flag, k);
  }
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: kbound mode rd [key=val..]\n"); return 1; }
  int mode = atoi(argv[1]), rdf = atoi(argv[2]);
  int rdev = 0, wdev = 1; u32 iters = 20000, grid = 0, lines = 64;
  for (int i = 3; i < argc; i++) {
    std::string a = argv[i]; auto eq = a.find('='); std::string k = a.substr(0, eq); long v = atol(a.c_str() + eq + 1);
    if (k == "rdev") rdev = v; else if (k == "wdev") wdev = v; else if (k == "iters") iters = v;
    else if (k == "grid") grid = v; else if (k == "lines") lines = v;
    else { fprintf(stderr, "bad key %s\n", k.c_str()); return 1; }
  }
  if (excluded_gpu(rdev) || excluded_gpu(wdev)) { fprintf(stderr, "GPU in PTXV_EXCLUDE_GPUS\n"); return 1; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, rdev));
  if (!grid) grid = prop.multiProcessorCount - (rdev == wdev ? 4 : 0);
  u32 words = lines * 32;
  CK(cudaSetDevice(rdev));
  if (rdev != wdev) { cudaDeviceEnablePeerAccess(wdev, 0); cudaGetLastError(); }
  u32 *pay, *flag, *cnt, *sink, *abort_; u64* ctr;
  CK(cudaMalloc(&pay, words * 4)); CK(cudaMalloc(&flag, 128)); CK(cudaMalloc(&cnt, 128));
  CK(cudaMalloc(&sink, grid * 256 * 4)); CK(cudaMalloc(&ctr, C_N * 8)); CK(cudaMalloc(&abort_, 128));
  CK(cudaMemset(pay, 0, words * 4)); CK(cudaMemset(flag, 0, 128)); CK(cudaMemset(cnt, 0, 128));
  CK(cudaMemset(ctr, 0, C_N * 8)); CK(cudaMemset(abort_, 0, 128));
  cudaStream_t s1, s2; CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  cudaEvent_t ev; CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  // writer
  cudaStream_t sw;
  CK(cudaSetDevice(wdev));
  if (rdev != wdev) { cudaDeviceEnablePeerAccess(rdev, 0); cudaGetLastError(); }
  CK(cudaStreamCreateWithFlags(&sw, cudaStreamNonBlocking));
  writer<<<1, 32, 0, sw>>>(pay, flag, cnt, words, iters, grid, abort_);
  CK(cudaGetLastError());
  CK(cudaSetDevice(rdev));
  P p{pay, flag, cnt, sink, ctr, words, 0, grid};
  for (u32 k = 1; k <= iters; k++) {
    p.k = k;
#define L1(RD) k1<RD><<<grid, 256, 0, s1>>>(p)
#define L2(RD, F, S) k2<RD, F><<<grid, 256, 0, S>>>(p)
#define L12(RD) k12<RD><<<grid, 256, 0, s1>>>(p)
    switch (mode * 10 + rdf) {
#define CASES(RD) \
      case 0 + RD: L1(RD); L2(RD, false, s1); break; \
      case 10 + RD: L12(RD); break; \
      case 20 + RD: L1(RD); L2(RD, true, s1); break; \
      case 30 + RD: L1(RD); CK(cudaEventRecord(ev, s1)); CK(cudaStreamWaitEvent(s2, ev, 0)); L2(RD, false, s2); \
                    CK(cudaEventRecord(ev, s2)); CK(cudaStreamWaitEvent(s1, ev, 0)); break; \
      case 40 + RD: L1(RD); kempty<<<1, 32, 0, s1>>>(); L2(RD, false, s1); break;
      CASES(0) CASES(1) CASES(2)
      default: fprintf(stderr, "bad mode/rd\n"); return 1;
    }
    if ((k & 255) == 0) CK(cudaStreamSynchronize(s1));
  }
  CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(wdev)); CK(cudaDeviceSynchronize()); CK(cudaSetDevice(rdev));
  u64 h[C_N]; CK(cudaMemcpy(h, ctr, sizeof h, cudaMemcpyDeviceToHost));
  printf("RESULT mode=%d rd=%d rdev=%d wdev=%d grid=%u lines=%u iters=%u reads=%llu stale=%llu waited_ctas=%llu (of %llu)\n",
         mode, rdf, rdev, wdev, grid, lines, iters, h[C_READS], h[C_STALE], h[C_WAITED], (u64)grid * iters);
  return 0;
}
