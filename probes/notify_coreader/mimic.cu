/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// notify_dispatch's count handoff (DeepEP intranode, 2 ranks) re-implemented with its exact
// synchronization (barrier_block<true>, remote stores, barrier_block<false> with its entry
// fence.acq_rel.sys, weak reads, closing barrier_block<false>), plus a co-reader: extra warps of
// the same CTA (same SM, same L1) keep weak-loading a word of the same 128-byte line that
// nobody writes (race-free line sharing, docs/deepep-repro.md "Co-reader").
//
// Build: nvcc -O3 -arch=sm_103a -std=c++17 -o mimic mimic.cu
// Run:   ./mimic [iters=N] [readers=W] [wdelay=NS] [rdelay=NS]   (GPUs 0 and 1)
//   wdelay: rank 1 (only) waits this many ns after leaving barrier 1 before its remote store
//           (NOT DeepEP behaviour: a positive control that opens the window).
// Per iteration each rank reads the count its peer wrote; stale = value of an older iteration.
// It also records, per rank, globaltimer at its barrier-2 entry fence and at the peer store.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
typedef unsigned long long u64;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2);} } while (0)

__device__ __forceinline__ int ld_volatile(const int* a) { int v; asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
// ptxas hoists a weak load with a loop-invariant address out of a spin loop (seen in SASS, even
// from asm volatile), so the reader loops index with n & g_zmask (g_zmask = 0 at run time).
__device__ __forceinline__ int ld_weak(const int* a) { int v; asm volatile("ld.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
__device__ unsigned g_zmask;   // 0, set at run time: ptxas cannot fold n & g_zmask
__device__ __forceinline__ int ld_weak_n(const int* a, unsigned long long n) { return ld_weak(a + (n & g_zmask)); }
__device__ __forceinline__ void sync128() { asm volatile("bar.sync 1, 128;" ::: "memory"); }
__device__ __forceinline__ u64 gt() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TAG 1024

template <bool kSyncOnly> __device__ void barrier_block(int** sig, int rank) {
  int t = threadIdx.x;
  if constexpr (!kSyncOnly) { asm volatile("fence.acq_rel.sys;" ::: "memory"); sync128(); }
  if (t < 2) { atomicAdd_system(sig[rank] + t, TAG); atomicSub_system(sig[t] + rank, TAG); }
  while (true) {
    int v = t < 2 ? ld_volatile(sig[rank] + t) : 0;
    if (__all_sync(0xffffffff, v <= 0)) break;
  }
  sync128();
}

struct P { const int* cnt_src; int* buf[2]; int* sig[2]; int rank; int iters; int readers; unsigned wdelay; int nexp; int self; int rsync; unsigned rdelay; int noack; int preload; int rword[2]; u64* out; u64* tfence; u64* tstore; };

__global__ void notify_like(P p) {
  __shared__ volatile int done, go, ack;
  int t = threadIdx.x, warp = t / 32, rank = p.rank;
  int* local = p.buf[rank];
  if (blockIdx.x > 0) {   // same-kernel co-reader CTAs: stay only on CTA 0's SM
    unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    unsigned c0; while ((c0 = *(volatile unsigned*)(p.out + 3)) == 0) {}
    if (sm + 1 != c0 || t != 0) return;
    atomicAdd(p.out + 2, 1ull);
    unsigned long long n = 0; int acc = 0;
    while (!*(volatile unsigned long long*)(p.out + 1 + 0) && !*(volatile int*)(p.out + 6)) { n++; acc += ld_weak_n(local + p.rword[rank], n); }
    if (acc == 0x7fffffff) p.out[7] = 1;
    return;
  }
  if (t == 0) { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); *(volatile unsigned long long*)(p.out + 3) = sm + 1; }
  if (t == 0) { done = 0; go = 0; ack = 0; }
  __syncthreads();
  if (warp >= 4) {               // co-reader warps: weak loads of the stable word local[1 - rank]
    if ((t & 31) == 0) {
      int acc = 0;
      if (p.rsync) {   // one load per iteration, right after CTA 0's entry fence (handshake in smem)
        if (warp != 4) return;
        for (int k = 1; k <= p.iters; k++) {
          while (go != k) { if (p.preload) acc += ld_weak(local + p.rword[rank]); }
          acc += ld_weak(local + p.rword[rank]); __threadfence_block(); ack = k;
          if (p.rsync > 1) { u64 t0 = gt(); while (gt() - t0 < (u64)p.rsync) { acc += ld_weak(local + p.rword[rank]); if (p.rdelay) { u64 t1 = gt(); while (gt() - t1 < p.rdelay) {} } } }
        }
      } else {
        unsigned long long n = 0;
        while (!done) { n++; acc += ld_weak_n(local + p.rword[rank], n); if (p.rdelay) { u64 t0 = gt(); while (gt() - t0 < p.rdelay) {} } }
        atomicAdd(p.out + 5, n);
      }
      if (acc == 0x7fffffff) p.out[7] = 1;
    }
    return;
  }
  u64 stale = 0, ok = 0;
  int* sigs[2] = {p.sig[0], p.sig[1]};
  for (int k = 1; k <= p.iters; k++) {
    // notify_dispatch: barrier_block<true>; per_rank_buffer[rank * R + t] = count (to rank t)
    barrier_block<true>(sigs, rank);
    if (t < 2) {
      if (p.wdelay && rank == 1) { u64 t0 = gt(); while (gt() - t0 < p.wdelay) {} }
      if (t == 1 - rank) p.tstore[k] = gt();
      int c = p.cnt_src[t] + (k * 4 + rank);   // DeepEP loads num_tokens_per_rank[t] (LDG.E) first
      p.buf[t][rank * 2 + t] = c;
      // DeepEP's per-expert loop: LDG num_tokens_per_expert, ST into rank t's buffer row `rank`
      for (int i = 0; i < p.nexp; i++) p.buf[t][4 + rank * p.nexp + i] = p.cnt_src[2 + i] + (k * 4 + rank);
    }
    // barrier_block<false> (entry fence), then read the peer's count weakly
    if (t == 0) { /* time of this thread's entry fence */ }
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    if (t == 0) p.tfence[k] = gt();
    if (t == 0 && p.rsync) { go = k; if (!p.noack) while (ack != k) {} }
    if (t == 0 && p.self == 1) { int x = ld_weak(local + (1 - rank) * 2 + rank); if (x == 0x7fffffff) p.out[6] = 1; }  // racy control (pr2)
    sync128();
    if (t < 2) { atomicAdd_system(sigs[rank] + t, TAG); atomicSub_system(sigs[t] + rank, TAG); }
    while (true) { int v = t < 2 ? ld_volatile(sigs[rank] + t) : 0; if (__all_sync(0xffffffff, v <= 0)) break; }
    sync128();
    if (t == 0) {
      int v = ld_weak(local + (1 - rank) * 2 + rank);   // written by the peer
      bool good = v == k * 4 + (1 - rank);
      for (int i = 0; i < p.nexp; i++) good &= ld_weak(local + 4 + (1 - rank) * p.nexp + i) == k * 4 + (1 - rank);
      if (good) ok++; else stale++;
      int mine = ld_weak(local + rank * 2 + rank); (void)mine;
    }
    barrier_block<false>(sigs, rank);                   // closing barrier (after memset in DeepEP)
  }
  if (t == 0) { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); p.out[4] = sm; p.out[0] = ok; p.out[1] = stale; done = 1; *(volatile unsigned long long*)(p.out + 6) = 1; }
}

// Legal co-running stress (never touches the handoff's memory): a streaming copy over 1 GB on
// a second stream (HBM / L2 pressure), or a kernel pushing stores into the peer GPU's scratch
// (NVLink traffic in one direction).
__global__ void hog_copy(const float4* a, float4* b, size_t n, volatile int* stop) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
  while (!*stop) for (size_t j = i; j < n; j += st) b[j] = a[j];
}
__global__ void nvl_push(float4* peer, size_t n, volatile int* stop) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
  float4 v = make_float4(1, 2, 3, 4);
  while (!*stop) for (size_t j = i; j < n; j += st) peer[j] = v;
}
// Co-reader as a separate kernel (xreader=N CTAs per GPU, one warp each): weak loads of the stable
// word until stop. Same address computation trick as the in-CTA reader.
__global__ void xreader(const int* w, volatile int* stop, unsigned long long* cnt) {
  if (threadIdx.x) return;
  unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); atomicOr((unsigned long long*)cnt + 2 + sm / 64, 1ull << (sm % 64));
  unsigned long long n = 0; int acc = 0;
  while (!*stop) { n++; acc += ld_weak_n(w, n); }
  atomicAdd(cnt, n); if (acc == 0x7fffffff) cnt[1] = 1;
}
int main(int argc, char** argv) {
  int hog = -1, push = 0, hogblk = 64, nexp = 0, xr = 0, xcta = 0, xafter = 0;
  int iters = 100000, readers = 4, self = 0, rw = -1, rsync = 0, noack = 0, preload = 0; unsigned rdelay = 0; unsigned wdelay = 0;
  for (int i = 1; i < argc; i++) { std::string a = argv[i]; auto e = a.find('='); long v = atol(a.c_str() + e + 1); auto k = a.substr(0, e);
    if (k == "iters") iters = v; else if (k == "readers") readers = v; else if (k == "wdelay") wdelay = v; else if (k == "self") self = v; else if (k == "rword") rw = v; else if (k == "rsync") rsync = v; else if (k == "rdelay") rdelay = v; else if (k == "noack") noack = v; else if (k == "preload") preload = v; else if (k == "hog") hog = v; else if (k == "push") push = v; else if (k == "hogblk") hogblk = v; else if (k == "nexp") nexp = v; else if (k == "xreader") xr = v; else if (k == "xcta") xcta = v; else if (k == "xafter") xafter = v; }
  P p[2]; int* buf[2]; int* sig[2]; u64* out[2]; u64* tf[2]; u64* ts[2];
  for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); cudaDeviceEnablePeerAccess(1 - d, 0); cudaGetLastError();
    CK(cudaMalloc(&buf[d], 4096)); CK(cudaMemset(buf[d], 0, 4096)); CK(cudaMalloc(&sig[d], 128)); CK(cudaMemset(sig[d], 0, 128));
    CK(cudaMalloc(&out[d], 64)); CK(cudaMemset(out[d], 0, 64));
    CK(cudaMalloc(&tf[d], 8 * (iters + 1))); CK(cudaMalloc(&ts[d], 8 * (iters + 1))); }
  int* cnt[2]; for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaMalloc(&cnt[d], 4096)); CK(cudaMemset(cnt[d], 0, 4096)); }
  int* stop; CK(cudaHostAlloc(&stop, 4, cudaHostAllocMapped | cudaHostAllocPortable)); *stop = 0;
  cudaStream_t hs[2];
  for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaStreamCreateWithFlags(&hs[d], cudaStreamNonBlocking)); }
  size_t hn = (1ull << 30) / 16;
  if (hog >= 0) { CK(cudaSetDevice(hog)); float4 *a, *b; CK(cudaMalloc(&a, hn * 16)); CK(cudaMalloc(&b, hn * 16));
    hog_copy<<<hogblk, 256, 0, hs[hog]>>>(a, b, hn, stop); CK(cudaGetLastError()); }
  if (push) { CK(cudaSetDevice(0)); float4* sc; CK(cudaMalloc(&sc, 256 << 20)); CK(cudaSetDevice(1));
    nvl_push<<<hogblk, 256, 0, hs[1]>>>(sc, (256 << 20) / 16, stop); CK(cudaGetLastError()); }
  unsigned long long* xcnt[2];
  for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaMalloc(&xcnt[d], 64)); CK(cudaMemset(xcnt[d], 0, 64));
    if (xr && !xafter) { xreader<<<xr, 32, 0, hs[d]>>>(buf[d] + (rw < 0 ? 1 - d : rw), stop, xcnt[d]); CK(cudaGetLastError()); } }
  cudaStream_t s[2];
  for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaStreamCreateWithFlags(&s[d], cudaStreamNonBlocking));
    p[d] = P{cnt[d], {buf[0], buf[1]}, {sig[0], sig[1]}, d, iters, readers, wdelay, nexp, self, rsync, rdelay, noack, preload, {rw < 0 ? 1 : rw, rw < 0 ? 0 : rw}, out[d], tf[d], ts[d]};
    notify_like<<<1 + xcta, 128 + 32 * readers, 0, s[d]>>>(p[d]); CK(cudaGetLastError()); }
  if (xr && xafter) for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); xreader<<<xr, 32, 0, hs[d]>>>(buf[d] + (rw < 0 ? 1 - d : rw), stop, xcnt[d]); CK(cudaGetLastError()); }
  for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(s[d])); }
  *stop = 1; __sync_synchronize();
  for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(hs[d])); }
  u64 h[2][8]; for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaMemcpy(h[d], out[d], 64, cudaMemcpyDeviceToHost)); }
  // margin = (peer store issue time) - (my entry fence time), both globaltimer (per-GPU clocks; approximate)
  long long hist[8] = {0}; double sum = 0; long long npos = 0;
  for (int d = 0; d < 2; d++) {
    u64 *hf = (u64*)malloc(8 * (iters + 1)), *hs = (u64*)malloc(8 * (iters + 1));
    CK(cudaSetDevice(d)); CK(cudaMemcpy(hf, tf[d], 8 * (iters + 1), cudaMemcpyDeviceToHost));
    CK(cudaSetDevice(1 - d)); CK(cudaMemcpy(hs, ts[1 - d], 8 * (iters + 1), cudaMemcpyDeviceToHost));
    for (int k = 1; k <= iters; k++) { long long m = (long long)hs[k] - (long long)hf[k]; sum += m; if (m > 0) npos++;
      int b = m < -2000 ? 0 : m < -1000 ? 1 : m < -500 ? 2 : m < 0 ? 3 : m < 500 ? 4 : m < 1000 ? 5 : m < 2000 ? 6 : 7; hist[b]++; }
    free(hf); free(hs);
  }
  printf("loads rank0 %llu rank1 %llu\n", h[0][5], h[1][5]);
  for (int d = 0; d < 2; d++) { unsigned long long xc[8]; CK(cudaSetDevice(d)); CK(cudaMemcpy(xc, xcnt[d], 64, cudaMemcpyDeviceToHost));
    unsigned sm = (unsigned)h[d][4]; int n = 0; for (int i = 0; i < 3; i++) n += __builtin_popcountll(xc[2 + i]);
    printf("gpu%d: same-kernel co-reader CTAs on its SM: %llu; notify CTA on SM %u; xreader on %d SMs, on that SM: %d; xreader loads %llu\n", d, h[d][2], sm, n, (int)((xc[2 + sm / 64] >> (sm % 64)) & 1), xc[0]); }
  printf("RESULT iters=%d readers=%d wdelay=%u reads=%llu stale=%llu (rank0 %llu, rank1 %llu)\n", iters, readers, wdelay,
         h[0][0] + h[0][1] + h[1][0] + h[1][1], h[0][1] + h[1][1], h[0][1], h[1][1]);
  printf("  peer-store-issue minus own-entry-fence (ns, cross-GPU globaltimer): mean %.0f, >0 in %lld of %d; "
         "hist <-2us %lld, -2..-1 %lld, -1..-0.5 %lld, -0.5..0 %lld, 0..0.5 %lld, 0.5..1 %lld, 1..2 %lld, >2us %lld\n",
         sum / (2.0 * iters), npos, 2 * iters, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7]);
}
