/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Ring-slot reuse litmus (DeepEP intranode dispatch, receiver side, k0.cubin 0x4250..0x7fc0):
// the receiver bulk-reads a ring slot into shared memory (UBLKCP.S.G + mbarrier wait), then
// publishes its head with a relaxed store and no MEMBAR (elected lane, STG.E.STRONG.SYS). The
// remote sender polls the head (LDG.E.STRONG.SYS), then overwrites the slot (STG.E.128) and
// releases the tail. Question: can the receiver's bulk read of iteration k see the sender's
// iteration k+1 overwrite (load buffering through the relaxed head)?
//
// Build:  nvcc -O3 -arch=sm_103a -std=c++17 -o ringwar ringwar.cu
// Run:    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 ./ringwar [key=val ...]
//   hb=0  DeepEP order: wait on the mbarrier, then store the head
//   hb=1  positive control: store the head right after issuing the bulk copy, then wait
//   cdev/pdev (receiver / sender GPU), pairs, W (warps per CTA), bytes (slot size), iters,
//   dmask (random delays, cycles), seed, loc=1 puts the slot in the sender's memory (the
//   receiver's bulk read crosses NVLink; DeepEP: loc=0)
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <chrono>

typedef uint32_t u32;
typedef unsigned long long u64;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2);} } while (0)

static constexpr u64 TIMEOUT = 40ull * 1000 * 1000 * 1000;
static constexpr int MAXW = 8, MAXB = 8192;

struct P {
  u32 *slot, *tail, *head;  // slot + tail in receiver memory, head in sender memory
  u64* cnt;                 // [0] ok [1] early (saw k+1: violation) [2] stale (< k) [3] timeout [4] iters
  u32 pairs, W, bytes, iters, dmask, seed, hb, role;  // role 0 receiver, 1 sender
};

__device__ __forceinline__ u32 ldv(const u32* a) { u32 v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ u32 lda(const u32* a) { u32 v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ void stv(u32* a, u32 v) { asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void strel(u32* a, u32 v) { asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory"); }
__device__ __forceinline__ u32 rng(u32& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
__device__ __forceinline__ void delay(u32& s, u32 m) {
  if (!m) return;
  u32 d = rng(s) & m; u64 t0 = clock64(); while (clock64() - t0 < d) {}
}

__global__ void k(P p) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) u64 mbar[MAXW];
  const u32 w = threadIdx.x / 32, l = threadIdx.x % 32;
  const u32 inst = blockIdx.x * p.W + w;
  const u32 words = p.bytes / 4;
  u32* slot = p.slot + (u64)inst * words;
  u32* tail = p.tail + inst * 32;
  u32* head = p.head + inst * 32;
  u32 seed = p.seed * 7919u + inst * 131u + l + 1;
  unsigned char* buf = sm + (u64)w * p.bytes;
  u32 bar = (u32)__cvta_generic_to_shared(&mbar[w]);
  u32 sbuf = (u32)__cvta_generic_to_shared(buf);
  u64 c[5] = {0};
  if (p.role == 1) {  // sender: wait head >= k-1, overwrite slot with k (STG.E.128), release tail = k
    for (u32 kk = 1; kk <= p.iters; kk++) {
      u64 t0 = clock64();
      while (ldv(head) < kk - 1) if (clock64() - t0 > TIMEOUT) { if (!l) atomicAdd(&p.cnt[3], 1ull); return; }
      delay(seed, p.dmask);
      for (u32 i = l * 4; i < words; i += 128)
        asm volatile("st.global.v4.u32 [%0], {%1,%1,%1,%1};" ::"l"(slot + i), "r"(kk) : "memory");
      __syncwarp();
      if (l == 0) strel(tail, kk);
    }
    return;
  }
  if (l == 0) asm volatile("mbarrier.init.shared.b64 [%0], 1;" ::"r"(bar) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncwarp();
  u32 phase = 0;
  for (u32 kk = 1; kk <= p.iters; kk++) {
    u64 t0 = clock64();
    bool bad = false;
    if (l == 0) while (lda(tail) < kk) if (clock64() - t0 > TIMEOUT) { bad = true; break; }
    if (__any_sync(~0u, bad)) { if (!l) atomicAdd(&p.cnt[3], 1ull); return; }
    delay(seed, p.dmask);
    if (l == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar), "r"(p.bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   ::"r"(sbuf), "l"(slot), "r"(p.bytes), "r"(bar) : "memory");
      if (p.hb) stv(head, kk);  // control: head published while the copy is in flight
    }
    u32 done = 0;
    while (!done)
      asm volatile("{ .reg .pred q; mbarrier.try_wait.parity.shared.b64 q, [%1], %2; selp.u32 %0, 1, 0, q; }"
                   : "=r"(done) : "r"(bar), "r"(phase) : "memory");
    phase ^= 1;
    __syncwarp();
    if (!p.hb && l == 0) stv(head, kk);  // DeepEP: relaxed head store after the wait, no MEMBAR
    // check the copy (smem) after publishing, as DeepEP's later UBLKCP.G.S does
    bool early = false, stale = false;
    for (u32 i = l; i < words; i += 32) {
      u32 v = ((volatile u32*)buf)[i];
      early |= v > kk; stale |= v < kk;
    }
    early = __any_sync(~0u, early); stale = __any_sync(~0u, stale);
    if (l == 0) { if (early) c[1]++; else if (stale) c[2]++; else c[0]++; c[4]++; }
    __syncwarp();
  }
  for (int i = 0; i < 5; i++) {
    u64 x = c[i];
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(~0u, x, o);
    if (l == 0 && x) atomicAdd(&p.cnt[i], x);
  }
}

int main(int argc, char** argv) {
  std::map<std::string, long> kv = {{"pairs", 148}, {"W", 4}, {"bytes", 4096}, {"iters", 100000},
      {"dmask", 255}, {"seed", 1}, {"hb", 0}, {"cdev", 0}, {"pdev", 1}, {"loc", 0}};
  for (int i = 1; i < argc; i++) {
    const char* e = strchr(argv[i], '=');
    std::string key(argv[i], e ? e - argv[i] : strlen(argv[i]));
    if (!e || !kv.count(key)) { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    kv[key] = atol(e + 1);
  }
  P p = {};
  p.pairs = kv["pairs"]; p.W = kv["W"]; p.bytes = kv["bytes"]; p.iters = kv["iters"];
  p.dmask = kv["dmask"]; p.seed = kv["seed"]; p.hb = kv["hb"];
  int cdev = kv["cdev"], pdev = kv["pdev"];
  if (p.W > MAXW || p.bytes > MAXB || p.bytes % 512 || cdev == pdev) { fprintf(stderr, "bad W/bytes/devs\n"); return 1; }
  CK(cudaSetDevice(cdev)); cudaDeviceEnablePeerAccess(pdev, 0); cudaGetLastError();
  CK(cudaSetDevice(pdev)); cudaDeviceEnablePeerAccess(cdev, 0); cudaGetLastError();
  u64 n = (u64)p.pairs * p.W;
  auto alloc = [](int d, u64 b) { void* q; CK(cudaSetDevice(d)); CK(cudaMalloc(&q, b)); CK(cudaMemset(q, 0, b)); return q; };
  p.slot = (u32*)alloc(kv["loc"] ? pdev : cdev, n * p.bytes);
  p.tail = (u32*)alloc(cdev, n * 128);
  p.head = (u32*)alloc(pdev, n * 128);
  p.cnt = (u64*)alloc(cdev, 64);
  int smem = p.W * p.bytes;
  cudaStream_t s[2];
  int dev[2] = {cdev, pdev};
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 1; r >= 0; r--) {
    CK(cudaSetDevice(dev[r])); CK(cudaStreamCreate(&s[r]));
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024));
    P q = p; q.role = r;
    k<<<p.pairs, p.W * 32, r == 0 ? smem : 160 * 1024, s[r]>>>(q);  // 160 KB: one CTA per SM
    CK(cudaGetLastError());
  }
  for (int r = 0; r < 2; r++) { CK(cudaSetDevice(dev[r])); CK(cudaStreamSynchronize(s[r])); }
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  u64 h[5]; CK(cudaSetDevice(cdev)); CK(cudaMemcpy(h, p.cnt, sizeof h, cudaMemcpyDeviceToHost));
  printf("RESULT ringwar loc=%ld hb=%u cdev=%d pdev=%d pairs=%u W=%u bytes=%u dmask=%u seed=%u iters=%u | "
         "copies=%llu ok=%llu early=%llu stale=%llu timeout=%llu secs=%.3f\n",
         kv["loc"], p.hb, cdev, pdev, p.pairs, p.W, p.bytes, p.dmask, p.seed, p.iters, h[0] + h[1] + h[2], h[0], h[1], h[2], h[3], secs);
  return h[3] ? 3 : 0;
}
