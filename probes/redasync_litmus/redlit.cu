/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Message-passing litmus for red.async.release.gpu.global (ptxas 13.0: per-lane ELECT loop of
// MEMBAR.ALL.CTA; UREDGR.E.ADD.NA.S32.STRONG.GPU; DEPBAR.LE SB0, 0) on GB300 (sm_103a).
// A producer warp (one CTA) and a consumer warp (another CTA, normally another SM) run
// `iters` iterations. In iteration k every producer lane l writes payload words of its own
// (weak st.global), then performs the release form on its own flag word; consumer lane l polls
// that flag with ld.acquire.gpu until it reaches k, then reads the payload (weak ld.global)
// and classifies it: all words k = ok, any word < k = stale. The consumer acks with
// st.release.gpu; the producer waits for the ack (ld.acquire.gpu) before iteration k+1.
//
// Build: nvcc -O3 -arch=sm_103a -std=c++17 -o redlit redlit.cu
// Run:   ./redlit <form> [dev=0] [pairs=64] [iters=20000] [nw=4] [ps=32] [dmask=255]
//                        [stress=0] [smode=1] [pp=0] [seed=1]
//   form 0 red.async.release.gpu.global.add.u32     (the form under test)
//        1 red.relaxed.gpu.global.add.u32           (negative control: no release)
//        2 red.release.gpu.global.add.u32           (reference: MEMBAR.ALL.GPU + REDG)
//        3 fence.acq_rel.cta; red.relaxed.gpu        (MEMBAR.ALL.CTA alone, as before UREDGR)
//        4 st.relaxed.gpu.global flag               (negative control, plain store)
//        5 st.release.gpu.global flag               (reference)
//   nw    payload words per lane; js stride between them (words; 32: a line each); ps lane
//         stride in words (32: a line per lane)
//   mode  0: per-lane payload and flag (above); 1: bulk (kern_bulk below; nw = payload words,
//         js = 1 for cp.reduce.async.bulk .add instead of a bulk copy)
//   pp    0: CTA i with CTA i+pairs; 1: CTA 2i with 2i+1
//   stress warps per CTA: smode 1 L2 thrash (weak ld/st over 256 MB), 2 the producer CTA's
//         stress warps store to lines next to the payload
// Output: one line "form=.. ok=.. stale=.. other=.. timeout=.. iters=.. crossSM=.. ...".
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

typedef uint32_t u32;
typedef unsigned long long u64;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2);} } while (0)

static constexpr u64 TIMEOUT = 4ull * 1000 * 1000 * 1000;  // cycles
static constexpr u64 SCR_WORDS = 64ull << 20;

struct P {
  u32 *flag, *pay, *ack, *scr;
  u64* cnt;
  u32* smid;
  u32* abort_;
  u32 pairs, iters, nw, ps, js, dmask, S, smode, pp, seed;
};
enum { C_OK = 0, C_STALE, C_OTHER, C_TIMEOUT, C_ITERS, C_WAITED, C_SINK, C_N };

__device__ __forceinline__ u32 lane() { u32 l; asm("mov.u32 %0, %%laneid;" : "=r"(l)); return l; }
__device__ __forceinline__ u32 smid() { u32 s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
__device__ __forceinline__ u32 rng(u32& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
__device__ __forceinline__ void delay(u32& s, u32 mask) {
  if (!mask) return;
  u32 d = rng(s) & mask;
  u64 t0 = clock64();
  while (clock64() - t0 < d) {}
}
__device__ __forceinline__ u32 ld_acq(const u32* a) {
  u32 v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
__device__ __forceinline__ void st_rel(u32* a, u32 v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void st_weak(u32* a, u32 v) {
  asm volatile("st.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ u32 ld_vol(const u32* a) {
  u32 v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
__device__ __forceinline__ u32 ld_weak(const u32* a) {
  u32 v; asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
template <int F> __device__ __forceinline__ void signal(u32* f, u32 k) {
  if constexpr (F == 0) asm volatile("red.async.release.gpu.global.add.u32 [%0], 1;" ::"l"(f) : "memory");
  if constexpr (F == 1) asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" ::"l"(f) : "memory");
  if constexpr (F == 2) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(f) : "memory");
  if constexpr (F == 3) asm volatile("fence.acq_rel.cta;\n\tred.relaxed.gpu.global.add.u32 [%0], 1;" ::"l"(f) : "memory");
  if constexpr (F == 4) asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(f), "r"(k) : "memory");
  if constexpr (F == 5) asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(f), "r"(k) : "memory");
}

template <int F> __global__ void kern(P p) {
  const u32 b = blockIdx.x, w = threadIdx.x / 32, l = lane();
  u32 s = p.seed * 2654435761u ^ (b * 97 + threadIdx.x) * 40503u;
  if (!s) s = 1;
  u32 pair, prod;
  if (p.pp == 0) { pair = b % p.pairs; prod = b < p.pairs; }
  else { pair = b / 2; prod = (b & 1) == 0; }
  if (w > 0) {  // stress warps
    if (p.smode == 1) {
      u64 i = (u64)(b * 32 + w) * 4099 + l;
      u32 acc = 0;
      while (ld_vol(&p.abort_[0]) == 0) {
        i = (i * 6364136223846793005ull + 1442695040888963407ull);
        u32* a = p.scr + ((i >> 20) % SCR_WORDS);
        acc += ld_weak(a);
        st_weak(a + 32, acc);
      }
      if (acc == 0xdeadbeef) atomicAdd(&p.cnt[C_SINK], 1);
    } else if (p.smode == 2 && prod) {
      u32* base = p.pay + (u64)pair * 32 * p.ps * 2 + 32 * p.ps;  // the lines after the payload
      u32 k = 0;
      while (ld_vol(&p.abort_[0]) == 0) st_weak(base + ((k++ * 32 + l) % (32 * p.ps)), k);
    }
    return;
  }
  if (l == 0) p.smid[b] = smid();
  u32* flag = p.flag + (u64)pair * 32 + l;
  u32* ack = p.ack + (u64)pair * 32 + l;
  u32* pay = p.pay + (u64)pair * 32 * p.ps * 2 + (u64)l * p.ps;
  u64 ok = 0, stale = 0, other = 0, waited = 0, timeout = 0;
  for (u32 k = 1; k <= p.iters; k++) {
    if (prod) {
      u64 t0 = clock64();
      while (ld_acq(ack) != k - 1) {
        if (clock64() - t0 > TIMEOUT) { timeout++; atomicExch(&p.abort_[0], 1); break; }
      }
      if (timeout) break;
      delay(s, p.dmask);
      for (u32 j = 0; j < p.nw; j++) st_weak(pay + j * p.js, k);
      signal<F>(flag, k);
    } else {
      delay(s, p.dmask);
      u64 t0 = clock64();
      bool first = true;
      while (ld_acq(flag) < k) {
        if (first) { waited++; first = false; }
        if (clock64() - t0 > TIMEOUT) { timeout++; atomicExch(&p.abort_[0], 1); break; }
      }
      if (timeout) break;
      u32 bad = 0, oth = 0;
      for (u32 j = 0; j < p.nw; j++) {
        u32 v = ld_weak(pay + j * p.js);
        if (v < k) bad = 1; else if (v != k) oth = 1;
      }
      if (bad) stale++; else if (oth) other++; else ok++;
      st_rel(ack, k);
    }
  }
  if (!prod) {
    atomicAdd(&p.cnt[C_OK], ok); atomicAdd(&p.cnt[C_STALE], stale);
    atomicAdd(&p.cnt[C_OTHER], other); atomicAdd(&p.cnt[C_WAITED], waited);
  }
  atomicAdd(&p.cnt[C_TIMEOUT], timeout);
  __syncwarp();
  // The last consumer warp to finish stops the stress warps.
  if (!prod && l == 0 && atomicAdd(&p.cnt[C_ITERS], 1) + 1 == p.pairs) atomicExch(&p.abort_[0], 1);
}


// mode=1 (bulk): FA backward's deterministic-accumulation shape. Producer warp 0 fills a
// shared buffer with k, fence.proxy.async.shared::cta, then lane 0 copies it to global with
// cp.async.bulk.global.shared::cta.bulk_group (bk=0) or adds it with
// cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 onto a zeroed... (bk=1, the
// payload then holds the running sum k(k+1)/2), commit_group, wait_group 0; bar.sync with
// warp 1, whose lane 0 then signals with the form. The consumer checks all `bw` words.
template <int F> __global__ void kern_bulk(P p) {
  extern __shared__ __align__(128) u32 sbuf[];
  const u32 b = blockIdx.x, w = threadIdx.x / 32, l = lane();
  u32 s = p.seed * 2654435761u ^ (b * 97 + threadIdx.x) * 40503u;
  if (!s) s = 1;
  u32 pair, prod;
  if (p.pp == 0) { pair = b % p.pairs; prod = b < p.pairs; }
  else { pair = b / 2; prod = (b & 1) == 0; }
  const u32 bw = p.nw;  // words in the bulk payload (multiple of 4)
  u32* flag = p.flag + (u64)pair * 32;
  u32* ack = p.ack + (u64)pair * 32;
  u32* pay = p.pay + (u64)pair * p.ps;
  if (threadIdx.x == 0) p.smid[b] = smid();
  u64 ok = 0, stale = 0, other = 0, waited = 0, timeout = 0;
  for (u32 k = 1; k <= p.iters; k++) {
    u32 want = p.js ? k * (k + 1) / 2 : k;  // js = bk
    if (prod) {
      if (w == 0) {
        if (l == 0) {
          u64 t0 = clock64();
          while (ld_acq(ack) != k - 1) {
            if (clock64() - t0 > TIMEOUT) { timeout++; atomicExch(&p.abort_[0], 1); break; }
          }
        }
        if (__shfl_sync(~0u, (u32)timeout, 0)) timeout = 1;
        if (!timeout) {
          delay(s, p.dmask);
          for (u32 j = l; j < bw; j += 32) sbuf[j] = k;
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          __syncwarp();
          if (l == 0) {
            u32 sa = (u32)__cvta_generic_to_shared(sbuf);
            if (p.js)
              asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;" ::"l"(pay), "r"(sa), "r"(bw * 4) : "memory");
            else
              asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(pay), "r"(sa), "r"(bw * 4) : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
          }
        }
      }
      __syncthreads();
      if (__syncthreads_or(timeout)) { timeout = timeout ? timeout : 0; break; }
      if (w == 1 && l == 0) signal<F>(flag, k);
    } else {
      if (w != 0) continue;
      delay(s, p.dmask);
      u64 t0 = clock64();
      bool first = true;
      if (l == 0) {
        while (ld_acq(flag) < k) {
          if (first) { waited++; first = false; }
          if (clock64() - t0 > TIMEOUT) { timeout++; atomicExch(&p.abort_[0], 1); break; }
        }
      }
      // The other lanes read after lane 0's acquire (bar.warp.sync orders them).
      __syncwarp();
      if (__shfl_sync(~0u, (u32)timeout, 0)) { timeout = l == 0 ? timeout : 0; break; }
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      u32 bad = 0, oth = 0;
      for (u32 j = l; j < bw; j += 32) {
        u32 v = ld_weak(pay + j);
        if (v < want) bad = 1; else if (v != want) oth = 1;
      }
      bad = __any_sync(~0u, bad); oth = __any_sync(~0u, oth);
      if (l == 0) { if (bad) stale++; else if (oth) other++; else ok++; st_rel(ack, k); }
    }
  }
  if (!prod && w == 0 && l == 0) {
    atomicAdd(&p.cnt[C_OK], ok); atomicAdd(&p.cnt[C_STALE], stale);
    atomicAdd(&p.cnt[C_OTHER], other); atomicAdd(&p.cnt[C_WAITED], waited);
  }
  if (threadIdx.x == 0) atomicAdd(&p.cnt[C_TIMEOUT], timeout);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: redlit <form> [key=val..]\n"); return 1; }
  int form = atoi(argv[1]);
  std::map<std::string, long> kv = {{"dev", 0}, {"pairs", 64}, {"iters", 20000}, {"nw", 4}, {"ps", 32}, {"js", 1},
                                    {"dmask", 255}, {"mode", 0}, {"stress", 0}, {"smode", 1}, {"pp", 0}, {"seed", 1}};
  for (int i = 2; i < argc; i++) {
    const char* e = strchr(argv[i], '=');
    if (!e) { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    std::string k(argv[i], e - argv[i]);
    if (!kv.count(k)) { fprintf(stderr, "unknown key %s\n", k.c_str()); return 1; }
    kv[k] = atol(e + 1);
  }
  CK(cudaSetDevice((int)kv["dev"]));
  P p{};
  p.pairs = kv["pairs"]; p.iters = kv["iters"]; p.nw = kv["nw"]; p.ps = kv["ps"]; p.js = kv["js"]; p.dmask = kv["dmask"];
  p.S = kv["stress"]; p.smode = kv["smode"]; p.pp = kv["pp"]; p.seed = kv["seed"];
  if (p.nw < 1 || (p.nw - 1) * p.js >= p.ps) { fprintf(stderr, "need 1 <= nw and (nw - 1) * js < ps\n"); return 1; }
  size_t payw = (size_t)p.pairs * 32 * p.ps * 2;
  CK(cudaMalloc(&p.flag, p.pairs * 32 * 4)); CK(cudaMemset(p.flag, 0, p.pairs * 32 * 4));
  CK(cudaMalloc(&p.ack, p.pairs * 32 * 4)); CK(cudaMemset(p.ack, 0, p.pairs * 32 * 4));
  CK(cudaMalloc(&p.pay, payw * 4)); CK(cudaMemset(p.pay, 0, payw * 4));
  CK(cudaMalloc(&p.cnt, C_N * 8)); CK(cudaMemset(p.cnt, 0, C_N * 8));
  CK(cudaMalloc(&p.smid, 2 * p.pairs * 4));
  CK(cudaMalloc(&p.abort_, 4)); CK(cudaMemset(p.abort_, 0, 4));
  p.scr = nullptr;
  if (p.S && p.smode == 1) { CK(cudaMalloc(&p.scr, SCR_WORDS * 4 + 4096)); CK(cudaMemset(p.scr, 0, SCR_WORDS * 4)); }
  dim3 grid(2 * p.pairs), block(32 * (1 + p.S));
  void (*ks[6])(P) = {kern<0>, kern<1>, kern<2>, kern<3>, kern<4>, kern<5>};
  void (*kb[6])(P) = {kern_bulk<0>, kern_bulk<1>, kern_bulk<2>, kern_bulk<3>, kern_bulk<4>, kern_bulk<5>};
  if (form < 0 || form > 5) { fprintf(stderr, "form 0..5\n"); return 1; }
  size_t smem = 0;
  if (kv["mode"] == 1) {
    if (p.nw % 4 || p.nw > p.ps || p.S) { fprintf(stderr, "bulk: nw multiple of 4, nw <= ps, stress=0\n"); return 1; }
    smem = p.nw * 4;
    block = dim3(64);
    CK(cudaFuncSetAttribute(kb[form], cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  if (kv["mode"] == 1) kb[form]<<<grid, block, smem>>>(p);
  else ks[form]<<<grid, block>>>(p);
  CK(cudaGetLastError());
  cudaEventRecord(e1);
  CK(cudaDeviceSynchronize());
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  u64 c[C_N]; CK(cudaMemcpy(c, p.cnt, sizeof c, cudaMemcpyDeviceToHost));
  u32* sm = (u32*)malloc(2 * p.pairs * 4); CK(cudaMemcpy(sm, p.smid, 2 * p.pairs * 4, cudaMemcpyDeviceToHost));
  u32 cross = 0;
  for (u32 i = 0; i < p.pairs; i++) {
    u32 a = p.pp == 0 ? i : 2 * i, bb = p.pp == 0 ? i + p.pairs : 2 * i + 1;
    cross += sm[a] != sm[bb];
  }
  printf("form=%d ok=%llu stale=%llu other=%llu timeout=%llu waited=%llu crossSM=%u/%u", form, c[C_OK], c[C_STALE],
         c[C_OTHER], c[C_TIMEOUT], c[C_WAITED], cross, p.pairs);
  for (auto& [k, v] : kv) printf(" %s=%ld", k.c_str(), v);
  printf(" ms=%.0f\n", ms);
  return 0;
}
