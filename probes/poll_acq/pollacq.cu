/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Litmus tests for E-POLL-ACQ and E-SMEM-CUMUL on GB300 (sm_103a): message passing from a
// producer on another SM or another GPU (NVLink) to a consumer that leaves a DeepEP-shaped
// poll loop, or that learns of the flag through a shared-memory relay (DeepEP combine).
//
// Build:  nvcc -O3 -arch=sm_103a -std=c++17 -o pollacq pollacq.cu
// Run:    ./pollacq <test> <form> <rd> [key=val ...]      (CUDA_VISIBLE_DEVICES is not used;
//         cdev= / pdev= pick the consumer / producer GPU; not in $PTXV_EXCLUDE_GPUS)
//
//   test  poll   E-POLL-ACQ: the consumer warp polls the flag(s) itself.
//         relay  E-SMEM-CUMUL: a coordinator warp polls the flag(s) and relays the value
//                through st.volatile.shared; a receiver warp polls the relay, then reads.
//   form  poll:  0 warp poll    lanes < nf: ld.volatile flag; __all_sync; clock timeout; loop
//                               (barrier_block: LDG.E.STRONG.SYS, VOTE.ALL, CS2R SR_CLOCKLO)
//                1 lane spin    straight-line per-lane spin, no vote (control shape)
//                2 warp poll + bar.sync: a second warp waits at a named barrier, then reads
//                               (barrier_block's trailing __syncthreads)
//         relay: 0 coordinator ld.acquire.sys (LDG.E.STRONG.SYS; CCTL.IVALL), then
//                               st.volatile.shared (STS) -- combine's tail relay
//                1 coordinator ld.volatile (no CCTL.IVALL): not an acquire (positive control)
//   rd    payload read by the consumer / receiver (4 words per lane):
//         0 generic weak ld.u32 x4        LD.E          notify_dispatch's reads
//         1 ld.global.u32 x4              LDG.E
//         2 ld.volatile.global.v4.u32     LDG.E.128.STRONG.SYS   combine's channel reads
//         3 fence.acq_rel.sys; ld.global  MEMBAR.ALL.SYS; CCTL.IVALL; LDG.E  (negative control)
//         4 ld.global.nc.v4.u32           LDG.E.128.CONSTANT
//
// keys: pairs= iters= W= (instances per pair) nf= (polling lanes) ps= (lane stride, words)
//       wf= (0: generic st x4; fence.acq_rel.sys; __syncwarp; atomicAdd_system flag
//            1: st.global.v4; __syncwarp; st.release.sys.global flag)
//       pr= cache state of the payload in the consumer's L1 when the poll exits:
//           0 fence.acq_rel.sys (CCTL.IVALL) before the poll, no payload load after it
//           1 the consumer's previous-iteration read (natural)
//           2 weak load of the payload just before the poll
//           3 weak load of the payload in every poll iteration (and before)
//       loc= 0 payload and flags in consumer GPU memory (DeepEP), 1 payload in producer GPU memory
//       pp=  intra-GPU pairing: 0 CTA i with CTA i+pairs, 1 CTA 2i with 2i+1, 2 same CTA (same SM)
//       dmask= producer / consumer random delay (cycles, mask)  stress= S warps per CTA
//       smode= stress: 0 none, 1 L2 thrash (weak ld/st over 512 MB), 2 hot atomics,
//              3 NVLink (ld/st to peer scratch), 4 L1 thrash (1 MB), 5 weak loads of the CTA's
//              own payload lines by the stress warps      seed=  sh= KB smem per CTA
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <chrono>

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

enum { T_POLL = 0, T_RELAY = 1 };
static constexpr u64 TIMEOUT = 40ull * 1000 * 1000 * 1000;  // cycles (~20 s)
static constexpr int MAXW = 4;
static constexpr u64 SCR_WORDS = 128ull << 20;  // 512 MB stress scratch per GPU

struct P {
  u32 *flag, *pay, *ack;     // flags + payload (consumer memory unless loc=1), acks (producer memory)
  u32 *scr, *pscr;           // local / peer stress scratch
  u64* cnt;                  // [16] counters
  u32* first;                // first violation record [16]
  u32* abort_;               // global abort word (consumer memory)
  u32 *smid;                 // [2 * pairs] consumer / producer SM ids
  u32 pairs, iters, W, nf, ps, wf, pr, pp, dmask, S, smode, seed, nc, np, intra;
};
// counters
enum { C_OK = 0, C_STALE, C_OTHER, C_TIMEOUT, C_ITERS, C_WAITED, C_POLLS, C_SINK, C_NCNT };

__device__ __forceinline__ u32 lane() { u32 l; asm("mov.u32 %0, %%laneid;" : "=r"(l)); return l; }
__device__ __forceinline__ u32 smid() { u32 s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
__device__ __forceinline__ u32 rng(u32& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
__device__ __forceinline__ void delay(u32& s, u32 mask) {
  if (!mask) return;
  u32 r = rng(s), d = r & mask;
  if (((r >> 24) & 15) == 0) { __nanosleep(d); return; }
  u64 t0 = clock64();
  while (clock64() - t0 < d) {}
}

// ------------------------------------------------------------------ memory operations
__device__ __forceinline__ u32 ld_volatile(const u32* a) {
  u32 v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
__device__ __forceinline__ u32 ld_acquire_sys(const u32* a) {
  u32 v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
__device__ __forceinline__ void st_volatile(u32* a, u32 v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_sys(u32* a, u32 v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ u32 lds_volatile(const u32* a) {
  u32 v; asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"((u32)__cvta_generic_to_shared(a)) : "memory"); return v;
}
__device__ __forceinline__ void sts_volatile(u32* a, u32 v) {
  asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"((u32)__cvta_generic_to_shared(a)), "r"(v) : "memory");
}
__device__ __forceinline__ u32 ld_weak(const u32* a) {
  u32 v; asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
// payload read forms
template <int RD> __device__ __forceinline__ void rd_pay(const u32* a, u32* o) {
  if constexpr (RD == 0) {
    asm volatile("ld.u32 %0, [%1];" : "=r"(o[0]) : "l"(a) : "memory");
    asm volatile("ld.u32 %0, [%1+4];" : "=r"(o[1]) : "l"(a) : "memory");
    asm volatile("ld.u32 %0, [%1+8];" : "=r"(o[2]) : "l"(a) : "memory");
    asm volatile("ld.u32 %0, [%1+12];" : "=r"(o[3]) : "l"(a) : "memory");
  } else if constexpr (RD == 1 || RD == 3) {
    if constexpr (RD == 3) fence_sys();
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(o[0]) : "l"(a) : "memory");
    asm volatile("ld.global.u32 %0, [%1+4];" : "=r"(o[1]) : "l"(a) : "memory");
    asm volatile("ld.global.u32 %0, [%1+8];" : "=r"(o[2]) : "l"(a) : "memory");
    asm volatile("ld.global.u32 %0, [%1+12];" : "=r"(o[3]) : "l"(a) : "memory");
  } else if constexpr (RD == 2) {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "l"(a) : "memory");
  } else {
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "l"(a) : "memory");
  }
}
// prime: weak loads of the payload words (allocate the line in L1)
__device__ __forceinline__ u32 prime(const u32* a) {
  u32 x, y;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(a) : "memory");
  asm volatile("ld.global.u32 %0, [%1+12];" : "=r"(y) : "l"(a) : "memory");
  return x ^ y;
}

// ------------------------------------------------------------------ roles
__device__ __forceinline__ bool aborted(const P& p) { return ld_volatile(p.abort_) != 0; }
__device__ __forceinline__ void do_abort(const P& p) {
  st_volatile(p.abort_, 1);
  atomicAdd(&p.cnt[C_TIMEOUT], 1ull);
}

__device__ void producer(const P& p, u32 slot, u32 seed) {
  const u32 l = lane();
  u32* pay = p.pay + (u64)slot * 32 * p.ps + l * p.ps;
  u32* flag = p.flag + slot * 32;
  u32* ack = p.ack + slot * 32;
  for (u32 k = 1; k <= p.iters; k++) {
    // wait until the consumer finished iteration k-1
    if (l == 0) {
      u64 t0 = clock64();
      while (ld_volatile(ack) < k - 1) {
        if (clock64() - t0 > TIMEOUT || aborted(p)) { do_abort(p); break; }
      }
    }
    if (__shfl_sync(~0u, ld_volatile(p.abort_), 0)) return;
    __syncwarp();
    delay(seed, p.dmask);
    if (p.wf == 0) {
      asm volatile("st.u32 [%0], %1;" ::"l"(pay), "r"(k) : "memory");
      asm volatile("st.u32 [%0+4], %1;" ::"l"(pay), "r"(k) : "memory");
      asm volatile("st.u32 [%0+8], %1;" ::"l"(pay), "r"(k) : "memory");
      asm volatile("st.u32 [%0+12], %1;" ::"l"(pay), "r"(k) : "memory");
      fence_sys();
      __syncwarp();
      if (l < p.nf) {  // generic atom (ATOM.E.ADD.STRONG.SYS), as DeepEP's atomicAdd_system
        u32 old;
        asm volatile("atom.add.sys.u32 %0, [%1], 1;" : "=r"(old) : "l"(flag + l) : "memory");
        (void)old;
      }
    } else {
      asm volatile("st.global.v4.u32 [%0], {%1,%1,%1,%1};" ::"l"(pay), "r"(k) : "memory");
      __syncwarp();
      if (l < p.nf) st_release_sys(flag + l, k);
    }
  }
}

template <int FORM, int RD>
__device__ __forceinline__ void classify(const P& p, u32 k, const u32* v, u64* c, u32 slot, u32 csm) {
  bool ok = v[0] == k && v[1] == k && v[2] == k && v[3] == k;
  bool stale = v[0] < k || v[1] < k || v[2] < k || v[3] < k;
  if (ok) c[C_OK]++;
  else if (stale) c[C_STALE]++;
  else c[C_OTHER]++;
  if (!ok && atomicCAS(p.first, 0u, 1u) == 0u) {
    p.first[1] = k; p.first[2] = lane(); p.first[3] = slot; p.first[4] = csm;
    p.first[5] = v[0]; p.first[6] = v[1]; p.first[7] = v[2]; p.first[8] = v[3];
  }
}

// E-POLL-ACQ consumer. FORM 0/1: this warp polls and reads. FORM 2: poller warp (rw = 0)
// and reader warp (rw = 1) meet at named barrier 1 + w.
template <int FORM, int RD>
__device__ void consumer_poll(const P& p, u32 slot, u32 w, u32 rw, u32 seed, u64* c) {
  const u32 l = lane();
  const u32* pay = p.pay + (u64)slot * 32 * p.ps + l * p.ps;
  const u32* flag = p.flag + slot * 32;
  u32* ack = p.ack + slot * 32;
  const u32 csm = smid();
  u32 sink = 0;
  const bool reader = FORM != 2 || rw == 1;
  const bool poller = FORM != 2 || rw == 0;
  for (u32 k = 1; k <= p.iters; k++) {
    delay(seed, p.dmask);
    if (reader) {
      if (p.pr == 0) fence_sys();
      else if (p.pr >= 2) sink += prime(pay);
    }
    bool bad = false;
    if (poller) {
      u64 t0 = clock64();
      u32 polls = 0;
      if constexpr (FORM == 1) {
        const u32* f = flag + (l % p.nf);
        while (ld_volatile(f) < k) {
          polls++;
          if (p.pr == 3) sink += prime(pay);
          if (clock64() - t0 > TIMEOUT) { bad = true; break; }
        }
      } else {
        while (true) {
          u32 v = l < p.nf ? ld_volatile(flag + l) : k;
          if (__all_sync(~0u, v >= k)) break;
          polls++;
          if (FORM == 0 && p.pr == 3) sink += prime(pay);
          if (clock64() - t0 > TIMEOUT) { bad = true; break; }
        }
      }
      bad = __any_sync(~0u, bad);
      if (l == 0) { c[C_POLLS] += polls; if (polls) c[C_WAITED]++; c[C_ITERS]++; }
      if (bad) { if (l == 0) do_abort(p); }
    }
    if constexpr (FORM == 2) asm volatile("bar.sync %0, 64;" ::"r"(1 + w) : "memory");
    if (bad || (FORM == 2 && __shfl_sync(~0u, ld_volatile(p.abort_), 0))) break;
    if (reader) {
      u32 v[4];
      rd_pay<RD>(pay, v);
      classify<FORM, RD>(p, k, v, c, slot, csm);
      __syncwarp();
      if (l == 0) st_volatile(ack, k);
    }
  }
  if (l == 0) c[C_SINK] += sink;
}

// E-SMEM-CUMUL consumer: coordinator warp (rw = 0) and receiver warp (rw = 1).
template <int FORM, int RD>
__device__ void consumer_relay(const P& p, u32 slot, u32 w, u32 rw, u32 seed, u64* c, u32* relay) {
  const u32 l = lane();
  const u32* pay = p.pay + (u64)slot * 32 * p.ps + l * p.ps;
  const u32* flag = p.flag + slot * 32;
  u32* ack = p.ack + slot * 32;
  const u32 csm = smid();
  u32 sink = 0;
  if (rw == 0) {  // coordinator: re-publishes the tail every iteration, like combine's lanes
    for (u32 k = 1; k <= p.iters; k++) {
      u64 t0 = clock64();
      while (true) {
        u32 v = 0;
        if (l < p.nf) {
          v = FORM == 0 ? ld_acquire_sys(flag + l) : ld_volatile(flag + l);
          sts_volatile(relay + l, v);
        }
        if (__all_sync(~0u, l >= p.nf || v >= k)) break;
        if (clock64() - t0 > TIMEOUT) { if (l == 0) do_abort(p); return; }
      }
    }
    return;
  }
  for (u32 k = 1; k <= p.iters; k++) {
    delay(seed, p.dmask);
    if (p.pr == 0) fence_sys();
    else if (p.pr >= 2) sink += prime(pay);
    u64 t0 = clock64();
    u32 polls = 0;
    bool bad = false;
    while (true) {
      u32 r = lds_volatile(relay + (l % p.nf));
      if (__all_sync(~0u, r >= k)) break;
      polls++;
      if (p.pr == 3) sink += prime(pay);
      if (clock64() - t0 > TIMEOUT) { bad = true; break; }
    }
    if (__any_sync(~0u, bad)) { if (l == 0) do_abort(p); break; }
    if (l == 0) { c[C_POLLS] += polls; if (polls) c[C_WAITED]++; c[C_ITERS]++; }
    u32 v[4];
    rd_pay<RD>(pay, v);
    classify<FORM, RD>(p, k, v, c, slot, csm);
    __syncwarp();
    if (l == 0) st_volatile(ack, k);
  }
  if (l == 0) c[C_SINK] += sink;
}

__device__ void stress(const P& p, volatile u32* done, u32 seed, u32 pair) {
  const u32 l = lane();
  u32 s = seed * 2654435761u + l * 97 + 1, sink = 0;
  while (!*done) {
    for (int i = 0; i < 32; i++) {
      u32 r = rng(s);
      switch (p.smode) {
        case 1: {
          u32* a = p.scr + ((u64)r * 32 % SCR_WORDS);
          if (r & 1) asm volatile("st.global.u32 [%0], %1;" ::"l"(a), "r"(r) : "memory");
          else sink += ld_weak(a);
        } break;
        case 2: atomicAdd(p.scr + (r & 7) * 32, 1u); break;
        case 3: {
          u32* a = p.pscr + ((u64)r * 32 % (SCR_WORDS / 4));
          if (r & 1) asm volatile("st.global.u32 [%0], %1;" ::"l"(a), "r"(r) : "memory");
          else sink += ld_weak(a);
        } break;
        case 4: sink += ld_weak(p.scr + (r & ((1u << 18) - 1))); break;
        case 5: {  // weak loads of this pair's payload lines (other warps of the SM reading them)
          u32 slot = pair * p.W + (r % p.W);
          sink += ld_weak(p.pay + (u64)slot * 32 * p.ps + l * p.ps);
        } break;
        default: break;
      }
    }
  }
  if (sink == 0x12345678u) p.cnt[C_SINK]++;
}

// Role of a CTA: returns 0 consumer, 1 producer, 2 both (pp=2); pair index in *pair.
__device__ int role(const P& p, u32 b, u32* pair) {
  if (!p.intra) {
    if (b < p.nc) { *pair = b; return 0; }
    *pair = b - p.nc; return 1;
  }
  if (p.pp == 0) { if (b < p.pairs) { *pair = b; return 0; } *pair = b - p.pairs; return 1; }
  if (p.pp == 1) { *pair = b >> 1; return b & 1; }
  *pair = b; return 2;
}

template <int TEST, int FORM, int RD>
__global__ void lit(P p) {
  __shared__ u32 relay[MAXW][32];
  __shared__ volatile u32 done;
  __shared__ u64 cs[C_NCNT];
  const u32 warp = threadIdx.x / 32, l = lane();
  if (threadIdx.x == 0) done = 0;
  if (threadIdx.x < C_NCNT) cs[threadIdx.x] = 0;
  for (u32 i = threadIdx.x; i < MAXW * 32; i += blockDim.x) relay[i / 32][i % 32] = 0;
  __syncthreads();
  u32 pair;
  int r = role(p, blockIdx.x, &pair);
  const u32 cw = (TEST == T_RELAY || FORM == 2) ? 2 : 1;  // warps per consumer instance
  const u32 ncw = r == 1 ? 0 : p.W * cw;                    // consumer warps in this CTA
  const u32 npw = r == 0 ? 0 : p.W;                         // producer warps in this CTA
  u32 seed = p.seed * 7919u + blockIdx.x * 131u + warp * 17u + 1;
  if (threadIdx.x == 0) {
    if (r != 1) p.smid[pair * 2] = smid();
    if (r != 0) p.smid[pair * 2 + 1] = smid();
  }
  u64 c[C_NCNT] = {0};
  if (warp < ncw) {
    u32 w = warp / cw, rw = warp % cw;
    if constexpr (TEST == T_POLL) consumer_poll<FORM, RD>(p, pair * p.W + w, w, rw, seed, c);
    else consumer_relay<FORM, RD>(p, pair * p.W + w, w, rw, seed, c, relay[w]);
  } else if (warp < ncw + npw) {
    producer(p, pair * p.W + (warp - ncw), seed);
  } else {
    stress(p, &done, seed, pair);
  }
  if (warp < ncw + npw) {
    for (int i = 0; i < C_NCNT; i++) {  // warp sum (per-lane counts)
      u64 x = c[i];
      for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(~0u, x, o);
      c[i] = x;
    }
    if (l == 0) for (int i = 0; i < C_NCNT; i++) if (c[i]) atomicAdd(&cs[i], c[i]);
  }
  // role warps arrive at a named barrier among themselves; then stop stress warps
  if (warp < ncw + npw) {
    asm volatile("bar.sync 15, %0;" ::"r"((ncw + npw) * 32) : "memory");
    if (threadIdx.x == 0) done = 1;
  }
  __syncthreads();
  if (threadIdx.x < C_NCNT && cs[threadIdx.x]) atomicAdd(&p.cnt[threadIdx.x], cs[threadIdx.x]);
}

// ------------------------------------------------------------------ host
typedef void (*KFn)(P);
#define K(T, F, R) {#T "/" #F "/" #R, lit<T, F, R>}
static std::map<std::string, KFn> kernels = {
    K(0, 0, 0), K(0, 0, 1), K(0, 0, 2), K(0, 0, 3), K(0, 0, 4),
    K(0, 1, 0), K(0, 1, 1), K(0, 1, 2), K(0, 1, 3),
    K(0, 2, 0), K(0, 2, 1), K(0, 2, 2), K(0, 2, 3),
    K(1, 0, 0), K(1, 0, 1), K(1, 0, 2), K(1, 0, 3), K(1, 0, 4),
    K(1, 1, 0), K(1, 1, 1), K(1, 1, 2), K(1, 1, 3),
};

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s <poll|relay> <form> <rd> [key=val..]\n", argv[0]); return 1; }
  std::map<std::string, long> kv = {{"pairs", 64}, {"iters", 100000}, {"W", 2}, {"nf", 2}, {"ps", 32},
      {"wf", 0}, {"pr", 1}, {"loc", 0}, {"pp", 0}, {"dmask", 255}, {"stress", 0}, {"smode", 0},
      {"seed", 1}, {"sh", 120}, {"cdev", 3}, {"pdev", 3}};
  for (int i = 4; i < argc; i++) {
    const char* e = strchr(argv[i], '=');
    if (!e) { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    std::string k(argv[i], e - argv[i]);
    if (!kv.count(k)) { fprintf(stderr, "unknown key %s\n", k.c_str()); return 1; }
    kv[k] = atol(e + 1);
  }
  int test = std::string(argv[1]) == "relay" ? 1 : 0;
  int form = atoi(argv[2]), rd = atoi(argv[3]);
  std::string kname = std::to_string(test) + "/" + std::to_string(form) + "/" + std::to_string(rd);
  if (!kernels.count(kname)) { fprintf(stderr, "no kernel %s\n", kname.c_str()); return 1; }
  KFn fn = kernels[kname];
  int cdev = kv["cdev"], pdev = kv["pdev"];
  if (excluded_gpu(cdev) || excluded_gpu(pdev)) { fprintf(stderr, "GPU in PTXV_EXCLUDE_GPUS\n"); return 1; }
  bool intra = cdev == pdev;
  P p = {};
  p.pairs = kv["pairs"]; p.iters = kv["iters"]; p.W = kv["W"]; p.nf = kv["nf"]; p.ps = kv["ps"];
  p.wf = kv["wf"]; p.pr = kv["pr"]; p.pp = kv["pp"]; p.dmask = kv["dmask"]; p.S = kv["stress"];
  p.smode = kv["smode"]; p.seed = kv["seed"]; p.intra = intra;
  if (p.W > MAXW || p.nf < 1 || p.nf > 32 || p.ps < 4 || p.ps % 4) { fprintf(stderr, "bad W/nf/ps\n"); return 1; }
  if (!intra) {
    CK(cudaSetDevice(cdev)); cudaDeviceEnablePeerAccess(pdev, 0); cudaGetLastError();
    CK(cudaSetDevice(pdev)); cudaDeviceEnablePeerAccess(cdev, 0); cudaGetLastError();
  }
  const u64 nslot = (u64)p.pairs * p.W;
  auto alloc = [](int dev, u64 bytes) { u32* q; CK(cudaSetDevice(dev)); CK(cudaMalloc(&q, bytes)); CK(cudaMemset(q, 0, bytes)); return q; };
  int paydev = kv["loc"] ? pdev : cdev;
  p.flag = alloc(cdev, nslot * 32 * 4);
  p.pay = alloc(paydev, nslot * 32 * p.ps * 4);
  p.ack = alloc(pdev, nslot * 32 * 4);
  p.abort_ = alloc(cdev, 128);
  u64* cnt = (u64*)alloc(cdev, 16 * 8);
  u32* first = alloc(cdev, 64);
  u32* sm = alloc(cdev, p.pairs * 2 * 4);
  p.cnt = cnt; p.first = first; p.smid = sm;
  u32* scr_c = p.S ? alloc(cdev, SCR_WORDS * 4) : nullptr;
  u32* scr_p = (p.S && !intra) ? alloc(pdev, SCR_WORDS * 4) : scr_c;
  const u32 cw = (test == 1 || form == 2) ? 2 : 1;
  int smem = kv["sh"] * 1024;
  CK(cudaSetDevice(cdev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, cdev));
  auto launch = [&](int dev, u32 grid, u32 nc, u32 np, u32* scr, u32* pscr, cudaStream_t st) {
    CK(cudaSetDevice(dev));
    CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    P q = p; q.nc = nc; q.np = np; q.scr = scr; q.pscr = pscr;
    u32 warps = intra ? (p.pp == 2 ? p.W * cw + p.W : p.W * cw) : (nc ? p.W * cw : p.W);
    warps += p.S;
    fn<<<grid, warps * 32, smem, st>>>(q);
    CK(cudaGetLastError());
  };
  cudaStream_t sc, sp;
  CK(cudaSetDevice(cdev)); CK(cudaStreamCreate(&sc));
  CK(cudaSetDevice(pdev)); CK(cudaStreamCreate(&sp));
  auto t0 = std::chrono::steady_clock::now();
  if (intra) {
    u32 grid = p.pp == 2 ? p.pairs : 2 * p.pairs;
    if ((int)grid > prop.multiProcessorCount) { fprintf(stderr, "grid %u > %d SMs\n", grid, prop.multiProcessorCount); return 1; }
    launch(cdev, grid, 0, 0, scr_c, scr_c, sc);
  } else {
    if ((int)p.pairs > prop.multiProcessorCount) { fprintf(stderr, "pairs > SMs\n"); return 1; }
    launch(pdev, p.pairs, 0, p.pairs, scr_p, scr_c, sp);  // producer: nc = 0
    launch(cdev, p.pairs, p.pairs, 0, scr_c, scr_p, sc);  // consumer
  }
  CK(cudaSetDevice(pdev)); CK(cudaStreamSynchronize(sp));
  CK(cudaSetDevice(cdev)); CK(cudaStreamSynchronize(sc));
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  u64 h[16]; u32 f[16]; u32* hs = (u32*)malloc(p.pairs * 8);
  CK(cudaMemcpy(h, cnt, sizeof h, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(f, first, sizeof f, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hs, sm, p.pairs * 8, cudaMemcpyDeviceToHost));
  u32 samesm = 0;
  for (u32 i = 0; i < p.pairs; i++) samesm += intra && hs[2 * i] == hs[2 * i + 1];
  printf("RESULT test=%s form=%d rd=%d wf=%ld pr=%ld loc=%ld pp=%ld cdev=%d pdev=%d pairs=%u W=%u nf=%u ps=%u "
         "dmask=%u stress=%u smode=%u seed=%u iters=%u | reads=%llu ok=%llu stale=%llu other=%llu timeout=%llu "
         "waits=%llu waited=%llu polls=%llu samesm=%u secs=%.3f",
         test ? "relay" : "poll", form, rd, kv["wf"], kv["pr"], kv["loc"], kv["pp"], cdev, pdev, p.pairs, p.W,
         p.nf, p.ps, p.dmask, p.S, p.smode, p.seed, p.iters, h[C_OK] + h[C_STALE] + h[C_OTHER], h[C_OK],
         h[C_STALE], h[C_OTHER], h[C_TIMEOUT], h[C_ITERS], h[C_WAITED], h[C_POLLS], samesm, secs);
  if (f[0]) printf(" | first k=%u lane=%u slot=%u sm=%u v=%u,%u,%u,%u", f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]);
  printf("\n");
  return h[C_TIMEOUT] ? 3 : 0;
}
