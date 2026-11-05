/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Shared-memory litmus tests on GB300 (sm_103a).
//
// Build:  nvcc -O3 -arch=sm_103a -std=c++17 -o litmus litmus.cu
// Run:    CUDA_VISIBLE_DEVICES=<free gpu> ./litmus <test> <mem> <group> <wf> <rf> [key=val ...]
//
// One "group" (a CTA, or a 2-CTA cluster) runs `iters` iterations. Each iteration:
//   group barrier; the evaluator warp classifies the previous iteration's results;
//   roles are assigned to randomly chosen warps (or lane groups of one warp, same=1);
//   each role thread sleeps a random number of cycles and then runs its litmus thread;
//   warps without a role stress shared memory (random LDS/STS to a scratch region).
// Lane l of every role warp works on instance l (its own set of locations), so every
// iteration runs 32 (16/8 in same-warp mode) independent litmus instances per group.
// Location values: iteration k writes k+1 (the location held k before: old/new).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <chrono>

typedef uint32_t u32;
typedef unsigned long long u64;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2);} } while (0)

enum { G_CTA = 0, G_CLU = 1 };
enum { M_SHV = 0, M_SHW, M_GLW, M_GLS, M_GLV, M_DSM, M_DSR };  // DSM = weak shared::cluster, DSR = relaxed.cluster
enum { T_MP = 0, T_MPS, T_SB, T_LB, T_CORR, T_COWR, T_IRIW, T_TEAR32, T_TEAR64, T_TEAR128, T_MPV, T_MPG, T_MPL1, T_MPGN };

static constexpr int ARENA_WORDS = 8192;  // 32 KB of test locations
static constexpr int SCR_WORDS = 2048;    // 8 KB stress scratch

struct Params {
  u32 iters, seed, dmask, smask, is, lsep, same, nsmode, xis;  // xis: lane stride of location 0 only
  u32* garena;   // global arena (per group ARENA_WORDS)
  u32* res;      // per group [2][NR][32][NRES]
  u64* hist;     // [32]
};

// ---------------------------------------------------------------- memory ops
template <int M> __device__ __forceinline__ u32 LD(u64 a) {
  u32 v;
  if constexpr (M == M_SHV) asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"((u32)a) : "memory");
  else if constexpr (M == M_SHW) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"((u32)a) : "memory");
  else if constexpr (M == M_GLW) asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  else if constexpr (M == M_GLS) asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  else if constexpr (M == M_GLV) asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  else if constexpr (M == M_DSM) asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"((u32)a) : "memory");
  else asm volatile("ld.relaxed.cluster.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"((u32)a) : "memory");
  return v;
}
template <int M> __device__ __forceinline__ void ST(u64 a, u32 v) {
  if constexpr (M == M_SHV) asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"((u32)a), "r"(v) : "memory");
  else if constexpr (M == M_SHW) asm volatile("st.shared.u32 [%0], %1;" ::"r"((u32)a), "r"(v) : "memory");
  else if constexpr (M == M_GLW) asm volatile("st.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
  else if constexpr (M == M_GLS) asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
  else if constexpr (M == M_GLV) asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
  else if constexpr (M == M_DSM) asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"((u32)a), "r"(v) : "memory");
  else asm volatile("st.relaxed.cluster.shared::cluster.u32 [%0], %1;" ::"r"((u32)a), "r"(v) : "memory");
}
template <int M, int G> __device__ __forceinline__ u32 LDACQ(u64 a) {
  u32 v;
  if constexpr (M == M_SHV || M == M_SHW) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(v) : "r"((u32)a) : "memory");
  else if constexpr (M == M_DSM || M == M_DSR) asm volatile("ld.acquire.cluster.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"((u32)a) : "memory");
  else if constexpr (G == G_CTA) asm volatile("ld.acquire.cta.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  else asm volatile("ld.acquire.cluster.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
template <int M, int G> __device__ __forceinline__ void STREL(u64 a, u32 v) {
  if constexpr (M == M_SHV || M == M_SHW) asm volatile("st.release.cta.shared.u32 [%0], %1;" ::"r"((u32)a), "r"(v) : "memory");
  else if constexpr (M == M_DSM || M == M_DSR) asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;" ::"r"((u32)a), "r"(v) : "memory");
  else if constexpr (G == G_CTA) asm volatile("st.release.cta.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
  else asm volatile("st.release.cluster.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
}
// F: 1 = fence.acq_rel, 3 = fence.sc (others: none)
template <int F, int G> __device__ __forceinline__ void FENCE() {
  if constexpr (F == 1 && G == G_CTA) asm volatile("fence.acq_rel.cta;" ::: "memory");
  else if constexpr (F == 1) asm volatile("fence.acq_rel.cluster;" ::: "memory");
  else if constexpr (F == 3 && G == G_CTA) asm volatile("fence.sc.cta;" ::: "memory");
  else if constexpr (F == 3) asm volatile("fence.sc.cluster;" ::: "memory");
}
// vector ops for tearing tests (shared / dsmem / global volatile)
template <int M> __device__ __forceinline__ void ST2(u64 a, u32 x, u32 y) {
  if constexpr (M == M_SHV) asm volatile("st.volatile.shared.v2.u32 [%0], {%1,%2};" ::"r"((u32)a), "r"(x), "r"(y) : "memory");
  else if constexpr (M == M_DSM) asm volatile("st.shared::cluster.v2.u32 [%0], {%1,%2};" ::"r"((u32)a), "r"(x), "r"(y) : "memory");
  else asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};" ::"l"(a), "r"(x), "r"(y) : "memory");
}
template <int M> __device__ __forceinline__ void LD2(u64 a, u32& x, u32& y) {
  if constexpr (M == M_SHV) asm volatile("ld.volatile.shared.v2.u32 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "r"((u32)a) : "memory");
  else if constexpr (M == M_DSM) asm volatile("ld.shared::cluster.v2.u32 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "r"((u32)a) : "memory");
  else asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "l"(a) : "memory");
}
template <int M> __device__ __forceinline__ void ST4(u64 a, u32 x, u32 y, u32 z, u32 w) {
  if constexpr (M == M_SHV) asm volatile("st.volatile.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"((u32)a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
  else if constexpr (M == M_DSM) asm volatile("st.shared::cluster.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"((u32)a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
  else asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
template <int M> __device__ __forceinline__ void LD4(u64 a, u32& x, u32& y, u32& z, u32& w) {
  if constexpr (M == M_SHV) asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"((u32)a) : "memory");
  else if constexpr (M == M_DSM) asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"((u32)a) : "memory");
  else asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(a) : "memory");
}

__device__ __forceinline__ int cls(u32 v, u32 k) { return v == k + 1 ? 1 : (v == k ? 0 : 2); }

// ---------------------------------------------------------------- tests
// L[j] = address of location j for this thread's instance; o[] = results.
template <int T> struct Test;

template <> struct Test<T_MP> {  // W: x=1; [wf]; y=1      R: r1=y; [rf]; r2=x
  enum { NR = 2, NL = 2, NRES = 2, NOUT = 9 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    u32 v = k + 1;
    if (r == 0) {
      ST<M>(L[0], v);
      FENCE<WF, G>();
      if constexpr (WF == 2) STREL<M, G>(L[1], v); else ST<M>(L[1], v);
    } else {
      u32 a; if constexpr (RF == 2) a = LDACQ<M, G>(L[1]); else a = LD<M>(L[1]);
      FENCE<RF, G>();
      u32 b = LD<M>(L[0]);
      o[0] = a; o[1] = b;
    }
  }
  __device__ static int classify(u32 k, const u32* R) { return cls(R[2], k) * 3 + cls(R[3], k); }
};
template <> struct Test<T_MPS> {  // as MP, but reader spins (<=256) until it sees y==1
  enum { NR = 2, NL = 2, NRES = 2, NOUT = 9 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    u32 v = k + 1;
    if (r == 0) {
      ST<M>(L[0], v);
      FENCE<WF, G>();
      if constexpr (WF == 2) STREL<M, G>(L[1], v); else ST<M>(L[1], v);
    } else {
      u32 a = 0;
      for (int i = 0; i < 256; i++) {
        if constexpr (RF == 2) a = LDACQ<M, G>(L[1]); else a = LD<M>(L[1]);
        if (a == v) break;
      }
      FENCE<RF, G>();
      u32 b = LD<M>(L[0]);
      o[0] = a; o[1] = b;
    }
  }
  __device__ static int classify(u32 k, const u32* R) { return cls(R[2], k) * 3 + cls(R[3], k); }
};
template <> struct Test<T_SB> {  // T0: x=1; [f]; r0=y     T1: y=1; [f]; r1=x
  enum { NR = 2, NL = 2, NRES = 1, NOUT = 9 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    u32 v = k + 1;
    ST<M>(L[r], v);
    FENCE<WF, G>();
    o[0] = LD<M>(L[r ^ 1]);
  }
  __device__ static int classify(u32 k, const u32* R) { return cls(R[0], k) * 3 + cls(R[1], k); }
};
template <> struct Test<T_LB> {  // T0: r0=x; [f]; y=1     T1: r1=y; [f]; x=1
  enum { NR = 2, NL = 2, NRES = 1, NOUT = 9 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    u32 v = k + 1;
    u32 a = LD<M>(L[r]);
    FENCE<WF, G>();
    ST<M>(L[r ^ 1], v);
    o[0] = a;
  }
  __device__ static int classify(u32 k, const u32* R) { return cls(R[0], k) * 3 + cls(R[1], k); }
};
template <> struct Test<T_CORR> {  // W: x=1;x=2;x=3;x=4    R: 4 reads of x, must be non-decreasing
  enum { NR = 2, NL = 1, NRES = 4, NOUT = 5 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    u32 b = 4 * k;
    if (r == 0) { ST<M>(L[0], b + 1); ST<M>(L[0], b + 2); ST<M>(L[0], b + 3); ST<M>(L[0], b + 4); }
    else { o[0] = LD<M>(L[0]); o[1] = LD<M>(L[0]); o[2] = LD<M>(L[0]); o[3] = LD<M>(L[0]); }
  }
  // 0 all old, 1 all final, 2 mixed non-decreasing, 3 DECREASE (violation), 4 out of range
  __device__ static int classify(u32 k, const u32* R) {
    const u32* a = R + 4; u32 b = 4 * k;
    for (int i = 0; i < 4; i++) if (a[i] < b || a[i] > b + 4) return 4;
    for (int i = 0; i < 3; i++) if (a[i + 1] < a[i]) return 3;
    if (a[0] == b && a[3] == b) return 0;
    if (a[0] == b + 4) return 1;
    return 2;
  }
};
template <> struct Test<T_COWR> {  // Ti: x=a_i; r=x; x=b_i; r'=x  (own-write coherence with a racing writer)
  enum { NR = 2, NL = 1, NRES = 2, NOUT = 3 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    u32 b = 4 * k + 1 + 2 * r;
    ST<M>(L[0], b); o[0] = LD<M>(L[0]);
    ST<M>(L[0], b + 1); o[1] = LD<M>(L[0]);
  }
  // 0 each saw only own writes, 1 saw other's writes (ok), 2 VIOLATION (saw own older / pre-iteration value)
  __device__ static int classify(u32 k, const u32* R) {
    int other = 0;
    for (int r = 0; r < 2; r++) {
      u32 a1 = 4 * k + 1 + 2 * r, a2 = a1 + 1, o1 = 4 * k + 1 + 2 * (r ^ 1), o2 = o1 + 1;
      u32 x = R[2 * r], y = R[2 * r + 1];
      if (x == a1) {} else if (x == o1 || x == o2) other = 1; else return 2;
      if (y == a2) {} else if (y == o1 || y == o2) other = 1; else return 2;
    }
    return other;
  }
};
template <> struct Test<T_IRIW> {  // W0: x=1  W1: y=1  R0: a=x;[f];b=y  R1: c=y;[f];d=x
  enum { NR = 4, NL = 2, NRES = 2, NOUT = 17 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    u32 v = k + 1;
    if (r < 2) ST<M>(L[r], v);
    else { int f = r - 2; o[0] = LD<M>(L[f]); FENCE<RF, G>(); o[1] = LD<M>(L[f ^ 1]); }
  }
  __device__ static int classify(u32 k, const u32* R) {
    int a = cls(R[4], k), b = cls(R[5], k), c = cls(R[6], k), d = cls(R[7], k);
    if (a > 1 || b > 1 || c > 1 || d > 1) return 16;
    return a | (b << 1) | (c << 2) | (d << 3);
  }
};
template <int W> struct Tear {  // W: 8 stores of a self-consistent pattern; R: 8 loads, check each
  enum { NR = 2, NL = 1, NRES = 1, NOUT = 3 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    if (r == 0) {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        u32 v = k * 8 + j + 1;
        if constexpr (W == 32) ST<M>(L[0], (v & 0xff) * 0x01010101u);
        else if constexpr (W == 64) ST2<M>(L[0], v, ~v);
        else ST4<M>(L[0], v, ~v, v ^ 0x5a5a5a5au, v + 0x10000000u);
      }
    } else {
      int torn = 0, mid = 0;
      u32 vo = k * 8, vf = k * 8 + 8;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        u32 v;
        if constexpr (W == 32) {
          u32 x = LD<M>(L[0]); u32 lo = x & 0xff;
          torn |= x != lo * 0x01010101u; v = lo; vo &= 0xff; vf &= 0xff;
        } else if constexpr (W == 64) {
          u32 x, y; LD2<M>(L[0], x, y); torn |= y != ~x; v = x;
        } else {
          u32 x, y, z, w; LD4<M>(L[0], x, y, z, w);
          torn |= (y != ~x) | (z != (x ^ 0x5a5a5a5au)) | (w != x + 0x10000000u); v = x;
        }
        mid |= (v != vo && v != vf);
      }
      o[0] = torn ? 2 : (mid ? 1 : 0);
    }
  }
  __device__ static int classify(u32 k, const u32* R) { return (int)R[1]; }
};
template <> struct Test<T_TEAR32> : Tear<32> {};
template <> struct Test<T_TEAR64> : Tear<64> {};
template <> struct Test<T_TEAR128> : Tear<128> {};

// MPV: the shape ptxas makes of a merged release: ONE STS.64 writes {data, flag} (wf=0: data at +0,
// flag at +4; wf=1: flag at +0, data at +4). Reader polls the flag word (LDS.32, <=256 tries), then
// loads the data word separately (LDS.32). rf=2: the flag poll is ld.acquire.cta.
template <> struct Test<T_MPV> {
  enum { NR = 2, NL = 1, NRES = 2, NOUT = 9 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    u32 v = k + 1;
    u64 fa = L[0] + (WF == 0 ? 4 : 0), da = L[0] + (WF == 0 ? 0 : 4);
    if (r == 0) {
      ST2<M>(L[0], v, v);
    } else {
      u32 a = 0;
      for (int i = 0; i < 256; i++) {
        if constexpr (RF == 2) a = LDACQ<M, G>(fa); else a = LD<M>(fa);
        if (a == v) break;
      }
      u32 b = LD<M>(da);
      o[0] = a; o[1] = b;
    }
  }
  __device__ static int classify(u32 k, const u32* R) { return cls(R[2], k) * 3 + cls(R[3], k); }
};
// MPG: a shared-memory flag guarding a GLOBAL payload, one CTA. Writer: STG data (weak);
// flag store = plain STS (wf=0) or st.release.cta.shared (wf=2: MEMBAR.ALL.CTA; STS).
// Reader: rf&1 = first prime L1 with a weak load of the payload (rf&4: of its neighbour word); poll the flag (<=256) with plain
// LDS (rf&2 == 0) or ld.acquire.cta.shared (rf&2: LDS + scoreboard wait only); then weak LDG payload.
template <> struct Test<T_MPG> {
  enum { NR = 2, NL = 2, NRES = 2, NOUT = 9 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    u32 v = k + 1;
    if (r == 0) {
      // wf=4: payload store is st.volatile.global (STG.E.STRONG.SYS) so ptxas keeps it before the
      // flag STS; with a weak payload store (wf=0) ptxas emits STS flag BEFORE STG payload.
      if constexpr (WF == 4) ST<M_GLV>(L[0], v);
      else ST<M_GLW>(L[0], v);
      if constexpr (WF == 2) STREL<M_SHV, G_CTA>(L[1], v); else ST<M_SHV>(L[1], v);
    } else {
      u32 p = 0;
      if constexpr (RF & 1) p = LD<M_GLW>(L[0] ^ ((RF & 4) ? 4 : 0));  // rf&4: prime via the neighbour word
      u32 a = 0;
      for (int i = 0; i < 256; i++) {
        if constexpr (RF & 2) a = LDACQ<M_SHV, G_CTA>(L[1]); else a = LD<M_SHV>(L[1]);
        if (a == v) break;
      }
      u32 b = LD<M_GLW>(L[0]);
      o[0] = a; o[1] = b + (p == 0xdeadbeefu);
    }
  }
  __device__ static int classify(u32 k, const u32* R) { return cls(R[2], k) * 3 + cls(R[3], k); }
};
// MPGN: MPG without the poll loop: reader issues the flag LDS and the payload LDG back to back
// (rf&2 == 0: no scoreboard wait between them; rf&2: ld.acquire.cta = LDS + wait). rf&1: prime L1
// (neighbour word of the same line).
template <> struct Test<T_MPGN> {
  enum { NR = 2, NL = 2, NRES = 2, NOUT = 9 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    u32 v = k + 1;
    if (r == 0) {
      ST<M_GLW>(L[0], v);
      if constexpr (WF == 2) STREL<M_SHV, G_CTA>(L[1], v); else ST<M_SHV>(L[1], v);
    } else {
      u32 p = 0;
      // prime the payload's L1 line through the NEIGHBOUR word (same 128B line): a load of the
      // payload word itself is merged by ptxas with the later payload load (one LDG hoisted above
      // the flag LDS, even across ld.acquire, since the racy prime makes the PTX program racy).
      if constexpr (RF & 1) p = LD<M_GLW>(L[0] ^ 4);
      u32 a;
      if constexpr (RF & 2) a = LDACQ<M_SHV, G_CTA>(L[1]); else a = LD<M_SHV>(L[1]);
      u32 b;
      // rf&4: payload load is ld.relaxed.cta.global (LDG.E.STRONG.SM), which ptxas does not hoist
      // above the flag LDS; a weak payload load is hoisted above it when there is no acquire.
      if constexpr (RF & 4) asm volatile("ld.relaxed.cta.global.u32 %0, [%1];" : "=r"(b) : "l"(L[0]) : "memory");
      else b = LD<M_GLW>(L[0]);
      o[0] = a; o[1] = b + (p == 0xdeadbeefu);
    }
  }
  __device__ static int classify(u32 k, const u32* R) { return cls(R[2], k) * 3 + cls(R[3], k); }
};
// MPL1 (positive control for the harness): global x,y. Writer: x=1; st.release y=1 (correct).
// Reader: weak load of x (primes L1), poll y with a STRONG relaxed load (no acquire), weak load of x.
// Without an acquire the second load may hit the stale L1 line: r1=1, r2=0 is allowed by PTX.
template <> struct Test<T_MPL1> {
  enum { NR = 2, NL = 2, NRES = 2, NOUT = 9 };
  template <int M, int G, int WF, int RF> __device__ static void run(int r, u32 k, const u64* L, u32* o) {
    u32 v = k + 1;
    if (r == 0) {
      ST<M_GLW>(L[0], v);
      STREL<M_GLW, G>(L[1], v);
    } else {
      u32 p = LD<M_GLW>(L[0]);
      u32 a = 0;
      for (int i = 0; i < 256; i++) { a = LD<M_GLS>(L[1]); if (a == v) break; }
      if constexpr (RF == 2) asm volatile("fence.acq_rel.gpu;" ::: "memory");
      u32 b = LD<M_GLW>(L[0]);
      o[0] = a; o[1] = b + (p == 0xdeadbeefu);
    }
  }
  __device__ static int classify(u32 k, const u32* R) { return cls(R[2], k) * 3 + cls(R[3], k); }
};

// ---------------------------------------------------------------- harness
__device__ __forceinline__ u32 xs(u32& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
__device__ __forceinline__ u32 hash(u32 x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x | 1; }

__device__ __forceinline__ void delay(u32 r, u32 dmask) {
  u32 d = r & dmask;
  if (d == 0) return;
  if (((r >> 20) & 15) == 0) { __nanosleep(d); return; }
  u32 t0 = (u32)clock();
  while ((u32)clock() - t0 < d) {}
}
template <int G> __device__ __forceinline__ void gbar() {
  if constexpr (G == G_CTA) __syncthreads();
  else asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

template <int T, int M, int G, int WF, int RF>
__global__ void __launch_bounds__(1024) litmus(Params p) {
  using TT = Test<T>;
  constexpr int NR = TT::NR, NRES = TT::NRES;
  extern __shared__ __align__(16) u32 sm[];
  __shared__ u32 hist[32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
  u32 rank = 0;
  if constexpr (G == G_CLU) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const u32 group = G == G_CLU ? blockIdx.x / 2 : blockIdx.x;

  for (int i = tid; i < ARENA_WORDS + SCR_WORDS; i += blockDim.x) sm[i] = 0;
  if (tid < 32) hist[tid] = 0;

  const int ninst = p.same ? 32 / (NR == 2 ? 2 : 4) : 32;
  // addresses of this thread's instance (computed per role assignment below)
  const u32 sbase = (u32)__cvta_generic_to_shared(sm);
  auto loc_addr = [&](int j, int inst) -> u64 {
    u32 off = (u32)(j * p.lsep + inst * (j == 0 ? p.xis : p.is)) % ARENA_WORDS;
    if constexpr (T == T_MPG || T == T_MPGN) { if (j == 1) return sbase + off * 4; return (u64)(p.garena + (size_t)group * ARENA_WORDS + off); }
    else if constexpr (M == M_SHV || M == M_SHW) return sbase + off * 4;
    else if constexpr (M == M_DSM || M == M_DSR) {
      u32 a = sbase + off * 4, d, tr = (G == G_CLU) ? (u32)(j & 1) : 0;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(tr));
      return d;
    } else return (u64)(p.garena + (size_t)group * ARENA_WORDS + off);
  };
  u32* res = p.res + (size_t)group * 2 * NR * 32 * NRES;
  u32 trng = hash(p.seed ^ hash(blockIdx.x * 1024 + tid + 7));
  u32 grng = hash(p.seed ^ hash(group + 0x9e3779b9u));
  gbar<G>();

  auto evaluate = [&](u32 k) {
    if (rank == 0 && warp == 0 && lane < ninst) {
      volatile u32* rb = res + (size_t)(k & 1) * NR * 32 * NRES;
      u32 R[NR * NRES];
#pragma unroll
      for (int r = 0; r < NR; r++)
#pragma unroll
        for (int j = 0; j < NRES; j++) R[r * NRES + j] = rb[(r * 32 + lane) * NRES + j];
      atomicAdd(&hist[TT::classify(k, R)], 1u);
    }
  };

  for (u32 k = 0; k < p.iters; k++) {
    gbar<G>();
    if (k > 1) evaluate(k - 1);  // iteration 0 is skipped (zero-initialised memory is not a valid tearing pattern)
    u32 g = xs(grng);
    int role = -1, inst = lane;
    if (p.same) {
      int w = (g >> 8) % nw;
      if (rank == 0 && warp == w) { int gs = 32 / (NR == 2 ? 2 : 4); role = lane / gs; inst = lane % gs; if (role >= NR) role = -1; }
    } else {
      int base = (g >> 8) % nw, s = (int)((g >> 20) % nw) | 1;
      // nsmode=1: fixed warps (roles r -> warp r)
      if (p.nsmode) { base = 0; s = 1; }
#pragma unroll
      for (int r = 0; r < NR; r++) {
        u32 rr = G == G_CLU ? (u32)(r & 1) : 0;
        if (rank == rr && warp == (base + r * s) % nw) role = r;
      }
    }
    u32 t = xs(trng);
    if (role >= 0) {
      u64 L[TT::NL];
#pragma unroll
      for (int j = 0; j < TT::NL; j++) L[j] = loc_addr(j, inst);
      u32 o[NRES];
#pragma unroll
      for (int j = 0; j < NRES; j++) o[j] = 0;
      // same-warp mode: per-lane-group delay (diverges the warp); diff-warp: per-warp delay
      u32 dr = __shfl_sync(__activemask(), t, p.same ? (lane & ~((32 / (NR == 2 ? 2 : 4)) - 1)) : 0);
      delay(dr, p.dmask);
      TT::template run<M, G, WF, RF>(role, k, L, o);
      u32* rb = res + (size_t)(k & 1) * NR * 32 * NRES;
#pragma unroll
      for (int j = 0; j < NRES; j++) rb[(role * 32 + inst) * NRES + j] = o[j];
    } else if (p.smask) {
      u32 n = (__shfl_sync(0xffffffffu, t, 0) >> 4) & p.smask;
      u32 acc = 0, a0 = (t >> 8);
      for (u32 i = 0; i < n; i++) {
        u32 a = sbase + 4 * ((ARENA_WORDS + ((a0 + i * 97 + lane) & (SCR_WORDS - 1))));
        ST<M_SHV>(a, i + acc);
        acc += LD<M_SHV>(a);
      }
      if (acc == 0xdeadbeef) p.hist[31] = acc;
    }
  }
  gbar<G>();
  evaluate(p.iters - 1);
  __syncthreads();
  if (tid < 32 && hist[tid]) atomicAdd(&p.hist[tid], (u64)hist[tid]);
}

// ---------------------------------------------------------------- host
typedef void (*KFn)(Params);
struct Cfg { int t, m, g, wf, rf; KFn fn; };
#define K(T, M, G, WF, RF) Cfg{T, M, G, WF, RF, litmus<T, M, G, WF, RF>}
static std::vector<Cfg> configs() {
  std::vector<Cfg> v = {
    // shared, volatile (LDS/STS), one CTA
    K(T_MP, M_SHV, G_CTA, 0, 0), K(T_MP, M_SHV, G_CTA, 1, 0), K(T_MP, M_SHV, G_CTA, 2, 0), K(T_MP, M_SHV, G_CTA, 3, 0),
    K(T_MP, M_SHV, G_CTA, 0, 1), K(T_MP, M_SHV, G_CTA, 0, 2), K(T_MP, M_SHV, G_CTA, 1, 1), K(T_MP, M_SHV, G_CTA, 2, 2),
    K(T_MPS, M_SHV, G_CTA, 0, 0), K(T_MPS, M_SHV, G_CTA, 2, 2),
    K(T_SB, M_SHV, G_CTA, 0, 0), K(T_SB, M_SHV, G_CTA, 1, 0), K(T_SB, M_SHV, G_CTA, 3, 0),
    K(T_LB, M_SHV, G_CTA, 0, 0), K(T_CORR, M_SHV, G_CTA, 0, 0), K(T_COWR, M_SHV, G_CTA, 0, 0),
    K(T_IRIW, M_SHV, G_CTA, 0, 0), K(T_IRIW, M_SHV, G_CTA, 0, 3),
    K(T_TEAR32, M_SHV, G_CTA, 0, 0), K(T_TEAR64, M_SHV, G_CTA, 0, 0), K(T_TEAR128, M_SHV, G_CTA, 0, 0),
    // shared, weak ld.shared/st.shared
    K(T_MP, M_SHW, G_CTA, 0, 0), K(T_MPS, M_SHW, G_CTA, 0, 0), K(T_SB, M_SHW, G_CTA, 0, 0), K(T_LB, M_SHW, G_CTA, 0, 0),
    K(T_CORR, M_SHW, G_CTA, 0, 0), K(T_IRIW, M_SHW, G_CTA, 0, 0),
    // global, one CTA
    K(T_MP, M_GLW, G_CTA, 0, 0), K(T_MPS, M_GLW, G_CTA, 0, 0), K(T_SB, M_GLW, G_CTA, 0, 0), K(T_LB, M_GLW, G_CTA, 0, 0),
    K(T_CORR, M_GLW, G_CTA, 0, 0), K(T_IRIW, M_GLW, G_CTA, 0, 0),
    K(T_MP, M_GLS, G_CTA, 0, 0), K(T_MPS, M_GLS, G_CTA, 0, 0), K(T_SB, M_GLS, G_CTA, 0, 0), K(T_LB, M_GLS, G_CTA, 0, 0),
    K(T_CORR, M_GLS, G_CTA, 0, 0), K(T_IRIW, M_GLS, G_CTA, 0, 0),
    // 2-CTA cluster: DSMEM
    K(T_MP, M_DSM, G_CLU, 0, 0), K(T_MP, M_DSM, G_CLU, 2, 2), K(T_MPS, M_DSM, G_CLU, 0, 0), K(T_MPS, M_DSM, G_CLU, 2, 2),
    K(T_SB, M_DSM, G_CLU, 0, 0), K(T_SB, M_DSM, G_CLU, 3, 0), K(T_LB, M_DSM, G_CLU, 0, 0), K(T_CORR, M_DSM, G_CLU, 0, 0),
    K(T_COWR, M_DSM, G_CLU, 0, 0), K(T_IRIW, M_DSM, G_CLU, 0, 0), K(T_IRIW, M_DSM, G_CLU, 0, 3),
    K(T_TEAR64, M_DSM, G_CLU, 0, 0), K(T_TEAR128, M_DSM, G_CLU, 0, 0),
    // merged STS.64 {data,flag}; shared flag guarding global payload; L1-staleness positive control
    K(T_MPV, M_SHV, G_CTA, 0, 0), K(T_MPV, M_SHV, G_CTA, 1, 0), K(T_MPV, M_SHV, G_CTA, 0, 2), K(T_MPV, M_SHV, G_CTA, 1, 2),
    K(T_MPG, M_SHV, G_CTA, 0, 0), K(T_MPG, M_SHV, G_CTA, 0, 1), K(T_MPG, M_SHV, G_CTA, 2, 2), K(T_MPG, M_SHV, G_CTA, 2, 3),
    K(T_MPG, M_SHV, G_CTA, 0, 2), K(T_MPG, M_SHV, G_CTA, 2, 1), K(T_MPG, M_SHV, G_CTA, 0, 3),
    K(T_MPG, M_SHV, G_CTA, 0, 5), K(T_MPG, M_SHV, G_CTA, 0, 7), K(T_MPG, M_SHV, G_CTA, 2, 7),
    K(T_MPG, M_SHV, G_CTA, 4, 1), K(T_MPG, M_SHV, G_CTA, 4, 3), K(T_MPG, M_SHV, G_CTA, 4, 7),
    K(T_MPGN, M_SHV, G_CTA, 2, 1), K(T_MPGN, M_SHV, G_CTA, 2, 3), K(T_MPGN, M_SHV, G_CTA, 0, 1), K(T_MPGN, M_SHV, G_CTA, 0, 3),
    K(T_MPGN, M_SHV, G_CTA, 2, 7), K(T_MPGN, M_SHV, G_CTA, 0, 7),
    K(T_MPL1, M_GLW, G_CTA, 0, 0), K(T_MPL1, M_GLW, G_CLU, 0, 0), K(T_MPL1, M_GLW, G_CLU, 0, 2),
    // 2-CTA cluster: DSMEM relaxed.cluster (ptxas folds weak repeated accesses, so coherence tests need this)
    K(T_MP, M_DSR, G_CLU, 0, 0), K(T_MPS, M_DSR, G_CLU, 0, 0), K(T_SB, M_DSR, G_CLU, 0, 0), K(T_LB, M_DSR, G_CLU, 0, 0),
    K(T_CORR, M_DSR, G_CLU, 0, 0), K(T_COWR, M_DSR, G_CLU, 0, 0), K(T_IRIW, M_DSR, G_CLU, 0, 0),
    // 2-CTA cluster: global
    K(T_MP, M_GLW, G_CLU, 0, 0), K(T_MPS, M_GLW, G_CLU, 0, 0), K(T_MPS, M_GLW, G_CLU, 2, 2), K(T_SB, M_GLW, G_CLU, 0, 0),
    K(T_LB, M_GLW, G_CLU, 0, 0), K(T_CORR, M_GLW, G_CLU, 0, 0), K(T_IRIW, M_GLW, G_CLU, 0, 0),
    K(T_MP, M_GLS, G_CLU, 0, 0), K(T_MPS, M_GLS, G_CLU, 0, 0), K(T_SB, M_GLS, G_CLU, 0, 0), K(T_SB, M_GLS, G_CLU, 3, 0),
    K(T_LB, M_GLS, G_CLU, 0, 0), K(T_CORR, M_GLS, G_CLU, 0, 0), K(T_IRIW, M_GLS, G_CLU, 0, 0), K(T_IRIW, M_GLS, G_CLU, 0, 3),
  };
  return v;
}
static const char* TN[] = {"MP", "MPS", "SB", "LB", "CORR", "COWR", "IRIW", "TEAR32", "TEAR64", "TEAR128", "MPV", "MPG", "MPL1", "MPGN"};
static const char* MN[] = {"shv", "shw", "glw", "gls", "glv", "dsm", "dsr"};
static const char* GN[] = {"cta", "clu"};
static int NRof(int t) { return t == T_IRIW ? 4 : 2; }
static int NRESof(int t) { return t == T_CORR ? 4 : (t == T_SB || t == T_LB || (t >= T_TEAR32 && t <= T_TEAR128)) ? 1 : 2; }
static int NOUTof(int t) {
  switch (t) { case T_CORR: return 5; case T_COWR: return 3; case T_IRIW: return 17; default: return (t >= T_TEAR32 && t <= T_TEAR128) ? 3 : 9; }
}
static std::string oname(int t, int o) {
  const char* on[3] = {"0", "1", "?"};
  char b[64];
  if (t <= T_LB || t >= T_MPV) { snprintf(b, sizeof b, "%s%s", on[o / 3], on[o % 3]); return b; }
  if (t == T_CORR) { const char* n[] = {"allold", "allnew", "mixed", "DECREASE", "range"}; return n[o]; }
  if (t == T_COWR) { const char* n[] = {"own", "saw_other", "VIOL"}; return n[o]; }
  if (t == T_IRIW) { if (o == 16) return "?"; snprintf(b, sizeof b, "%d%d%d%d", o & 1, (o >> 1) & 1, (o >> 2) & 1, (o >> 3) & 1); return b; }
  const char* n[] = {"clean", "overlap", "TORN"}; return n[o];
}
static bool forbidden(int t, int o) {
  switch (t) {
    case T_MP: case T_MPS: case T_MPV: case T_MPG: case T_MPL1: case T_MPGN: return o == 1 * 3 + 0;  // r1=1, r2=0
    case T_SB: return o == 0;                     // 0,0
    case T_LB: return o == 4;                     // 1,1
    case T_CORR: return o >= 3;
    case T_COWR: return o == 2;
    case T_IRIW: return o == 5 || o == 16;        // a=1 b=0 c=1 d=0
    default: return o == 2;
  }
}

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: %s test mem group wf rf [iters= launches= nw= dmask= smask= is= lsep= same= seed= blocks= fixed=]\n", argv[0]); return 1; }
  int t = -1, m = -1, g = -1;
  for (int i = 0; i < 14; i++) if (!strcmp(argv[1], TN[i])) t = i;
  for (int i = 0; i < 7; i++) if (!strcmp(argv[2], MN[i])) m = i;
  for (int i = 0; i < 2; i++) if (!strcmp(argv[3], GN[i])) g = i;
  int wf = atoi(argv[4]), rf = atoi(argv[5]);
  std::map<std::string, long> kv = {{"iters", 200000}, {"launches", 1}, {"nw", 16}, {"dmask", 63}, {"smask", 63}, {"is", 1},
                                     {"lsep", 0}, {"same", 0}, {"seed", 1}, {"blocks", 0}, {"fixed", 0}, {"xis", 0}};
  for (int i = 6; i < argc; i++) { char* e = strchr(argv[i], '='); if (!e) continue; *e = 0; kv[argv[i]] = atol(e + 1); }
  KFn fn = nullptr;
  for (auto& c : configs()) if (c.t == t && c.m == m && c.g == g && c.wf == wf && c.rf == rf) fn = c.fn;
  if (!fn) { fprintf(stderr, "config not instantiated\n"); return 1; }
  int dev = 0, nsm = 0; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  int nw = kv["nw"]; int blocks = kv["blocks"] ? kv["blocks"] : nsm * 2;
  if (g == G_CLU) blocks &= ~1;
  int NR = NRof(t), NRES = NRESof(t), groups = g == G_CLU ? blocks / 2 : blocks;
  long lsep = kv["lsep"] ? kv["lsep"] : 32 * kv["is"] + 1;
  size_t smem = (ARENA_WORDS + SCR_WORDS) * 4;
  CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  if (g == G_CLU) CK(cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  u32 *garena, *res; u64* hist;
  CK(cudaMalloc(&garena, (size_t)groups * ARENA_WORDS * 4));
  CK(cudaMalloc(&res, (size_t)groups * 2 * NR * 32 * NRES * 4));
  CK(cudaMalloc(&hist, 32 * 8));
  std::vector<u64> tot(32, 0);
  auto t0 = std::chrono::steady_clock::now();
  for (int L = 0; L < kv["launches"]; L++) {
    CK(cudaMemset(garena, 0, (size_t)groups * ARENA_WORDS * 4));
    CK(cudaMemset(hist, 0, 32 * 8));
    Params p{(u32)kv["iters"], (u32)(kv["seed"] * 7919 + L * 104729), (u32)kv["dmask"], (u32)kv["smask"], (u32)kv["is"],
             (u32)lsep, (u32)kv["same"], (u32)kv["fixed"], (u32)(kv["xis"] ? kv["xis"] : kv["is"]), garena, res, hist};
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(blocks); lc.blockDim = dim3(nw * 32); lc.dynamicSmemBytes = smem;
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = g == G_CLU ? 2 : 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    lc.attrs = at; lc.numAttrs = 1;
    CK(cudaLaunchKernelEx(&lc, fn, p));
    CK(cudaDeviceSynchronize());
    std::vector<u64> h(32); CK(cudaMemcpy(h.data(), hist, 32 * 8, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 32; i++) tot[i] += h[i];
  }
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  u64 n = 0, bad = 0;
  for (int o = 0; o < NOUTof(t); o++) { n += tot[o]; if (forbidden(t, o)) bad += tot[o]; }
  printf("%s %s %s wf=%d rf=%d nw=%ld dmask=%ld smask=%ld is=%ld xis=%ld lsep=%ld same=%ld fixed=%ld blocks=%d | n=%llu FORBIDDEN=%llu | ",
         TN[t], MN[m], GN[g], wf, rf, kv["nw"], kv["dmask"], kv["smask"], kv["is"], kv["xis"], lsep, kv["same"], kv["fixed"], blocks, n, bad);
  for (int o = 0; o < NOUTof(t); o++) if (tot[o]) printf("%s%s:%llu ", forbidden(t, o) ? "!" : "", oname(t, o).c_str(), tot[o]);
  printf("| %.1fs\n", secs);
  return 0;
}
