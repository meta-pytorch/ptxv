/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// E-CPASYNC (SASS divergent waits) probe: do the LDGDEPBARs of a warp's two paths count on
// one scoreboard, so that one path's DEPBAR.LE SB0, 1 waits for the other path's earlier
// commit?
//
// One warp per CTA. Every lane scrubs its 16-byte shared slot (0xdeadbeef) and the warp
// syncs. The even lanes copy 16 bytes of a cold line (an L2 flush runs before every launch;
// each lane's source is its own 128-byte line) with cp.async.cg and commit (the warp's
// commit 1); the warp syncs; the odd lanes copy their own 16 bytes and commit (commit 2),
// then per mode:
//   0 WAIT1:  cp.async.wait_group 1 (DEPBAR.LE SB0, 1). One warp scoreboard: the even group
//             is the older one, so it has landed and the odd lane reads its even neighbour's
//             data. Per-path counting (PTX's per-thread groups): nothing to wait for, so the
//             read can see the scrub value while the even copy is in flight.
//   1 NOWAIT: no wait (control: the even copy must still be in flight, scrub values seen).
//   2 WAIT0:  cp.async.wait_group 0 (control: both groups landed).
//   3 WAIT1W: WAIT1 with the odd lanes' sources made L2-resident after the flush (a warm-up
//             kernel reads them), so the odd group lands before the even one. A scoreboard
//             that retired groups as they complete (not in commit order) would let the odd
//             DEPBAR.LE SB0, 1 pass once the odd copy landed and read the scrub value.
//   4 ORDERW: as 3 without the wait: the odd lane spins on its own slot until its copy landed,
//             then reads the neighbour's (control for 3: how often the odd group lands first).
//   5 ORDER:  as 4 with cold odd sources.
// The odd lane stores the neighbour's first word; the host counts scrub values (stale) and
// wrong values. All lanes wait_group 0 before exit. Usage: divcount MODE NCTA LAUNCHES
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define LINE_WORDS 32u  // 128 bytes

template <int MODE>
__global__ void divcount(const unsigned *src, unsigned *out) {
  __shared__ __align__(16) unsigned slot[32 * 4];
  unsigned lane = threadIdx.x & 31;
  for (int i = 0; i < 4; i++) slot[lane * 4 + i] = 0xdeadbeefu;
  __syncwarp();
  const unsigned *g = src + ((size_t)blockIdx.x * 32 + lane) * LINE_WORDS;
  unsigned sa = (unsigned)__cvta_generic_to_shared(&slot[lane * 4]);
  if ((lane & 1) == 0) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(g) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  __syncwarp();
  if (lane & 1) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(g) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
    if (MODE == 0 || MODE == 3) asm volatile("cp.async.wait_group 1;" ::: "memory");
    if (MODE == 2) asm volatile("cp.async.wait_group 0;" ::: "memory");
    if (MODE == 4 || MODE == 5) {
      unsigned own;
      do {
        asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(own) : "r"(sa) : "memory");
      } while (own == 0xdeadbeefu);
    }
    unsigned v;
    unsigned na = (unsigned)__cvta_generic_to_shared(&slot[(lane - 1) * 4]);
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(na) : "memory");
    out[blockIdx.x * 16 + (lane >> 1)] = v;
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// Makes the odd lanes' source lines L2-resident (modes 3 and 4).
__global__ void warm(const unsigned *src, unsigned *sink) {
  unsigned lane = threadIdx.x & 31;
  if ((lane & 1) == 0) return;
  const unsigned *g = src + ((size_t)blockIdx.x * 32 + lane) * LINE_WORDS;
  unsigned v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(g) : "memory");
  if (v == 0) sink[blockIdx.x] = v;
}

typedef void (*Fn)(const unsigned *, unsigned *);

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s MODE NCTA LAUNCHES\n", argv[0]);
    return 2;
  }
  int mode = atoi(argv[1]), ncta = atoi(argv[2]), launches = atoi(argv[3]);
  const char *names[] = {"WAIT1", "NOWAIT", "WAIT0", "WAIT1W", "ORDERW", "ORDER"};
  if (mode < 0 || mode > 5) return 2;
  Fn fns[] = {divcount<0>, divcount<1>, divcount<2>, divcount<3>, divcount<4>, divcount<5>};
  Fn fn = fns[mode];
  bool warmup = mode == 3 || mode == 4;
  size_t words = (size_t)ncta * 32 * LINE_WORDS;
  unsigned *src, *out;
  char *flush;
  size_t flush_bytes = (size_t)512 << 20;
  CK(cudaMalloc(&src, words * 4));
  CK(cudaMalloc(&out, (size_t)ncta * 16 * 4));
  CK(cudaMalloc(&flush, flush_bytes));
  std::vector<unsigned> hsrc(words);
  for (size_t i = 0; i < words; i++) hsrc[i] = (unsigned)(i * 2654435761u) | 1u;
  CK(cudaMemcpy(src, hsrc.data(), words * 4, cudaMemcpyHostToDevice));
  std::vector<unsigned> h((size_t)ncta * 16);
  long stale = 0, wrong = 0, okc = 0, stale_launches = 0;
  for (int it = 0; it < launches; it++) {
    CK(cudaMemset(flush, it & 0xff, flush_bytes));  // evict the sources from L2
    CK(cudaMemset(out, 0, (size_t)ncta * 16 * 4));
    if (warmup) warm<<<ncta, 32>>>(src, out);
    fn<<<ncta, 32>>>(src, out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost));
    long s = 0;
    for (int c = 0; c < ncta; c++)
      for (int k = 0; k < 16; k++) {
        unsigned v = h[(size_t)c * 16 + k];
        unsigned want = hsrc[((size_t)c * 32 + 2 * k) * LINE_WORDS];
        if (v == 0xdeadbeefu) s++;
        else if (v == want) okc++;
        else wrong++;
      }
    stale += s;
    stale_launches += s > 0;
  }
  printf("mode=%s ncta=%d launches=%d reads=%ld stale=%ld ok=%ld wrong=%ld launches_with_stale=%ld\n",
         names[mode], ncta, launches, (long)ncta * 16 * launches, stale, okc, wrong,
         stale_launches);
  return 0;
}
