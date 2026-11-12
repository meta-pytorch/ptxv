/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// E-HW-L1 probe: griddepcontrol.wait lowers to ACQBULK with no CCTL.IVALL. Can a dependent
// grid's weak load after it hit an L1 line its SM filled before the wait?
//
// primary (1 CTA): thread 0 runs griddepcontrol.launch_dependents, waits DELAY ns, stores
// D[5] = 5 (weak), then F = 1 (st.release.gpu). D is zeroed before each launch.
// secondary (PDL, NCTA CTAs of 32 threads): thread 0 loads D[0] (the same 32-byte sector as
// D[5]: fills its SM's L1 line while D[5] is still 0), uses the value (the load completes
// first), waits per mode, loads D[5] (weak, L1-cached) into out[cta] and records its SM.
//   0 WAIT:   griddepcontrol.wait                      (the open question)
//   1 WAITF:  griddepcontrol.wait; fence.acq_rel.gpu   (CCTL.IVALL: must read 5)
//   2 NOWAIT: no wait                                  (timing control: reads before the store)
//   3 POLL:   ld.relaxed.gpu poll of F, no fence       (L1 control: a stale hit is allowed)
//   4 POLLF:  poll of F; fence.acq_rel.gpu             (must read 5)
//   5 POLLDONE: poll of F, then 20 us more (the primary has exited), no wait, no fence:
//             does the primary's completion drop the lines (then 0 stale)?
// Modes 6-8 ask whether ACQBULK drops lines when there is nothing left to wait for: the
// primary only triggers and exits; in the secondary every CTA's thread 0 first spins 20 us
// (the primary has completed), then CTA 0 is a writer and the others readers. A reader loads
// D[0] (fills its line, D[5] = 0) and counts itself in; once all have, the writer stores
// D[5] = 7 (weak) and F = 1 (st.release.gpu); the reader polls F (ld.relaxed.gpu), then
//   6 LATEWAIT:  griddepcontrol.wait                   (the question)
//   7 LATENONE:  nothing                               (L1 control: a stale hit is allowed)
//   8 LATEFENCE: fence.acq_rel.gpu                     (must read 7)
// and loads D[5]. Readers' results: 7 counts as read5, 0 as read0.
// Per mode and delay: launches, CTAs, CTAs that read 0 (stale or early), launches with one,
// CTAs on the primary's SM. Usage: pdl MODE DELAY_NS NCTA LAUNCHES
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

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ unsigned smid() {
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

__global__ void primary(unsigned *D, unsigned *F, unsigned long long delay, unsigned *psm) {
  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    *psm = smid();
    unsigned long long t0 = gtimer();
    while (gtimer() - t0 < delay) {
    }
    asm volatile("st.global.u32 [%0], %1;" ::"l"(D + 5), "r"(5u) : "memory");
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(F), "r"(1u) : "memory");
  }
}

template <int MODE>
__global__ void secondary(const unsigned *D, const unsigned *F, unsigned *out, unsigned *sm,
                          unsigned *pre) {
  if (threadIdx.x != 0) return;
  unsigned a;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(a) : "l"(D) : "memory");
  if (a == 0xdeadbeefu) pre[blockIdx.x] = a;
  if (MODE == 0 || MODE == 1) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (MODE == 3 || MODE == 4 || MODE == 5) {
    unsigned f;
    do {
      asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(F) : "memory");
    } while (f == 0);
  }
  if (MODE == 5) {
    unsigned long long t0 = gtimer();
    while (gtimer() - t0 < 20000) {
    }
  }
  if (MODE == 1 || MODE == 4) asm volatile("fence.acq_rel.gpu;" ::: "memory");
  unsigned v;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(D + 5) : "memory");
  out[blockIdx.x] = v;
  sm[blockIdx.x] = smid();
}

__global__ void primary_nop(unsigned *psm) {
  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    *psm = smid();
  }
}

// Modes 6-8 (see the header). N: the readers' count cell (zeroed before each launch).
template <int MODE>
__global__ void late(unsigned *D, unsigned *F, unsigned *out, unsigned *sm, unsigned *N) {
  if (threadIdx.x != 0) return;
  unsigned long long t0 = gtimer();
  while (gtimer() - t0 < 20000) {
  }
  unsigned readers = gridDim.x - 1;
  if (blockIdx.x == 0) {
    unsigned n;
    do {
      asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(n) : "l"(N) : "memory");
    } while (n < readers);
    asm volatile("st.global.u32 [%0], %1;" ::"l"(D + 5), "r"(7u) : "memory");
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(F), "r"(1u) : "memory");
    out[0] = 7;
    sm[0] = smid();
    return;
  }
  unsigned a;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(a) : "l"(D) : "memory");
  if (a == 0xdeadbeefu) out[blockIdx.x] = a;
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" ::"l"(N) : "memory");
  unsigned f;
  do {
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(F) : "memory");
  } while (f == 0);
  if (MODE == 6) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (MODE == 8) asm volatile("fence.acq_rel.gpu;" ::: "memory");
  unsigned v;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(D + 5) : "memory");
  out[blockIdx.x] = v == 7 ? 5 : v;
  sm[blockIdx.x] = smid();
}

typedef void (*SecFn)(const unsigned *, const unsigned *, unsigned *, unsigned *, unsigned *);
static SecFn sec_for(int m) {
  switch (m) {
    case 0: return secondary<0>;
    case 1: return secondary<1>;
    case 2: return secondary<2>;
    case 3: return secondary<3>;
    case 4: return secondary<4>;
    default: return secondary<5>;
  }
}

typedef void (*LateFn)(unsigned *, unsigned *, unsigned *, unsigned *, unsigned *);
static LateFn late_for(int m) {
  return m == 6 ? late<6> : m == 7 ? late<7> : late<8>;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s MODE DELAY_NS NCTA LAUNCHES\n", argv[0]);
    return 2;
  }
  int mode = atoi(argv[1]);
  unsigned long long delay = strtoull(argv[2], 0, 10);
  int ncta = atoi(argv[3]), launches = atoi(argv[4]);
  const char *names[] = {"WAIT",     "WAITF",    "NOWAIT",   "POLL",     "POLLF",
                         "POLLDONE", "LATEWAIT", "LATENONE", "LATEFENCE"};
  if (mode < 0 || mode > 8) return 2;
  bool is_late = mode >= 6;
  unsigned *D, *F, *out, *sm, *pre, *psm, *N;
  CK(cudaMalloc(&D, 4096));
  CK(cudaMalloc(&F, 4096));
  CK(cudaMalloc(&out, ncta * 4));
  CK(cudaMalloc(&sm, ncta * 4));
  CK(cudaMalloc(&pre, ncta * 4));
  CK(cudaMalloc(&psm, 4));
  CK(cudaMalloc(&N, 4));
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  std::vector<unsigned> h(ncta), hs(ncta);
  long zero = 0, five = 0, other = 0, zero_launches = 0, on_psm = 0, zero_on_psm = 0;
  SecFn fn = sec_for(mode);
  for (int it = 0; it < launches; it++) {
    CK(cudaMemsetAsync(D, 0, 4096, s));
    CK(cudaMemsetAsync(F, 0, 4096, s));
    CK(cudaMemsetAsync(out, 0xff, ncta * 4, s));
    CK(cudaMemsetAsync(N, 0, 4, s));
    if (is_late)
      primary_nop<<<1, 32, 0, s>>>(psm);
    else
      primary<<<1, 32, 0, s>>>(D, F, delay, psm);
    CK(cudaGetLastError());
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(ncta);
    cfg.blockDim = dim3(32);
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (is_late)
      CK(cudaLaunchKernelEx(&cfg, late_for(mode), D, F, out, sm, N));
    else
      CK(cudaLaunchKernelEx(&cfg, fn, (const unsigned *)D, (const unsigned *)F, out, sm, pre));
    CK(cudaStreamSynchronize(s));
    unsigned hp;
    CK(cudaMemcpy(h.data(), out, ncta * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), sm, ncta * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hp, psm, 4, cudaMemcpyDeviceToHost));
    long z = 0;
    for (int c = is_late ? 1 : 0; c < ncta; c++) {
      bool same = hs[c] == hp;
      on_psm += same;
      if (h[c] == 0) {
        z++;
        zero_on_psm += same;
      } else if (h[c] == 5) {
        five++;
      } else {
        other++;
      }
    }
    zero += z;
    zero_launches += z > 0;
  }
  printf("mode=%s delay_ns=%llu ncta=%d launches=%d ctas=%ld read0=%ld read5=%ld other=%ld "
         "launches_with_read0=%ld ctas_on_primary_sm=%ld read0_on_primary_sm=%ld\n",
         names[mode], delay, ncta, launches, (long)(is_late ? ncta - 1 : ncta) * launches, zero,
         five, other,
         zero_launches, on_psm, zero_on_psm);
  return 0;
}
