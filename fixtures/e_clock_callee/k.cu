/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// E-CLOCK (MISS 70): the clock is read in a non-inlined device function (SASS CALL.REL) and the
// caller compares the returned value: a clock64 timeout whose timeout path reads the data
// without the flag (e_clock_timeout's shape across a call).
__device__ __noinline__ long long now_cycles() { return clock64(); }

__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

extern "C" __global__ void clock_call(unsigned* out) {
  unsigned tid = threadIdx.x;
  if (tid == 0) {
    __nanosleep(200000);
    out[1] = 7;
    st_release(&out[2], 1);
    return;
  }
  if (tid != 32) return;
  long long t0 = now_cycles();
  while (ld_acquire(&out[2]) == 0) {
    if (now_cycles() - t0 > 2000) break;
  }
  out[0] = out[1];
}
