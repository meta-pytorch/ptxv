/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// E-CLOCK MISS 78 x MISS 70: a 2000-cycle trap guard with both clock reads in one __noinline__
// callee (start, a 100 us sleep, now). Both reads are the same CS2R, so the hang guard's slice
// has one clock read: hang_bound_holds sets it to S in every run, the delay is never applied
// (now - start = 0 in the proof), and the guard is "proven" >= 2^31 whatever its bound.
// GB300 sleeps 100 us between the reads (about 2e5 cycles): the guard fires.
__device__ __noinline__ long long now_cycles() { return clock64(); }

extern "C" __global__ void hang_callee(unsigned* out) {
  long long t0 = now_cycles();
  __nanosleep(100000);
  long long t1 = now_cycles();
  if (t1 - t0 > 2000) __trap();
  out[threadIdx.x] = 1;
}
