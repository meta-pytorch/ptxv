/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// E-CLOCK MISS 76 follow-up (SASS): the clock is read in a __noinline__ callee called by the odd
// lanes only. lane_varying_choices marks values written under lane-varying control through
// elect_lanes::Cfg regions, which do not follow CALL into the callee (nor BRX targets), so the
// callee's clock read is not lane-varying on the SASS front, while the lanes that skipped the
// call keep v = 0. A branch on v & 1 after reconvergence then splits the warp on hardware
// (odd lanes one side, even lanes the other) whenever the odd lanes' clock is odd.
__device__ __noinline__ unsigned rd_clk() { return (unsigned)clock(); }

extern "C" __global__ void lanes_call(unsigned* out, const unsigned* n) {
  __shared__ unsigned sm[32];
  unsigned lane = threadIdx.x;
  unsigned cnt = *n;
  sm[lane] = 0;
  __syncthreads();
  unsigned v = 0;
  if (lane & 1) v = rd_clk();
  asm volatile("" : "+r"(v));
  __syncwarp();
  unsigned i = 0;
  if (v & 1) {
    do { sm[lane] = 1; } while (++i < cnt);
  } else {
    do { sm[lane ^ 1] = 1; } while (++i < cnt);
  }
  __syncthreads();
  out[lane] = sm[lane];
}
