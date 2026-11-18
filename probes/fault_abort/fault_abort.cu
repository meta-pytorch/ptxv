// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// probes/fault_abort: does a memory fault stop only the faulting thread on GB300, or the grid?
// Build and run: nvcc -arch=sm_103a -O3 -o fault_abort fault_abort.cu && ./fault_abort
#include <cstdio>
#include <chrono>
// Lane 3 of warp 0 loads from an unmapped address inside a divergent branch; the other lanes
// reconverge after it (BSSY/BSYNC) and then spin until lane 3 has written a flag it would only
// write after its load. If a fault stopped only the faulting thread, the spin (or the BSYNC)
// would never end; the launch returns cudaErrorIllegalAddress instead if the grid is aborted.
__global__ void k(volatile int* flag, int* out, const int* bad_idx) {
  int lane = threadIdx.x;
  int v = 0;
  if (lane == 3) {
    const volatile int* p = (const volatile int*)(0x10ull + bad_idx[0]);
    v = *p;           // unmapped: a genuine fault
    flag[0] = 1;
  }
  __syncwarp();
  if (lane != 3) { while (flag[0] == 0) { } }
  out[lane] = v + 1;
}
int main() {
  int *flag, *out, *idx;
  cudaMalloc(&flag, 4); cudaMalloc(&out, 128); cudaMalloc(&idx, 4);
  cudaMemset(flag, 0, 4); cudaMemset(idx, 0, 4);
  auto t0 = std::chrono::steady_clock::now();
  k<<<1, 32>>>((volatile int*)flag, out, idx);
  cudaError_t e = cudaDeviceSynchronize();
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  printf("launch result: %d (%s) after %.1f ms\n", (int)e, cudaGetErrorString(e), ms);
  return 0;
}
