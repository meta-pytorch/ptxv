/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// E-WARPID probe: %warpid against tid / 32, read twice 200 us apart; %laneid against tid % 32.
// nvcc -arch=sm_103a -o warpid warpid.cu && ./warpid
#include <cstdio>
#include <vector>
#include <map>
__global__ void k(unsigned *out, unsigned *out2, unsigned *lane) {
  unsigned w, l, w2;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(w));
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(l));
  for (int i = 0; i < 2000; i++) __nanosleep(100);
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(w2));
  unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
  out[g] = w; out2[g] = w2; lane[g] = l;
}
int main() {
  int B = 1024, T = 64, N = B * T;
  unsigned *o, *o2, *l;
  cudaMallocManaged(&o, N * 4); cudaMallocManaged(&o2, N * 4); cudaMallocManaged(&l, N * 4);
  k<<<B, T>>>(o, o2, l); cudaDeviceSynchronize();
  int diff_tid = 0, unstable = 0, badlane = 0; std::map<unsigned,int> hist;
  for (int g = 0; g < N; g++) {
    int t = g % T;
    if (o[g] != (unsigned)(t / 32)) diff_tid++;
    if (o[g] != o2[g]) unstable++;
    if (l[g] != (unsigned)(t % 32)) badlane++;
    hist[o[g]]++;
  }
  printf("threads %d: warpid != tid/32: %d, warpid changed during the warp: %d, laneid != tid%%32: %d\n", N, diff_tid, unstable, badlane);
  printf("distinct warpid values: %zu (max %u)\n", hist.size(), hist.rbegin()->first);
  // within a warp, all lanes equal?
  int split = 0;
  for (int w = 0; w < N / 32; w++) for (int i = 1; i < 32; i++) if (o[w*32+i] != o[w*32]) { split++; break; }
  printf("warps whose lanes disagree: %d\n", split);
}
