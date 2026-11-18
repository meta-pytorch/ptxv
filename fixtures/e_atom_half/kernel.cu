// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// fixtures/e_atom_half: cuda_fp16.h / cuda_bf16.h atomicAdd on __half, __nv_bfloat16, __half2
// and __nv_bfloat162 through generic pointers (ATen's fp16 / bf16 index kernels).
// kernel.ptx = nvcc 13.0 -arch=sm_103a -O3 -ptx kernel.cu.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void e_atom_half(__half* h1, __nv_bfloat16* b1, __half2* h2,
                                       __nv_bfloat162* b2, const __half* vh,
                                       const __nv_bfloat16* vb, const int* idx, int* out) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx[t];
  if (i >= 0) {
    atomicAdd(h1 + i, vh[t]);
    atomicAdd(b1 + i, vb[t]);
    atomicAdd(h2 + (i >> 1), __halves2half2(vh[t], vh[t]));
    atomicAdd(b2 + (i >> 1), __halves2bfloat162(vb[t], vb[t]));
  }
  out[t] = i;
}
