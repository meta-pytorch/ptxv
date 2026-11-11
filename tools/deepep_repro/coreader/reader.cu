/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Co-resident reader for tools/deepep_repro/coreader.py (docs/deepep-repro.md, "Co-reader").
// Every CTA (one warp; lane 0 loads) repeatedly weak-loads ONE 32-bit word of DeepEP's local
// NVLink buffer for `cycles` clock cycles: ld.global.u32 (LDG.E, allocates in L1). The word is
// one that no DeepEP kernel writes while the reader runs (see the legality argument), so the
// reader never races with DeepEP; it only shares the 128-byte L1 line with words DeepEP's
// peers do write. Build: nvcc -O3 -arch=sm_103a -shared -Xcompiler -fPIC -o libreader.so reader.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
__global__ void reader(const int* p, long long cycles, int* sink, unsigned* smids, unsigned* iters, unsigned zmask) {
  unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  if (threadIdx.x != 0) return;
  long long t0 = clock64();
  int acc = 0; unsigned n = 0;
  while (clock64() - t0 < cycles) {
    // ptxas hoists a weak load with a loop-invariant address out of the loop (seen in SASS, even
    // from asm volatile); offset (n & zmask) with zmask = 0 passed at run time keeps one load per
    // iteration.
    unsigned z = n & zmask;
    int v; asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p + z) : "memory");
    acc += v; n++;
  }
  sink[blockIdx.x] = acc; smids[blockIdx.x] = s; iters[blockIdx.x] = n;
}
extern "C" int launch_reader(const int* p, long long cycles, int nblk, int* sink, unsigned* smids,
                             unsigned* iters, void* stream) {
  reader<<<nblk, 32, 0, (cudaStream_t)stream>>>(p, cycles, sink, smids, iters, 0u);
  return (int)cudaGetLastError();
}

// NVLink traffic toward a peer GPU (legal co-running work): stores into a scratch buffer the
// peer exported with CUDA IPC (not DeepEP memory), for `cycles` clock cycles.
__global__ void push(float4* peer, unsigned long long n, long long cycles) {
  long long t0 = clock64();
  unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x, st = (unsigned long long)gridDim.x * blockDim.x;
  float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
  while (clock64() - t0 < cycles)
    for (unsigned long long j = i; j < n && clock64() - t0 < cycles; j += st) peer[j] = v;
}
extern "C" int ipc_export(unsigned long long bytes, void** ptr, void* handle64) {
  cudaError_t e = cudaMalloc(ptr, bytes); if (e) return (int)e;
  return (int)cudaIpcGetMemHandle((cudaIpcMemHandle_t*)handle64, *ptr);
}
extern "C" int ipc_open(const void* handle64, void** ptr) {
  cudaIpcMemHandle_t h; memcpy(&h, handle64, sizeof h);
  return (int)cudaIpcOpenMemHandle(ptr, h, cudaIpcMemLazyEnablePeerAccess);
}
extern "C" int launch_push(void* peer, unsigned long long bytes, long long cycles, int nblk, void* stream) {
  push<<<nblk, 256, 0, (cudaStream_t)stream>>>((float4*)peer, bytes / 16, cycles);
  return (int)cudaGetLastError();
}

// Delayed launch from a host thread (legal: an ordinary launch on the reader stream, issued
// while the main thread is inside buf.dispatch()). The main thread joins before it orders the
// next combine after the reader stream.
#include <thread>
#include <chrono>
static std::thread g_th;
extern "C" int launch_reader_delayed(const int* p, long long cycles, int nblk, int* sink, unsigned* smids,
                                     unsigned* iters, void* stream, long long delay_ns) {
  int dev = 0; cudaGetDevice(&dev);
  g_th = std::thread([=] {
    cudaSetDevice(dev);
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count() < delay_ns) {}
    reader<<<nblk, 32, 0, (cudaStream_t)stream>>>(p, cycles, sink, smids, iters, 0u);
  });
  return 0;
}
extern "C" int join_reader() { if (g_th.joinable()) g_th.join(); return (int)cudaGetLastError(); }
