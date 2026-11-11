/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Non-aligned barrier.sync in divergent code (CuTeDSL barrier(aligned=False), FA4's landed fix
// for its warp-role barrier bug): ptxas emits `BRA.DIV` around the plain `BAR.SYNC id, n`, and an out-of-line
// path `WARPSYNC.COLLECTIVE.ALL end; BAR.SYNC R, R; ENDCOLLECTIVE` with R = n << 16 | id.
// GB300 (2026-11-05, GPU 3):
//   bardiv_split: both warps split by parity into two branches, each with barrier.sync 1, 64:
//     completes; each thread reads the value its neighbour stored before the barrier.
//   bardiv_mixed: warp 0 split, warp 1 converged, barrier.sync 1, 64: completes.
//   bardiv_hang:  lanes with x % 4 in {0, 2} reach barrier.sync 1, 32 in two branches, the
//     odd lanes spin on a flag set after it: hangs (the warp arrives only when all its live
//     lanes have; 16 + 16 threads do not complete a 32-thread barrier).
//   bardiv_short: warp 0 split, warp 1 converged, barrier.sync 1, 96 with 64 threads: hangs.
// nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin bardiv.cu
extern "C" __global__ void bardiv_split(int* o, const int* a) {
    int t = threadIdx.x;
    int x = a[t];
    if (x & 1) { o[t] = x; asm volatile("barrier.sync 1, 64;" ::: "memory"); x = o[t ^ 1] + 1000; }
    else { o[t] = x; asm volatile("barrier.sync 1, 64;" ::: "memory"); x = o[t ^ 1] + 2000; }
    o[64 + t] = x;
}
extern "C" __global__ void bardiv_mixed(int* o, const int* a) {
    int t = threadIdx.x;
    int x = a[t];
    o[t] = x;
    if (t < 32 && (x & 1)) { asm volatile("barrier.sync 1, 64;" ::: "memory"); o[64 + t] = o[(t + 32) & 63] + 1000; }
    else { asm volatile("barrier.sync 1, 64;" ::: "memory"); o[64 + t] = o[(t + 32) & 63] + 2000; }
}
extern "C" __global__ void bardiv_hang(volatile int* o, const int* a) {
    int t = threadIdx.x;
    int x = a[t];
    if ((x & 3) == 0) {
        o[t] = x; asm volatile("barrier.sync 1, 32;" ::: "memory"); o[200] = 1; o[64 + t] = 7;
    } else if ((x & 3) == 2) {
        o[t] = x; asm volatile("barrier.sync 1, 32;" ::: "memory"); o[200] = 1; o[64 + t] = 8;
    } else {
        while (o[200] == 0) { }
        o[64 + t] = 9;
    }
}
extern "C" __global__ void bardiv_short(int* o, const int* a) {
    int t = threadIdx.x;
    int x = a[t];
    o[t] = x;
    if (t < 32 && (x & 1)) { asm volatile("barrier.sync 1, 96;" ::: "memory"); o[128 + t] = 1; }
    else if (t < 64) { asm volatile("barrier.sync 1, 96;" ::: "memory"); o[128 + t] = 2; }
}
