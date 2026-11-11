/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Probe for the rung 1e PDL model (E-PDL): what griddepcontrol.launch_dependents and
// griddepcontrol.wait do on GB300. Each experiment launches a primary and a dependent kernel on
// one stream with cudaLaunchAttributeProgrammaticStreamSerialization (unless noted) and
// records, in the dependent, what it saw of the primary before and after griddepcontrol.wait.
//
//   flag   written by the primary at the very end (after a 2 ms spin)
//   early  the dependent's read of flag before griddepcontrol.wait (0 = it ran early)
//   late   the same read after griddepcontrol.wait
//
// Build: nvcc -arch=sm_103a -O2 -o pdl pdl.cu   Run: CUDA_VISIBLE_DEVICES=<free gpu> ./pdl
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

__device__ __forceinline__ unsigned long long now() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ void spin(unsigned long long ns) {
    unsigned long long t0 = now(); while (now() - t0 < ns) {}
}
__device__ __forceinline__ void trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void gdwait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// mode bit 0: trigger at all; bit 1: only thread 0 of each CTA triggers (others spin, then
// exit); bit 2: CTA 0 never triggers (it spins and exits); bit 3: write `pre` = 42 before the
// trigger.
__global__ void primary(int mode, volatile int* flag, volatile int* pre) {
    if ((mode & 8) && threadIdx.x == 0 && blockIdx.x == 0) *pre = 42;
    bool trig = (mode & 1) && !((mode & 4) && blockIdx.x == 0);
    if (trig && (!(mode & 2) || threadIdx.x == 0)) trigger();
    spin(2000000);
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) { __threadfence(); *flag = 1; }
}

__global__ void dependent(volatile int* flag, volatile int* pre, int* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    out[0] = *flag;
    out[1] = *pre;
    gdwait();
    out[2] = *flag;
}

int main() {
    int *flag, *pre, *out, h[3];
    CK(cudaMalloc(&flag, 4)); CK(cudaMalloc(&pre, 4)); CK(cudaMalloc(&out, 12));
    int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    struct { const char* name; int mode, pgrid, pdl; } ex[] = {
        {"E1 all threads trigger", 1, 1, 1},
        {"E2 no trigger", 0, 1, 1},
        {"E3 only thread 0 of the CTA triggers", 3, 1, 1},
        {"E4 4 CTAs trigger, CTA 0 does not", 5, 4, 1},
        {"E5 data written before the trigger", 9, 1, 1},
        {"E6 trigger, launched WITHOUT the PDL attribute", 1, 1, 0},
    };
    for (auto& e : ex) {
        int early = 0, late = 0, sawpre = 0, n = 20;
        for (int r = 0; r < n; r++) {
            CK(cudaMemset(flag, 0, 4)); CK(cudaMemset(pre, 0, 4)); CK(cudaMemset(out, 0xff, 12));
            CK(cudaDeviceSynchronize());
            primary<<<e.pgrid, 128>>>(e.mode, flag, pre);
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = 1; cfg.blockDim = 32;
            cudaLaunchAttribute at[1];
            at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            at[0].val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = at; cfg.numAttrs = e.pdl ? 1 : 0;
            CK(cudaLaunchKernelEx(&cfg, dependent, (volatile int*)flag, (volatile int*)pre, out));
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(h, out, 12, cudaMemcpyDeviceToHost));
            early += h[0] == 0; late += h[2] == 1; sawpre += h[1] == 42;
        }
        printf("%-48s dependent ran early %2d/%d | flag after wait %2d/%d | pre seen early %2d/%d\n",
               e.name, early, n, late, n, sawpre, n);
    }
    return 0;
}
