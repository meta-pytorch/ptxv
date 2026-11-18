// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// N-MUFU (probes/mufu_range): exhaustive GB300 sweep of the PTX f32 approx ops over all 2^32
// operands. Records every operand whose result is finite with |result| among the 256 largest f32
// magnitudes (|bits| >= 0x7f7fff00), every operand in an overflow window (ex2: x in
// [0x42ffff00, 0x43000100]; rcp: |x| in [0x001fff00, 0x00200100]) whatever its result, and per op
// the count of finite results in the top 16 magnitudes, of infinite and NaN results, and the
// largest finite magnitude.
//   nvcc -O2 -gencode arch=compute_103a,code=sm_103a -o sweep sweep.cu
//   ./sweep RECORDS.csv SUMMARY.csv
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NOPS 15
static const char *NAMES[NOPS] = {
    "ex2.approx.f32",   "ex2.approx.ftz.f32",   "rcp.approx.f32",    "rcp.approx.ftz.f32",
    "rsqrt.approx.f32", "rsqrt.approx.ftz.f32", "lg2.approx.f32",    "lg2.approx.ftz.f32",
    "sqrt.approx.f32",  "sqrt.approx.ftz.f32",  "sin.approx.f32",    "sin.approx.ftz.f32",
    "cos.approx.f32",   "cos.approx.ftz.f32",   "tanh.approx.f32"};

template <int OP>
__device__ __forceinline__ float apply(float x) {
    float y;
    if (OP == 0) asm volatile("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 1) asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 2) asm volatile("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 3) asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 4) asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 5) asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 6) asm volatile("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 7) asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 8) asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 9) asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 10) asm volatile("sin.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 11) asm volatile("sin.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 12) asm volatile("cos.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 13) asm volatile("cos.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    if (OP == 14) asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

#define CAP (1u << 22)
struct Stat {
    unsigned long long top16, inf, nan;
    unsigned maxfin;
};

template <int OP>
__global__ void sweep(uint2 *rec, unsigned *nrec, Stat *st) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long nth = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long t16 = 0, ninf = 0, nnan = 0;
    unsigned mx = 0;
    for (unsigned long long i = tid; i < (1ull << 32); i += nth) {
        unsigned xb = (unsigned)i;
        float y = apply<OP>(__uint_as_float(xb));
        unsigned yb = __float_as_uint(y);
        unsigned ya = yb & 0x7fffffffu;
        unsigned xa = xb & 0x7fffffffu;
        bool fin = ya < 0x7f800000u;
        if (ya == 0x7f800000u) ninf++;
        if (ya > 0x7f800000u) nnan++;
        if (fin && ya > mx) mx = ya;
        if (fin && ya >= 0x7f7ffff0u) t16++;
        bool win = (OP <= 1 && xb >= 0x42ffff00u && xb <= 0x43000100u) ||
                   ((OP == 2 || OP == 3) && xa >= 0x001fff00u && xa <= 0x00200100u);
        if ((fin && ya >= 0x7f7fff00u) || win) {
            unsigned k = atomicAdd(nrec, 1u);
            if (k < CAP) rec[k] = make_uint2(xb, yb);
        }
    }
    atomicAdd(&st->top16, t16);
    atomicAdd(&st->inf, ninf);
    atomicAdd(&st->nan, nnan);
    atomicMax(&st->maxfin, mx);
}

typedef void (*KFn)(uint2 *, unsigned *, Stat *);
static KFn K[NOPS] = {sweep<0>, sweep<1>, sweep<2>,  sweep<3>,  sweep<4>,  sweep<5>,  sweep<6>, sweep<7>,
                      sweep<8>, sweep<9>, sweep<10>, sweep<11>, sweep<12>, sweep<13>, sweep<14>};

#define RK(x)                                                                   \
    do {                                                                        \
        cudaError_t e_ = (x);                                                   \
        if (e_ != cudaSuccess) {                                                \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: sweep RECORDS.csv SUMMARY.csv\n");
        return 2;
    }
    FILE *fo = fopen(argv[1], "w"), *fs = fopen(argv[2], "w");
    fprintf(fo, "op,x,y\n");
    fprintf(fs, "op,records,finite_top16,inf,nan,max_finite\n");
    uint2 *rec;
    unsigned *nrec;
    Stat *st;
    RK(cudaMalloc(&rec, CAP * sizeof(uint2)));
    RK(cudaMalloc(&nrec, 4));
    RK(cudaMalloc(&st, sizeof(Stat)));
    uint2 *h = (uint2 *)malloc(CAP * sizeof(uint2));
    for (int op = 0; op < NOPS; op++) {
        RK(cudaMemset(nrec, 0, 4));
        RK(cudaMemset(st, 0, sizeof(Stat)));
        K[op]<<<8192, 256>>>(rec, nrec, st);
        RK(cudaGetLastError());
        RK(cudaDeviceSynchronize());
        unsigned n;
        Stat s;
        RK(cudaMemcpy(&n, nrec, 4, cudaMemcpyDeviceToHost));
        RK(cudaMemcpy(&s, st, sizeof s, cudaMemcpyDeviceToHost));
        unsigned m = n < CAP ? n : CAP;
        RK(cudaMemcpy(h, rec, m * sizeof(uint2), cudaMemcpyDeviceToHost));
        for (unsigned k = 0; k < m; k++) fprintf(fo, "%s,0x%08x,0x%08x\n", NAMES[op], h[k].x, h[k].y);
        fprintf(fs, "%s,%u,%llu,%llu,%llu,0x%08x\n", NAMES[op], n, s.top16, s.inf, s.nan, s.maxfin);
        printf("%s records %u top16 %llu inf %llu nan %llu maxfin 0x%08x%s\n", NAMES[op], n, s.top16, s.inf, s.nan,
               s.maxfin, n > CAP ? " (TRUNCATED)" : "");
        fflush(stdout);
    }
    fclose(fo);
    fclose(fs);
    return 0;
}
