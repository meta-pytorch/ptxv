/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Intra-warp shared-memory ordering probe (E-LOCKSTEP): in a converged warp, is a shared
// store by one lane visible to a shared load by another lane on the next instruction, and
// does a load by one lane always see the old value when another lane stores on the next
// instruction? No bar.warp.sync / __syncwarp between them. The fla Triton kernels
// (num_warps = 1) have exactly this shape: `STSM.16.M88.4` by every lane, then `LDS.128` by
// another lane; `LDSM.16.M88.4`, then `STS.128` by another lane.
//
// Each warp owns a 512-byte region of shared memory and runs ITERS iterations; every
// iteration writes values that differ from the previous iteration's, so a load that runs
// early or late returns a wrong value. Every check counts mismatches per warp.
//
// Tests (`mode`):
//   0 wr32   lane W: st.shared.u32 [x], v_i;   lane R: ld.shared.u32 [x]    expect v_i
//   1 wr128  lane W: st.shared.v4.u32 [x];     lane R: ld.shared.v4.u32 [x] (STS.128 / LDS.128)
//   2 wr64r  lane W: st.shared.u32 [x+4];      lane R: ld.shared.v2.u32 [x] (partial overlap)
//   3 stsm   all lanes stmatrix.x4 (STSM.16.M88.4); lane R: ld.shared.v4 of one row
//   4 rw32   lane R: ld.shared.u32 [x];        lane W: st.shared.u32 [x]    expect old value
//   5 rw128  lane R: ld.shared.v4 [x];         lane W: st.shared.v4 [x]     expect old value
//   6 ldsm   all lanes ldmatrix.x4 (LDSM.16.M88.4); lane W: st.shared.v4 to one row (old)
//   7 xwarp  control: warp 2k+1 stores v_i, warp 2k loads the same word in iteration i with
//            no synchronization: mismatches show that timing between warps varies.
//   8 diverge control: the writer and reader loops on the two sides of a lane-dependent
//            branch (a diverged warp, no reconvergence inside the loops).
// `off` moves the accessed word within the region (bank), `wl` / `rl` pick the lanes.
//   nvcc -O3 -arch=sm_103a -o warp_order warp_order.cu    (and -DUNROLL1, run.sh)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define ITERS 4096
// -DUNROLL1: no unrolling (each store / load pair adjacent or nearly so in SASS).
#ifdef UNROLL1
#define LOOP _Pragma("unroll 1")
#else
#define LOOP
#endif
#define REGION 512

__device__ __forceinline__ unsigned val(unsigned i, unsigned k) {
    return (i * 0x9e3779b9u) ^ (k * 0x85ebca6bu) ^ 0x5bd1e995u;
}

extern "C" __global__ void probe(unsigned* errs, int mode, int off, int wl, int rl) {
    extern __shared__ __align__(1024) unsigned char smem[];
    int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    unsigned char* reg = smem + w * REGION;
    unsigned base = (unsigned)__cvta_generic_to_shared(reg);
    // Initialize the region (iteration "-1" values) and converge.
    for (int j = lane; j < REGION / 4; j += 32) ((unsigned*)reg)[j] = val(0xffffffffu, j);
    __syncwarp();
    __syncthreads();
    unsigned err = 0;
    unsigned a = base + off;
    bool isw = lane == wl, isr = lane == rl;
    if (mode == 0) {
        LOOP for (unsigned i = 0; i < ITERS; i++) {
            unsigned v = val(i, 0), x;
            asm volatile(
                "{ .reg .pred pw, pr;\n setp.ne.u32 pw, %3, 0;\n setp.ne.u32 pr, %4, 0;\n"
                " @pw st.shared.u32 [%1], %2;\n @pr ld.shared.u32 %0, [%1];\n}"
                : "=r"(x) : "r"(a), "r"(v), "r"((unsigned)isw), "r"((unsigned)isr) : "memory");
            if (isr && x != v) err++;
        }
    } else if (mode == 1) {
        LOOP for (unsigned i = 0; i < ITERS; i++) {
            unsigned v0 = val(i, 0), v1 = val(i, 1), v2 = val(i, 2), v3 = val(i, 3);
            unsigned x0, x1, x2, x3;
            asm volatile(
                "{ .reg .pred pw, pr;\n setp.ne.u32 pw, %9, 0;\n setp.ne.u32 pr, %10, 0;\n"
                " @pw st.shared.v4.u32 [%4], {%5, %6, %7, %8};\n"
                " @pr ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n}"
                : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                : "r"(a), "r"(v0), "r"(v1), "r"(v2), "r"(v3), "r"((unsigned)isw),
                  "r"((unsigned)isr)
                : "memory");
            if (isr && (x0 != v0 || x1 != v1 || x2 != v2 || x3 != v3)) err++;
        }
    } else if (mode == 2) {
        LOOP for (unsigned i = 0; i < ITERS; i++) {
            unsigned v = val(i, 1), x0, x1;
            asm volatile(
                "{ .reg .pred pw, pr;\n setp.ne.u32 pw, %4, 0;\n setp.ne.u32 pr, %5, 0;\n"
                " @pw st.shared.u32 [%2+4], %3;\n @pr ld.shared.v2.u32 {%0, %1}, [%2];\n}"
                : "=r"(x0), "=r"(x1) : "r"(a), "r"(v), "r"((unsigned)isw), "r"((unsigned)isr)
                : "memory");
            if (isr && x1 != v) err++;
        }
    } else if (mode == 3) {
        // stmatrix.x4: lane j gives the address of row j % 8 of matrix j / 8 (16 bytes); row
        // r of matrix m holds the 32-bit register m of lanes 4r..4r+3. Rows are 16 bytes
        // apart, matrices 128 bytes apart: the 512-byte region.
        unsigned ra = base + (lane / 8) * 128 + (lane % 8) * 16;
        // Lane R reads the row lane W addressed.
        unsigned rd = base + (wl / 8) * 128 + (wl % 8) * 16;
        LOOP for (unsigned i = 0; i < ITERS; i++) {
            unsigned r0 = val(i, lane * 4), r1 = val(i, lane * 4 + 1), r2 = val(i, lane * 4 + 2),
                     r3 = val(i, lane * 4 + 3);
            unsigned x0, x1, x2, x3;
            asm volatile(
                "{ .reg .pred pr;\n setp.ne.u32 pr, %9, 0;\n"
                " stmatrix.sync.aligned.m8n8.x4.shared.b16 [%4], {%5, %6, %7, %8};\n"
                " @pr ld.shared.v4.u32 {%0, %1, %2, %3}, [%10];\n}"
                : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                : "r"(ra), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"((unsigned)isr), "r"(rd)
                : "memory");
            if (isr) {
                int m = wl / 8, r = wl % 8;
                // Row r of matrix m: register m of lanes 4r .. 4r+3.
                if (x0 != val(i, (4 * r) * 4 + m) || x1 != val(i, (4 * r + 1) * 4 + m) ||
                    x2 != val(i, (4 * r + 2) * 4 + m) || x3 != val(i, (4 * r + 3) * 4 + m))
                    err++;
            }
        }
    } else if (mode == 4) {
        LOOP for (unsigned i = 0; i < ITERS; i++) {
            unsigned v = val(i, 0), old = i ? val(i - 1, 0) : *(volatile unsigned*)(reg + off);
            unsigned x;
            asm volatile(
                "{ .reg .pred pw, pr;\n setp.ne.u32 pw, %3, 0;\n setp.ne.u32 pr, %4, 0;\n"
                " @pr ld.shared.u32 %0, [%1];\n @pw st.shared.u32 [%1], %2;\n}"
                : "=r"(x) : "r"(a), "r"(v), "r"((unsigned)isw), "r"((unsigned)isr) : "memory");
            if (isr && x != old) err++;
            __syncwarp();
        }
    } else if (mode == 5) {
        unsigned o0 = ((unsigned*)(reg + off))[0], o1 = ((unsigned*)(reg + off))[1],
                 o2 = ((unsigned*)(reg + off))[2], o3 = ((unsigned*)(reg + off))[3];
        __syncwarp();
        LOOP for (unsigned i = 0; i < ITERS; i++) {
            unsigned v0 = val(i, 0), v1 = val(i, 1), v2 = val(i, 2), v3 = val(i, 3);
            unsigned x0, x1, x2, x3;
            asm volatile(
                "{ .reg .pred pw, pr;\n setp.ne.u32 pw, %9, 0;\n setp.ne.u32 pr, %10, 0;\n"
                " @pr ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                " @pw st.shared.v4.u32 [%4], {%5, %6, %7, %8};\n}"
                : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                : "r"(a), "r"(v0), "r"(v1), "r"(v2), "r"(v3), "r"((unsigned)isw),
                  "r"((unsigned)isr)
                : "memory");
            if (isr && (x0 != o0 || x1 != o1 || x2 != o2 || x3 != o3)) err++;
            o0 = v0, o1 = v1, o2 = v2, o3 = v3;
        }
    } else if (mode == 6) {
        // ldmatrix.x4 (same addressing as mode 3), then lane W stores a new row where lane R's
        // address points. Lane 4r+q receives row r of matrix m in register m (its 4-byte word q).
        unsigned ra = base + (lane / 8) * 128 + (lane % 8) * 16;
        unsigned wa = base + (rl / 8) * 128 + (rl % 8) * 16;
        int m = rl / 8, r = rl % 8;
        // The word lane `lane` receives from matrix m: row lane / 4, word lane % 4.
        unsigned* rowp = (unsigned*)(reg + m * 128 + (lane / 4) * 16 + (lane % 4) * 4);
        LOOP for (unsigned i = 0; i < ITERS; i++) {
            unsigned want = *(volatile unsigned*)rowp;
            __syncwarp();
            unsigned v = val(i, 7);
            unsigned x0, x1, x2, x3;
            asm volatile(
                "{ .reg .pred pw;\n setp.ne.u32 pw, %6, 0;\n"
                " ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                " @pw st.shared.v4.u32 [%7], {%5, %5, %5, %5};\n}"
                : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                : "r"(ra), "r"(v), "r"((unsigned)isw), "r"(wa)
                : "memory");
            unsigned got = m == 0 ? x0 : m == 1 ? x1 : m == 2 ? x2 : x3;
            // Lanes 4r .. 4r+3 receive the row lane W overwrote: they must see the old value.
            if (lane / 4 == r && got != want) err++;
            __syncwarp();
        }
    } else if (mode == 7) {
        // Cross-warp control: odd warps store into the even warp's region, no sync.
        unsigned* x = (unsigned*)(smem + (w & ~1) * REGION + off);
        unsigned stale = 0;
        LOOP for (unsigned i = 0; i < ITERS; i++) {
            if (w & 1) {
                if (lane == wl) *(volatile unsigned*)x = val(i, 0);
            } else if (lane == rl) {
                if (*(volatile unsigned*)x != val(i, 0)) stale++;
            }
        }
        err = stale;
    } else if (mode == 8) {
        // Divergence control: the writer lane and the reader lane run their loops on the two
        // sides of a lane-dependent branch, so the warp is diverged for the whole loop.
        volatile unsigned* x = (volatile unsigned*)(reg + off);
        if (isw) {
            LOOP for (unsigned i = 0; i < ITERS; i++) *x = val(i, 0);
        } else if (isr) {
            LOOP for (unsigned i = 0; i < ITERS; i++) if (*x != val(i, 0)) err++;
        }
        __syncwarp();
    }
    // One counter per warp.
    for (int o = 16; o; o >>= 1) err += __shfl_xor_sync(0xffffffffu, err, o);
    if (lane == 0) atomicAdd(errs + mode, err);
}

int main(int argc, char** argv) {
    int runs = argc > 1 ? atoi(argv[1]) : 10;
    int warps = argc > 2 ? atoi(argv[2]) : 32;
    int ctas = argc > 3 ? atoi(argv[3]) : 148 * 2;
    const char* only = argc > 4 ? argv[4] : nullptr;
    const char* names[] = {"wr32", "wr128", "wr64r", "stsm", "rw32", "rw128", "ldsm", "xwarp", "diverge"};
    unsigned* errs;
    cudaMalloc(&errs, 9 * sizeof(unsigned));
    size_t sm = (size_t)warps * REGION;
    cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    // Lane pairs (W, R) and byte offsets: banks 0, 1, 7, 31 and 16-byte aligned rows.
    int pairs[][2] = {{1, 0}, {0, 1}, {31, 0}, {5, 17}};
    int offs[] = {0, 4, 28, 124, 256, 496};
    printf("# runs=%d warps/cta=%d ctas=%d iters=%d\n", runs, warps, ctas, ITERS);
    for (int mode = 0; mode < 9; mode++) {
        if (only && strcmp(only, names[mode])) continue;
        unsigned long long checks = 0, wrong = 0;
        for (auto& p : pairs) {
            for (int off : offs) {
                // 16-byte accesses need 16-byte alignment, 64-bit 8-byte.
                if ((mode == 1 || mode == 5) && off % 16) continue;
                if (mode == 2 && off % 8) continue;
                if (mode == 2 && off + 8 > REGION) continue;
                if ((mode == 3 || mode == 6) && off) continue;
                if ((mode == 7 || mode == 8) && p[0] == p[1]) continue;
                if (mode == 7 && warps < 2) continue;
                for (int r = 0; r < runs; r++) {
                    cudaMemset(errs, 0, 9 * sizeof(unsigned));
                    probe<<<ctas, warps * 32, sm>>>(errs, mode, off, p[0], p[1]);
                    cudaError_t e = cudaDeviceSynchronize();
                    if (e != cudaSuccess) {
                        printf("%s: %s\n", names[mode], cudaGetErrorString(e));
                        return 1;
                    }
                    unsigned h[9];
                    cudaMemcpy(h, errs, sizeof h, cudaMemcpyDeviceToHost);
                    wrong += h[mode];
                    // Checks per run: every warp (mode 7: every reader warp) ITERS times, by
                    // one lane (modes 3 / 6: 1 / 4 lanes, counted once per warp).
                    checks += (unsigned long long)ctas * (mode == 7 ? warps / 2 : warps) * ITERS;
                }
            }
        }
        printf("%-6s checks %llu wrong %llu\n", names[mode], checks, wrong);
    }
    return 0;
}
