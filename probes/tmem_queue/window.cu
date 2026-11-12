/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// E-TMEM-QUEUE window calibration (results in window.txt). Can the check see a TMEM access land
// inside another access's in-flight window at ns scale, and does one thread's queue reorder in
// that window? A long access (32x32b.x128: columns 0..127 of the warp's 32 lanes) is followed by
// a one-column access to column 127, the last column of the long one:
//   cross-warp (warp w < 4 issues the long access, then at once sets a shared flag; warp w + 4,
// the same TMEM lanes, spins on the flag and issues the short access at once):
//   C0:  LDTM.x128 old, then STTM.x1 new by the partner: the load returns new = a store issued
//        after the load's issue landed before the load read column 127;
//   C0w: STTM.x128 a, then STTM.x1 b by the partner; later the column is read back: a = the
//        earlier-issued long store landed after the partner's store;
//   C0r: STTM.x128 new, then LDTM.x1 by the partner: old = the later load read before the long
//        store reached column 127.
//   same thread (warps 0..3; warps 4..7 idle), the short access right after the long one with
//   no wait between:
//   M0': LDTM.x128, STTM.x1 new to column 127, wait::ld (new = the store overtook the load);
//   M4': STTM.x128 a, STTM.x1 b to column 127, wait::st, LDTM (a = the stores swapped);
//   M2': STTM.x128 new, LDTM.x1 column 127, wait::ld (old = the load overtook the store).
// Every word starts as "old" (st.x128; wait::st; barrier). Counted per checked word of column
// 127: "hit" (the out-of-order value above), the expected value, anything else.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

enum { C0, C0W, C0R, M0, M4, M2, NV };
__device__ unsigned long long g_cnt[NV][3];  // [variant][hit, else, checked]

#define LD128(addr, r) asm volatile("tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63]), "=r"(r[64]), "=r"(r[65]), "=r"(r[66]), "=r"(r[67]), "=r"(r[68]), "=r"(r[69]), "=r"(r[70]), "=r"(r[71]), "=r"(r[72]), "=r"(r[73]), "=r"(r[74]), "=r"(r[75]), "=r"(r[76]), "=r"(r[77]), "=r"(r[78]), "=r"(r[79]), "=r"(r[80]), "=r"(r[81]), "=r"(r[82]), "=r"(r[83]), "=r"(r[84]), "=r"(r[85]), "=r"(r[86]), "=r"(r[87]), "=r"(r[88]), "=r"(r[89]), "=r"(r[90]), "=r"(r[91]), "=r"(r[92]), "=r"(r[93]), "=r"(r[94]), "=r"(r[95]), "=r"(r[96]), "=r"(r[97]), "=r"(r[98]), "=r"(r[99]), "=r"(r[100]), "=r"(r[101]), "=r"(r[102]), "=r"(r[103]), "=r"(r[104]), "=r"(r[105]), "=r"(r[106]), "=r"(r[107]), "=r"(r[108]), "=r"(r[109]), "=r"(r[110]), "=r"(r[111]), "=r"(r[112]), "=r"(r[113]), "=r"(r[114]), "=r"(r[115]), "=r"(r[116]), "=r"(r[117]), "=r"(r[118]), "=r"(r[119]), "=r"(r[120]), "=r"(r[121]), "=r"(r[122]), "=r"(r[123]), "=r"(r[124]), "=r"(r[125]), "=r"(r[126]), "=r"(r[127]) : "r"(addr) : "memory")
#define ST128(addr, v) asm volatile("tcgen05.st.sync.aligned.32x32b.x128.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128};" :: "r"(addr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]), "r"(v[16]), "r"(v[17]), "r"(v[18]), "r"(v[19]), "r"(v[20]), "r"(v[21]), "r"(v[22]), "r"(v[23]), "r"(v[24]), "r"(v[25]), "r"(v[26]), "r"(v[27]), "r"(v[28]), "r"(v[29]), "r"(v[30]), "r"(v[31]), "r"(v[32]), "r"(v[33]), "r"(v[34]), "r"(v[35]), "r"(v[36]), "r"(v[37]), "r"(v[38]), "r"(v[39]), "r"(v[40]), "r"(v[41]), "r"(v[42]), "r"(v[43]), "r"(v[44]), "r"(v[45]), "r"(v[46]), "r"(v[47]), "r"(v[48]), "r"(v[49]), "r"(v[50]), "r"(v[51]), "r"(v[52]), "r"(v[53]), "r"(v[54]), "r"(v[55]), "r"(v[56]), "r"(v[57]), "r"(v[58]), "r"(v[59]), "r"(v[60]), "r"(v[61]), "r"(v[62]), "r"(v[63]), "r"(v[64]), "r"(v[65]), "r"(v[66]), "r"(v[67]), "r"(v[68]), "r"(v[69]), "r"(v[70]), "r"(v[71]), "r"(v[72]), "r"(v[73]), "r"(v[74]), "r"(v[75]), "r"(v[76]), "r"(v[77]), "r"(v[78]), "r"(v[79]), "r"(v[80]), "r"(v[81]), "r"(v[82]), "r"(v[83]), "r"(v[84]), "r"(v[85]), "r"(v[86]), "r"(v[87]), "r"(v[88]), "r"(v[89]), "r"(v[90]), "r"(v[91]), "r"(v[92]), "r"(v[93]), "r"(v[94]), "r"(v[95]), "r"(v[96]), "r"(v[97]), "r"(v[98]), "r"(v[99]), "r"(v[100]), "r"(v[101]), "r"(v[102]), "r"(v[103]), "r"(v[104]), "r"(v[105]), "r"(v[106]), "r"(v[107]), "r"(v[108]), "r"(v[109]), "r"(v[110]), "r"(v[111]), "r"(v[112]), "r"(v[113]), "r"(v[114]), "r"(v[115]), "r"(v[116]), "r"(v[117]), "r"(v[118]), "r"(v[119]), "r"(v[120]), "r"(v[121]), "r"(v[122]), "r"(v[123]), "r"(v[124]), "r"(v[125]), "r"(v[126]), "r"(v[127]) : "memory")
// The long access and the flag store in one asm statement, a bar.warp.sync between them: without
// it ptxas issues the STS before the long access (SASS checked per pair, window.txt: the STS
// follows the long access with no wait on its scoreboards; a fence.acq_rel.cta would add a NOP
// that waits on every scoreboard, the LDTM's too).
#define LD128F(addr, r, f, x) asm volatile("tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128]; bar.warp.sync 0xffffffff; st.volatile.shared.u32 [%129], %130;" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63]), "=r"(r[64]), "=r"(r[65]), "=r"(r[66]), "=r"(r[67]), "=r"(r[68]), "=r"(r[69]), "=r"(r[70]), "=r"(r[71]), "=r"(r[72]), "=r"(r[73]), "=r"(r[74]), "=r"(r[75]), "=r"(r[76]), "=r"(r[77]), "=r"(r[78]), "=r"(r[79]), "=r"(r[80]), "=r"(r[81]), "=r"(r[82]), "=r"(r[83]), "=r"(r[84]), "=r"(r[85]), "=r"(r[86]), "=r"(r[87]), "=r"(r[88]), "=r"(r[89]), "=r"(r[90]), "=r"(r[91]), "=r"(r[92]), "=r"(r[93]), "=r"(r[94]), "=r"(r[95]), "=r"(r[96]), "=r"(r[97]), "=r"(r[98]), "=r"(r[99]), "=r"(r[100]), "=r"(r[101]), "=r"(r[102]), "=r"(r[103]), "=r"(r[104]), "=r"(r[105]), "=r"(r[106]), "=r"(r[107]), "=r"(r[108]), "=r"(r[109]), "=r"(r[110]), "=r"(r[111]), "=r"(r[112]), "=r"(r[113]), "=r"(r[114]), "=r"(r[115]), "=r"(r[116]), "=r"(r[117]), "=r"(r[118]), "=r"(r[119]), "=r"(r[120]), "=r"(r[121]), "=r"(r[122]), "=r"(r[123]), "=r"(r[124]), "=r"(r[125]), "=r"(r[126]), "=r"(r[127]) : "r"(addr), "r"(f), "r"(x) : "memory")
#define ST128F(addr, v, f, x) asm volatile("tcgen05.st.sync.aligned.32x32b.x128.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128}; bar.warp.sync 0xffffffff; st.volatile.shared.u32 [%129], %130;" :: "r"(addr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]), "r"(v[16]), "r"(v[17]), "r"(v[18]), "r"(v[19]), "r"(v[20]), "r"(v[21]), "r"(v[22]), "r"(v[23]), "r"(v[24]), "r"(v[25]), "r"(v[26]), "r"(v[27]), "r"(v[28]), "r"(v[29]), "r"(v[30]), "r"(v[31]), "r"(v[32]), "r"(v[33]), "r"(v[34]), "r"(v[35]), "r"(v[36]), "r"(v[37]), "r"(v[38]), "r"(v[39]), "r"(v[40]), "r"(v[41]), "r"(v[42]), "r"(v[43]), "r"(v[44]), "r"(v[45]), "r"(v[46]), "r"(v[47]), "r"(v[48]), "r"(v[49]), "r"(v[50]), "r"(v[51]), "r"(v[52]), "r"(v[53]), "r"(v[54]), "r"(v[55]), "r"(v[56]), "r"(v[57]), "r"(v[58]), "r"(v[59]), "r"(v[60]), "r"(v[61]), "r"(v[62]), "r"(v[63]), "r"(v[64]), "r"(v[65]), "r"(v[66]), "r"(v[67]), "r"(v[68]), "r"(v[69]), "r"(v[70]), "r"(v[71]), "r"(v[72]), "r"(v[73]), "r"(v[74]), "r"(v[75]), "r"(v[76]), "r"(v[77]), "r"(v[78]), "r"(v[79]), "r"(v[80]), "r"(v[81]), "r"(v[82]), "r"(v[83]), "r"(v[84]), "r"(v[85]), "r"(v[86]), "r"(v[87]), "r"(v[88]), "r"(v[89]), "r"(v[90]), "r"(v[91]), "r"(v[92]), "r"(v[93]), "r"(v[94]), "r"(v[95]), "r"(v[96]), "r"(v[97]), "r"(v[98]), "r"(v[99]), "r"(v[100]), "r"(v[101]), "r"(v[102]), "r"(v[103]), "r"(v[104]), "r"(v[105]), "r"(v[106]), "r"(v[107]), "r"(v[108]), "r"(v[109]), "r"(v[110]), "r"(v[111]), "r"(v[112]), "r"(v[113]), "r"(v[114]), "r"(v[115]), "r"(v[116]), "r"(v[117]), "r"(v[118]), "r"(v[119]), "r"(v[120]), "r"(v[121]), "r"(v[122]), "r"(v[123]), "r"(v[124]), "r"(v[125]), "r"(v[126]), "r"(v[127]), "r"(f), "r"(x) : "memory")
#define ST1(addr, x) asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(addr), "r"(x) : "memory")
#define LD1(addr, x) asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(x) : "r"(addr) : "memory")
#define WAITLD() asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory")
#define WAITST() asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory")
#define SYNC()                                              \
    do {                                                    \
        asm volatile("tcgen05.fence::before_thread_sync;"); \
        __syncthreads();                                    \
        asm volatile("tcgen05.fence::after_thread_sync;");  \
    } while (0)

extern "C" __global__ void __launch_bounds__(256) probe(int var, int iters) {
    __shared__ unsigned tslot;
    __shared__ unsigned flag[4];
    int tid = threadIdx.x, w = tid / 32, lane = tid % 32, q = w % 4;
    unsigned slot = (unsigned)__cvta_generic_to_shared(&tslot);
    if (tid < 4) flag[tid] = 0;
    if (w == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::"r"(slot));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    SYNC();
    unsigned tm = *(volatile unsigned*)&tslot;
    unsigned tD = tm + ((unsigned)(32 * q) << 16);  // warps w and w + 4 share lanes 32q..32q+31
    unsigned t127 = tD + 127;
    volatile unsigned* fl = &flag[q];
    unsigned fla = (unsigned)__cvta_generic_to_shared(&flag[q]);  // every lane stores the same value
    bool cross = var <= C0R;
    unsigned long long hit = 0, els = 0, checked = 0;
    for (int it = 0; it < iters; it++) {
        unsigned base = 0x40000000u | ((blockIdx.x * 7919u + it * 104729u + q * 31u + lane) & 0xfffffu) << 4;
        unsigned old = base | 1, nw = base | 2, a = base | 3;
        unsigned v[128], r[128];
        if (w < 4) {
#pragma unroll
            for (int j = 0; j < 128; j++) v[j] = old;
            ST128(tD, v);
            WAITST();
        }
        SYNC();
        unsigned got = 0, want = 0, alt = 0;
        bool check = false;
        if (cross) {
            if (w < 4) {
                if (var == C0) {
                    LD128F(tD, r, fla, it + 1);
                    WAITLD();
                    got = r[127]; want = old; alt = nw; check = true;
                } else {
#pragma unroll
                    for (int j = 0; j < 128; j++) v[j] = var == C0W ? a : nw;
                    ST128F(tD, v, fla, it + 1);
                    WAITST();
                }
            } else {
                while (*fl != (unsigned)(it + 1)) {}
                if (var == C0R) {
                    LD1(t127, got);
                    WAITLD();
                    want = nw; alt = old; check = true;
                } else {
                    ST1(t127, var == C0 ? nw : nw ^ 0x00f00000u);
                    WAITST();
                }
            }
            SYNC();
            if (var == C0W && w < 4) {
                LD1(t127, got);
                WAITLD();
                want = nw ^ 0x00f00000u; alt = a; check = true;
            }
        } else if (w < 4) {
            if (var == M0) {
                LD128(tD, r);
                ST1(t127, nw);
                WAITLD();
                WAITST();
                got = r[127]; want = old; alt = nw;
            } else if (var == M4) {
#pragma unroll
                for (int j = 0; j < 128; j++) v[j] = a;
                ST128(tD, v);
                ST1(t127, nw);
                WAITST();
                LD1(t127, got);
                WAITLD();
                want = nw; alt = a;
            } else {
#pragma unroll
                for (int j = 0; j < 128; j++) v[j] = nw;
                ST128(tD, v);
                LD1(t127, got);
                WAITLD();
                WAITST();
                want = nw; alt = old;
            }
            check = true;
        }
        if (check) {
            checked++;
            if (got == alt) hit++;
            else if (got != want) els++;
        }
        SYNC();
    }
    atomicAdd(&g_cnt[var][0], hit); atomicAdd(&g_cnt[var][1], els); atomicAdd(&g_cnt[var][2], checked);
    SYNC();
    if (w == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tm));
}

int main(int argc, char** argv) {
    int ctas = 148, iters = argc > 1 ? atoi(argv[1]) : 100, runs = argc > 2 ? atoi(argv[2]) : 1000;
    const char* name[NV] = {"C0  cross: LDTM.x128; partner STTM.x1", "C0w cross: STTM.x128 a; partner STTM.x1 b",
                            "C0r cross: STTM.x128; partner LDTM.x1", "M0' same: LDTM.x128; STTM.x1",
                            "M4' same: STTM.x128 a; STTM.x1 b", "M2' same: STTM.x128; LDTM.x1"};
    const char* what[NV] = {"load saw the later store", "long store landed last", "load missed the earlier store",
                            "load saw the later store", "stores swapped", "load missed the earlier store"};
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("%s; %d CTAs x 256 threads (warps 0-3 issue, 4-7 partners) x %d iterations x %d runs per variant\n",
           p.name, ctas, iters, runs);
    for (int v = 0; v < NV; v++)
        for (int k = 0; k < runs; k++) {
            probe<<<ctas, 256>>>(v, iters);
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) { printf("variant %d: %s\n", v, cudaGetErrorString(e)); return 1; }
        }
    unsigned long long c[NV][3];
    cudaMemcpyFromSymbol(c, g_cnt, sizeof c);
    for (int v = 0; v < NV; v++)
        printf("%-44s %llu words checked, %llu %s, %llu else\n", name[v], c[v][2], c[v][0], what[v], c[v][1]);
    return 0;
}
