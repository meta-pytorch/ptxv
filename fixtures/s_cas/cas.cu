/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SASS compare-and-swap against GB300 bitwise (rung 4a: torch symmetric memory's signals are
// `ATOM.E.CAS.STRONG.SYS` loops). Thread i owns word mem[i] and gmem[i]:
//   o[0] = atomicCAS_system(&w[i], c, n)  with c = w's value (swaps) for odd i, c = w ^ (bits)
//          (fails) for even i, n random;  o[1] = atomicCAS(&gmem[i], ...) at GPU scope, same rule with the
//          parity flipped.
// The words are reached through a device pointer table (`ptrs`: mem, flags, payload; the
// case's buffer `ptrs`), so the generic forms are used, as in torch: ATOM.E.CAS. o[1] uses the
// global pointer `gmem` (ATOMG.E.CAS.STRONG.GPU).
// Then a CAS signal handoff, as torch's sync_remote_blocks: thread i (< 64) puts with
// `while (cas(&f[i], 0, 1) != 0)`, thread i + 64 waits with `while (cas(&f[i], 1, 0) != 1)` and
// writes o[2] = the payload the putter stored before its put (MEMBAR.SYS release before the
// put, CCTL.IVALL acquire after the wait). mem and gmem are outputs too.
//   nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin cas.cu
#include <stdint.h>
#define N 128

__device__ __forceinline__ uint32_t cas_sys_acq_rel(uint32_t* a, uint32_t c, uint32_t n) {
    uint32_t old;
    asm volatile("atom.acq_rel.sys.cas.b32 %0, [%1], %2, %3;"
                 : "=r"(old) : "l"(a), "r"(c), "r"(n) : "memory");
    return old;
}

extern "C" __global__ void __launch_bounds__(N)
cas(const uint32_t* __restrict__ rnd, uint32_t* const* ptrs, uint32_t* gmem,
    uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t* mem = ptrs[0];
    uint32_t* flags = ptrs[1];
    uint32_t* payload = ptrs[2];
    uint32_t* o = out + i * 4;
    uint32_t a = rnd[3 * i], b = rnd[3 * i + 1], x = rnd[3 * i + 2] | 1;
    uint32_t c0 = (i & 1) ? mem[i] : mem[i] ^ x;
    o[0] = atomicCAS_system((unsigned int*)&mem[i], c0, a);
    uint32_t c1 = (i & 1) ? gmem[i] ^ x : gmem[i];
    o[1] = atomicCAS((unsigned int*)&gmem[i], c1, b);
    if (i < 64) {
        payload[i] = a ^ b;
        while (cas_sys_acq_rel(&flags[i], 0, 1) != 0) {
        }
    } else {
        while (cas_sys_acq_rel(&flags[i - 64], 1, 0) != 1) {
        }
        o[2] = payload[i - 64];
    }
    o[3] = 0;
}
