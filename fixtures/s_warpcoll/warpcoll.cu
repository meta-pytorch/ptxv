/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Warp-collective SASS forms of torch.topk's gatherTopK, checked against the simulator bitwise
// (`ptxv hw fixtures/s_warpcoll`): REDUX.OR / .AND / .XOR / .MIN / .MAX / .SUM (uniform result,
// any active subset), SR_LEMASK / GEMASK / GTMASK / EQMASK / LTMASK, and ptxas's collective
// lowering of `vote.sync.ballot` / `redux.sync` with a run-time member mask:
//   VOTE.ANY Rm, ...; REDUX.OR URa, Rm; R2UR P, URZ, Rm; @!P BRA.DIV URa, slow
//   fast: the collective op            slow: WARPSYNC.COLLECTIVE Rm, end; op; ENDCOLLECTIVE
// One CTA of 128 threads (4 warps). Each lane's data word x = rnd[t] picks its group in the
// divergent sections, so every warp runs a different, data-dependent set of active masks.
// nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin warpcoll.cu
#include <stdint.h>

#define NOUT 40

__device__ __forceinline__ uint32_t sreg_le() { uint32_t r; asm volatile("mov.u32 %0, %%lanemask_le;" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t sreg_ge() { uint32_t r; asm volatile("mov.u32 %0, %%lanemask_ge;" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t sreg_gt() { uint32_t r; asm volatile("mov.u32 %0, %%lanemask_gt;" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t sreg_eq() { uint32_t r; asm volatile("mov.u32 %0, %%lanemask_eq;" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t sreg_lt() { uint32_t r; asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(r)); return r; }

// The six redux.sync forms over member mask m.
__device__ __forceinline__ void reduxes(uint32_t m, uint32_t x, uint32_t* o) {
    o[0] = __reduce_or_sync(m, x);
    o[1] = __reduce_and_sync(m, x);
    o[2] = __reduce_xor_sync(m, x);
    o[3] = __reduce_min_sync(m, x);
    o[4] = __reduce_max_sync(m, x);
    o[5] = (uint32_t)__reduce_min_sync(m, (int)x);
    o[6] = (uint32_t)__reduce_max_sync(m, (int)x);
    o[7] = __reduce_add_sync(m, x);
}

__global__ void warpcoll(const uint32_t* rnd, const uint32_t* masks, const uint32_t* lim,
                         uint32_t* out) {
    uint32_t t = threadIdx.x, lane = t & 31, w = t >> 5;
    uint32_t x = rnd[t];
    uint32_t* o = out + t * NOUT;
    o[0] = sreg_le();
    o[1] = sreg_ge();
    o[2] = sreg_gt();
    o[3] = sreg_eq();
    o[4] = sreg_lt();
    // Full warp (constant mask).
    reduxes(0xffffffffu, x, o + 5);
    // Divergent: four data-dependent groups, each reducing over its own active mask.
    uint32_t g = (x >> 7) & 3;
    if (g == 0) {
        uint32_t m = __activemask();
        reduxes(m, x, o + 13);
    } else if (g == 1) {
        uint32_t m = __activemask();
        reduxes(m, x ^ 0x5a5a5a5au, o + 13);
    } else if (g == 2) {
        uint32_t m = __activemask();
        o[13] = __reduce_or_sync(m, x);
        o[14] = __ballot_sync(m, x & 1);
    } else {
        o[13] = __activemask();
    }
    // Run-time member mask (masks[w] = 0xffffffff, unknown to ptxas): collective lowering.
    uint32_t mm = masks[w];
    o[21] = __ballot_sync(mm, x & 2);
    o[22] = __reduce_or_sync(mm, x);
    // gatherTopK's loop: the member mask is the ballot of the lanes still in the loop.
    uint32_t n = lim[w], acc = 0, cnt = 0;
    uint32_t mask = __ballot_sync(0xffffffffu, lane < n);
    for (uint32_t i = lane; i < n; i += 7) {
        uint32_t v = rnd[(i * 13 + w) & 127];
        uint32_t b = __ballot_sync(mask, v & 4);
        acc ^= b;
        cnt += __popc(b & sreg_le());
        mask = __ballot_sync(mask, i + 7 < n);
    }
    o[23] = acc;
    o[24] = cnt;
    // Legal non-uniform member masks: half-warp groups, each lane naming its own half.
    if (lane < 32) {
        uint32_t hm = masks[4 + (lane >> 4)];  // 0x0000ffff / 0xffff0000
        o[25] = __ballot_sync(hm, x & 8);
        o[26] = __reduce_or_sync(hm, x);
    }
    // Divergent groups again, each with a run-time mask equal to its active set.
    if (g & 1) {
        uint32_t m = __activemask() & masks[w];
        o[27] = __ballot_sync(m, x & 16);
        o[28] = __reduce_or_sync(m, x);
    } else {
        uint32_t m = __activemask() & masks[w];
        o[27] = __ballot_sync(m, x & 32);
        o[28] = __reduce_add_sync(m, x);
    }
    o[29] = sreg_le() & mask;
    // Three groups of legal non-uniform member masks (lanes 0-7, 8-15, 16-31).
    {
        uint32_t gm = masks[6 + (lane >> 3 > 2 ? 2 : lane >> 3)];  // 0xff / 0xff00 / 0xffff0000
        o[30] = __ballot_sync(gm, x & 64);
        o[31] = __reduce_add_sync(gm, x);
    }
}
