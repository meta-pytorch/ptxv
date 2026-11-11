/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CAS semaphores (E-WELLSYNC C-CAS): torch symmetric memory's signal shape
// (`while (cas(&w, 0, 1) != 0);` put, `while (cas(&w, 1, 0) != 1);` wait, atom.acq_rel.sys:
// MEMBAR.SYS / ATOM.E.CAS / CCTL.IVALL), and patterns the certificate must refuse.
//   nvcc -cubin -arch=sm_103a -O3 -o kernel.cubin cas_sem.cu
// Entry points (same parameters: words w[64], payload p[32], out[96]):
//   cas_sem        (the case's entry) three rounds of a full/empty CAS handoff between lane
//                  i of warp 0 and lane i of warp 1: every word is put, waited, put, ...
//                  Clean, deterministic, certified.
//   cas_lock       a CAS spin lock taken by two threads: correct, but which thread takes the
//                  lock first depends on the schedule (two unordered 0 -> 1 CAS polls).
//   cas_two_kinds  two putters of different values compete for the empty word (0 -> 1 and
//                  0 -> 2); the waiter wants 1 first: a deadlock when 2 goes first.
//   cas_exit_fail  the second put's loop exits on a failed compare too (`!= 1` written as
//                  `== 1`): it may see 2 and leave without writing (the final word differs).
//   cas_store      a third thread stores 0 to the signal word (st.relaxed.sys): the wait may
//                  never see the put.
#include <stdint.h>

__device__ __forceinline__ uint32_t cas(uint32_t* a, uint32_t c, uint32_t n) {
    uint32_t old;
    asm volatile("atom.acq_rel.sys.cas.b32 %0, [%1], %2, %3;"
                 : "=r"(old) : "l"(a), "r"(c), "r"(n) : "memory");
    return old;
}

extern "C" __global__ void __launch_bounds__(96)
cas_sem(uint32_t* w, uint32_t* p, uint32_t* out) {
    int i = threadIdx.x;
    if (i < 32) {
#pragma unroll 1
        for (int r = 0; r < 3; r++) {
            p[i] = i * 7 + r;
            while (cas(&w[i], 0, 1) != 0) {
            }
            while (cas(&w[32 + i], 1, 0) != 1) {
            }
        }
    } else if (i < 64) {
        int j = i - 32;
        uint32_t s = 0;
#pragma unroll 1
        for (int r = 0; r < 3; r++) {
            while (cas(&w[j], 1, 0) != 1) {
            }
            s = s * 31 + p[j];
            while (cas(&w[32 + j], 0, 1) != 0) {
            }
        }
        out[j] = s;
    }
}

extern "C" __global__ void __launch_bounds__(96)
cas_lock(uint32_t* w, uint32_t* p, uint32_t* out) {
    int i = threadIdx.x;
    if (i == 0 || i == 32) {
        while (cas(&w[0], 0, 1) != 0) {
        }
        p[0] = p[0] * 3 + i;
        while (cas(&w[0], 1, 0) != 1) {
        }
        out[i] = 1;
    }
}

extern "C" __global__ void __launch_bounds__(96)
cas_two_kinds(uint32_t* w, uint32_t* p, uint32_t* out) {
    int i = threadIdx.x;
    if (i == 0) {
        while (cas(&w[0], 0, 1) != 0) {
        }
        out[i] = 1;
    } else if (i == 32) {
        while (cas(&w[0], 0, 2) != 0) {
        }
        out[i] = 1;
    } else if (i == 64) {
        while (cas(&w[0], 1, 0) != 1) {
        }
        while (cas(&w[0], 2, 0) != 2) {
        }
        out[i] = 1;
    }
}

extern "C" __global__ void __launch_bounds__(96)
cas_exit_fail(uint32_t* w, uint32_t* p, uint32_t* out) {
    int i = threadIdx.x;
    if (i == 0) {
        while (cas(&w[0], 0, 1) != 0) {
        }
        while (cas(&w[0], 0, 1) == 1) {
        }
        out[i] = 1;
    } else if (i == 32) {
        while (cas(&w[0], 1, 2) != 1) {
        }
        while (cas(&w[0], 2, 0) != 2) {
        }
        out[i] = 1;
    }
}

extern "C" __global__ void __launch_bounds__(96)
cas_store(uint32_t* w, uint32_t* p, uint32_t* out) {
    int i = threadIdx.x;
    if (i == 0) {
        while (cas(&w[0], 0, 1) != 0) {
        }
        out[i] = 1;
    } else if (i == 32) {
        while (cas(&w[0], 1, 0) != 1) {
        }
        out[i] = 1;
    } else if (i == 64) {
        asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(&w[0]), "r"(0u) : "memory");
    }
}
