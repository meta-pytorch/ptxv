#!/usr/bin/env python3
"""Probe B2 under load (probes/bsync_coll/b2s.cu): runb2s.py WORK GRID BLOCK RUNS; per warp, did lane 0 pass
the join's BSYNC while lanes 16-31 waited at their WARPSYNC.COLLECTIVE? Build: nvcc -arch=sm_103a -cubin
-O3 b2s.cu -o b2s.cubin."""

import ctypes
import json
import sys

work, grid, block, runs = map(int, sys.argv[1:5])
c = ctypes.CDLL("libcuda.so.1")


def ck(r, w):
    if r:
        raise SystemExit(f"{w}: {r}")


ck(c.cuInit(0), "i")
d = ctypes.c_int()
ck(c.cuDeviceGet(ctypes.byref(d), 0), "d")
x = ctypes.c_void_p()
ck(c.cuCtxCreate_v2(ctypes.byref(x), 0, d), "c")
m = ctypes.c_void_p()
ck(c.cuModuleLoad(ctypes.byref(m), b"b2s.cubin"), "m")
f = ctypes.c_void_p()
ck(c.cuModuleGetFunction(ctypes.byref(f), m, b"b2s"), "f")
dc = ctypes.c_uint64()
ck(c.cuMemAlloc_v2(ctypes.byref(dc), 128), "a")
ck(c.cuMemsetD32_v2(dc, 1, 32), "s")
W = grid * block // 32
dt = ctypes.c_uint64()
ck(c.cuMemAlloc_v2(ctypes.byref(dt), W * 32), "a")
h = (ctypes.c_uint64 * (4 * W))()
args = [ctypes.c_uint64(dc.value), ctypes.c_uint64(dt.value), ctypes.c_int32(work)]
pp = (ctypes.c_void_p * 3)(
    *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args]
)
obs = state = released_first = other = 0
for r in range(runs):
    ck(c.cuMemsetD8_v2(dt, 0, W * 32), "s")
    ck(c.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, None, pp, None), "l")
    ck(c.cuCtxSynchronize(), "y")
    ck(c.cuMemcpyDtoH_v2(h, dt, W * 32), "h")
    for w in range(W):
        t0, t1, t2, t3 = h[4 * w : 4 * w + 4]
        if not (t0 and t1 and t2 and t3):
            continue
        obs += 1
        # the state of interest: lane 16 parked (arrived) before lane 0 is past the BSYNC,
        # and lane 0 past the BSYNC strictly before lane 16 leaves its WARPSYNC.COLLECTIVE
        if t3 < t1 and t1 < t2:
            state += 1
        elif t2 <= t1:
            released_first += 1
        else:
            other += 1
print(
    json.dumps(
        dict(
            work=work,
            grid=grid,
            block=block,
            runs=runs,
            completed_warps=obs,
            bsync_passed_while_collective_parked=state,
            collective_left_first=released_first,
            other=other,
        )
    )
)
