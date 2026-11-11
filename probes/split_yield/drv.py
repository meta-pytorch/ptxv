#!/usr/bin/env python3
"""Split-interleaving stress driver (CUDA driver API via ctypes) for kernel.ptx
(ptxas -arch=sm_103a kernel.ptx -o kernel.cubin).
   drv.py --dev D --n N --mode M --grid G --block B --hi M2 --runs R --tag T
Per run and warp: `strict_overlap` counts warps whose two paths' [start, end] intervals overlap
strictly (the measure); `interleaved` uses <= and also counts paths that start in the same
32 ns %globaltimer tick as the other one ends (`h0_equals_t1`), which are not interleavings."""

import argparse
import ctypes
import json
import random
import struct
import sys
import time

ap = argparse.ArgumentParser()
for k, d in [
    ("dev", 0),
    ("n", 2000),
    ("mode", 0),
    ("grid", 1),
    ("block", 32),
    ("hi", 64),
    ("runs", 10),
]:
    ap.add_argument("--" + k, type=int, default=d)
ap.add_argument("--tag", default="")
a = ap.parse_args()
c = ctypes.CDLL("libcuda.so.1")


def ck(r, w):
    if r != 0:
        raise SystemExit(f"{w}: CUDA error {r}")


ck(c.cuInit(0), "init")
dev = ctypes.c_int()
ck(c.cuDeviceGet(ctypes.byref(dev), a.dev), "dev")
ctx = ctypes.c_void_p()
ck(c.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "ctx")
mod = ctypes.c_void_p()
ck(c.cuModuleLoad(ctypes.byref(mod), b"kernel.cubin"), "mod")
fn = ctypes.c_void_p()
ck(c.cuModuleGetFunction(ctypes.byref(fn), mod, b"k"), "fn")
N = 1 << 14
stride = 4096
random.seed(1)
perm = list(range(N))
random.shuffle(perm)
buf = bytearray(N * stride)
for i in range(N):
    struct.pack_into("<Q", buf, perm[i] * stride, perm[(i + 1) % N] * stride)
dch = ctypes.c_uint64()
ck(c.cuMemAlloc_v2(ctypes.byref(dch), len(buf)), "alloc")
ck(c.cuMemcpyHtoD_v2(dch, (ctypes.c_char * len(buf)).from_buffer(buf), len(buf)), "h2d")
warps = a.grid * a.block // 32
osz = warps * 32
dout = ctypes.c_uint64()
ck(c.cuMemAlloc_v2(ctypes.byref(dout), osz), "alloc")
host = (ctypes.c_uint64 * (warps * 4))()
mask = N - 1
args = [
    ctypes.c_uint64(dout.value),
    ctypes.c_uint64(dch.value),
    ctypes.c_uint32(a.n),
    ctypes.c_uint32(a.mode),
    ctypes.c_uint32(a.hi),
    ctypes.c_uint32(mask),
]
pp = (ctypes.c_void_p * len(args))(
    *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in args]
)
obs = inter = bad = strict = eq = 0
margins = []
lo_ns = []
first_hi = 0
for r in range(a.runs):
    ck(c.cuMemsetD8_v2(dout, 0, osz), "memset")
    ck(c.cuLaunchKernel(fn, a.grid, 1, 1, a.block, 1, 1, 0, None, pp, None), "launch")
    ck(c.cuCtxSynchronize(), "sync")
    ck(c.cuMemcpyDtoH_v2(host, dout, osz), "d2h")
    for w in range(warps):
        t0, t1, h0, h1 = host[4 * w : 4 * w + 4]
        if not (t0 and t1 and h0 and h1):
            bad += 1
            continue
        obs += 1
        lo_ns.append(t1 - t0)
        if h0 <= t1 and h1 >= t0:
            inter += 1
            margins.append(t1 - h0)
            if h0 < t1 and h1 > t0:
                strict += 1
            if h0 == t1:
                eq += 1
        if h0 < t0:
            first_hi += 1
lo_ns.sort()
res = dict(
    tag=a.tag,
    dev=a.dev,
    n=a.n,
    mode=a.mode,
    grid=a.grid,
    block=a.block,
    hi=a.hi,
    runs=a.runs,
    observations=obs,
    interleaved=inter,
    strict_overlap=strict,
    h0_equals_t1=eq,
    max_overlap_ns=max(margins) if margins else None,
    incomplete=bad,
    other_path_first=first_hi,
    path_ns_min=lo_ns[0] if lo_ns else None,
    path_ns_median=lo_ns[len(lo_ns) // 2] if lo_ns else None,
    path_ns_max=lo_ns[-1] if lo_ns else None,
)
print(json.dumps(res))
