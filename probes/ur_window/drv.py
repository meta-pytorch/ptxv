#!/usr/bin/env python3
"""Driver for probes/ur_window (CUDA driver API via ctypes; kernel.cubin from
`ptxas -arch=sm_103a kernel.ptx -o kernel.cubin`).
   drv.py --dev D --n N --grid G --block B --runs R
Per run and warp: the even and the odd path's [t0, t1] (%globaltimer); `strict_overlap`
counts warps whose two intervals overlap strictly (max start < min end)."""

import argparse
import ctypes
import json
import random
import struct

ap = argparse.ArgumentParser()
for k, d in [("dev", 0), ("n", 64), ("grid", 1), ("block", 32), ("runs", 10)]:
    ap.add_argument("--" + k, type=int, default=d)
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
threads = a.grid * a.block
warps = threads // 32
osz = warps * 32
dout = ctypes.c_uint64()
ck(c.cuMemAlloc_v2(ctypes.byref(dout), osz), "alloc")
dsink = ctypes.c_uint64()
ck(c.cuMemAlloc_v2(ctypes.byref(dsink), threads * 8), "alloc")
host = (ctypes.c_uint64 * (warps * 4))()
args = [
    ctypes.c_uint64(dout.value),
    ctypes.c_uint64(dch.value),
    ctypes.c_uint64(dsink.value),
    ctypes.c_uint32(a.n),
    ctypes.c_uint32(N - 1),
]
pp = (ctypes.c_void_p * len(args))(
    *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in args]
)
obs = strict = bad = 0
lens = []
for r in range(a.runs):
    ck(c.cuMemsetD8_v2(dout, 0, osz), "memset")
    ck(c.cuLaunchKernel(fn, a.grid, 1, 1, a.block, 1, 1, 0, None, pp, None), "launch")
    ck(c.cuCtxSynchronize(), "sync")
    ck(c.cuMemcpyDtoH_v2(host, dout, osz), "d2h")
    for w in range(warps):
        e0, e1, o0, o1 = host[4 * w : 4 * w + 4]
        if not (e0 and e1 and o0 and o1):
            bad += 1
            continue
        obs += 1
        lens.append(e1 - e0)
        lens.append(o1 - o0)
        if max(e0, o0) < min(e1, o1):
            strict += 1
lens.sort()
print(
    json.dumps(
        dict(
            dev=a.dev,
            n=a.n,
            grid=a.grid,
            block=a.block,
            runs=a.runs,
            observations=obs,
            strict_overlap=strict,
            incomplete=bad,
            path_ns_min=lens[0] if lens else None,
            path_ns_median=lens[len(lens) // 2] if lens else None,
            path_ns_max=lens[-1] if lens else None,
        )
    )
)
