# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Capture the cuBLAS kernel torch uses for C[M,N] f32 = A[M,K] bf16 @ B[N,K]^T (binary only).
#   CUDA_VISIBLE_DEVICES=0 $PTXV_PYTHON capture.py [M N K]
# Runs torch.mm(a, b.t(), out_dtype=float32) under probes/launchcap with image capture, finds
# the code image that defines the launched kernel (kernel.cubin), and writes case.toml. cuBLAS
# passes its arguments as one packed block (CU_LAUNCH_PARAM_BUFFER_POINTER); the pointers and
# tensor maps inside it become `fields` of a `bytes` parameter, matched by address.
import glob
import json
import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LC = os.path.join(HERE, "../../probes/launchcap")
CAP = "/tmp/ptxv_launchcap_cublas.jsonl"
IMG = "/tmp/ptxv_launchcap_cublas_images"
if "LD_PRELOAD" not in os.environ:  # re-exec under the launch recorder
    so = "/tmp/ptxv_launchcap.so"
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            "-o",
            so,
            f"{LC}/launchcap.c",
            "-ldl",
            "-I/usr/local/cuda-13.0/include",
        ],
        check=True,
    )
    for p in (CAP,):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(IMG, ignore_errors=True)
    os.makedirs(IMG)
    env = dict(os.environ, LD_PRELOAD=so, LAUNCHCAP_OUT=CAP, LAUNCHCAP_IMAGES=IMG)
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)
os.chdir(HERE)

import torch

M, N, K = (int(x) for x in (sys.argv[1:4] if len(sys.argv) > 3 else (512, 128, 128)))


# The case's deterministic buffer contents (crates/case: SplitMix64 + Box-Muller, bf16 RN), so
# the GPU runs here match the simulator's inputs.
def bf16_buffer(rows, cols, seed):
    # Not needed for the capture itself (the kernel choice does not depend on values).
    return torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")


a = bf16_buffer(M, K, 1)
b = bf16_buffer(N, K, 2)
torch.cuda.synchronize()
c = torch.mm(a, b.t(), out_dtype=torch.float32)
torch.cuda.synchronize()
ref = a.float() @ b.float().t()
print("max err", (c - ref).abs().max().item())

recs = [json.loads(ln) for ln in open(CAP)]
recs = [r for r in recs if r["params"] and r["params"][0].get("packed")]
assert len(recs) == 1, [r["name"] for r in recs]
rec = recs[0]
name = rec["name"]
img = [p for p in glob.glob(f"{IMG}/*.bin") if name.encode() in open(p, "rb").read()]
assert img, f"no captured image defines {name}"
# The smallest image with the name is the kernel's own cubin.
img = min(img, key=os.path.getsize)
assert open(img, "rb").read(4) == b"\x7fELF", "image is not a cubin"
shutil.copy(img, "kernel.cubin")

blob = bytes.fromhex(rec["params"][0]["hex"])
ptrs = {a.data_ptr(): ("A", a), b.data_ptr(): ("B", b), c.data_ptr(): ("C", c)}


def decode_tmap(raw, t, bname):
    """Geometry of a captured CUtensorMap, read from the descriptor (layout learned from
    cuTensorMapEncodeTiled outputs and the CuTeDSL captures): word 2 bits 4-6 = rank - 1,
    bits 13-14 = swizzle; bytes 12..15 / 16..19 = byte strides of dims 1 / 2, >> 4; u32 at
    0x20 + 4i = dim i - 1; byte 0x37 + i = box i - 1."""
    w2 = struct.unpack_from("<I", raw, 8)[0]
    rank = ((w2 >> 4) & 7) + 1
    swz = {0: "none", 0x2000: "32B", 0x4000: "64B", 0x6000: "128B"}[w2 & 0x6000]
    dims = [struct.unpack_from("<I", raw, 0x20 + 4 * i)[0] + 1 for i in range(rank)]
    box = [raw[0x37 + i] + 1 for i in range(rank)]
    es = t.element_size()
    strides = []
    for i in range(1, rank):
        s = struct.unpack_from("<I", raw, 8 + 4 * i)[0] << 4
        # A dimension of extent 1 is never stepped: keep its stride consistent.
        strides.append(
            s if dims[i] > 1 else dims[i - 1] * (strides[-1] if strides else es)
        )
    assert dims[0] * es == strides[0] or dims[0] * es <= strides[0], (dims, strides)
    assert dims[1] * dims[0] == t.numel(), (bname, dims, t.shape)
    return {
        "buffer": bname,
        "box": box,
        "dims": dims,
        "strides": strides,
        "swizzle": swz,
        "raw": raw[:64].hex(),
    }  # maps may sit 64 bytes apart in the block


fields = []
for off in range(0, len(blob) - 7, 8):
    v = struct.unpack_from("<Q", blob, off)[0]
    if v not in ptrs:
        continue
    bname, t = ptrs[v]
    # A tensor map starts with the global address and is 64-byte aligned; anything else that
    # holds the address is a plain pointer.
    if (
        off % 64 == 0
        and off + 128 <= len(blob)
        and blob[off + 8 : off + 16] != bytes(8)
    ):
        fields.append((off, "tensormap", decode_tmap(blob[off : off + 128], t, bname)))
        continue
    fields.append((off, "ptr", {"buffer": bname}))
print("fields:", [(o, k, f["buffer"]) for o, k, f in fields])

out = [
    f'''name = "r2_cublas_gemm"
description = """
Rung 2e. The cuBLAS kernel torch {torch.__version__} picks for C[{M}x{N}] f32 = A[{M}x{K}] bf16 @
B[{N}x{K}]^T (torch.mm(a, b.t(), out_dtype=float32)): {name}. Binary only (cubin from the
loaded library image, probes/launchcap). Captured by capture.py."""
cubin = "kernel.cubin"
entry = "{name}"
grid = {rec["grid"]}
block = {rec["block"]}
cluster = {rec["cluster"]}
dynamic_smem = {rec["smem"]}
outputs = ["C"]
spec = "gemm_tn"
'''
]
for n, dt, shape, init, seed in (
    ("A", "bf16", [M, K], "normal", 1),
    ("B", "bf16", [N, K], "normal", 2),
    ("C", "f32", [M, N], "zeros", 0),
):
    out.append(
        f'[[buffers]]\nname = "{n}"\ndtype = "{dt}"\nshape = {shape}\ninit = "{init}"\nseed = {seed}\n'
    )
out.append(f'[[params]]\nkind = "bytes"\nhex = "{blob.hex()}"\n')
for off, kind, f in fields:
    s = f'[[params.fields]]\noffset = {off}\nkind = "{kind}"\nbuffer = "{f["buffer"]}"\n'
    if kind == "tensormap":
        s += (
            f"box = {f['box']}\ndims = {f['dims']}\nstrides = {f['strides']}\n"
            f'swizzle = "{f["swizzle"]}"\nraw = "{f["raw"]}"\n'
        )
    out.append(s)
out.append(
    '[[variants]]\nid = "good"\nsummary = "as shipped in cuBLAS"\nexpect = ["clean"]\n'
)
open("case.toml", "w").write("\n".join(out))
print("captured", name, rec["grid"], rec["block"], rec["cluster"], rec["smem"])
