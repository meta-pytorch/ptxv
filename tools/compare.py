# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Compare a candidate GPU kernel against a reference: capture both on the same inputs
# (tools/capture.py), then `ptxv compare` (execution verdicts, equivalence level, per-output
# bounds, the captured GPU outputs against the bound, and why anything is unproven).
#
#   source env.sh
#   $PTXV_PYTHON tools/compare.py --ref 'torch.softmax(x, -1)' \
#       --cand tools/capture_examples/triton_softmax.py:softmax_kernel \
#       --inputs 'x=randn(4,128,f32)' --out /tmp/cmp_softmax
#
# --ref / --cand each take one of:
#   * an existing case directory (has case.toml): used as is;
#   * FILE.py:KERNEL  - a Triton / CuTeDSL kernel (capture.py --triton), launched by the
#                       file's run(**inputs);
#   * FILE.py         - everything run(**inputs) launches (capture.py --file);
#   * anything else   - a torch expression over the inputs (capture.py --op).
# --inputs NAME=EXPR (capture.py's helpers: randn / rand / ... with a dtype) is given to both
# sides with the same --seed, so both captures read the same bytes. Without --inputs, a file's
# own inputs() is used.
#
# Captures run on one GPU (--gpu, default 3; GPUs in $PTXV_EXCLUDE_GPUS are refused). The comparison itself needs no
# GPU. Outputs: <out>/ref, <out>/cand (cases), <out>/report.md, <out>/report.json.
import argparse
import os
import shlex
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def kind(spec):
    if os.path.isfile(os.path.join(spec, "case.toml")):
        return "case"
    path = spec.rsplit(":", 1)[0]
    if path.endswith(".py") and os.path.isfile(path):
        return "triton" if ":" in spec else "file"
    return "op"


def capture(spec, out, a):
    k = kind(spec)
    if k == "case":
        return os.path.abspath(spec)
    cmd = [os.environ.get("PTXV_PYTHON", sys.executable), f"{ROOT}/tools/capture.py"]
    cmd += [f"--{k}", spec, "--out", out, "--seed", str(a.seed)]
    for i in a.inputs:
        cmd += ["--inputs", i]
    if k != "op":
        # The cubin the JIT built and the GPU ran (not ptxas's rebuild of its PTX).
        cmd.append("--binary")
    if a.kernels:
        cmd += ["--kernels", a.kernels]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    with open(out + ".capture.log", "w") as f:
        f.write(r.stdout + r.stderr)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
        sys.exit(f"capture of {spec!r} failed (log: {out}.capture.log):\n{tail}")
    return out


def main():
    ap = argparse.ArgumentParser(description="capture + ptxv compare")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--inputs", action="append", default=[], help="NAME=EXPR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--kernels", help="regex: keep only matching launches (both sides)")
    ap.add_argument("--max-states", type=int, default=200000)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--segments", choices=["true", "false"])
    ap.add_argument("--sweep-cache")
    ap.add_argument(
        "--function", help="elementwise: the real function (gelu, silu, ...)"
    )
    ap.add_argument(
        "--ptxv",
        default=os.path.join(
            os.environ.get("CARGO_TARGET_DIR", f"{ROOT}/target"), "release/ptxv"
        ),
    )
    a = ap.parse_args()
    if str(a.gpu) in os.environ.get("PTXV_EXCLUDE_GPUS", "").split(","):
        sys.exit(f"GPU {a.gpu} is in PTXV_EXCLUDE_GPUS")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.abspath(a.out)
    ref = capture(a.ref, f"{out}/ref", a)
    cand = capture(a.cand, f"{out}/cand", a)
    cmd = [a.ptxv, "compare", ref, cand, "--out", out]
    cmd += ["--max-states", str(a.max_states), "--trials", str(a.trials)]
    cmd += ["--ref-label", a.ref, "--cand-label", a.cand]
    if a.segments:
        cmd += ["--segments", a.segments]
    if a.sweep_cache:
        cmd += ["--sweep-cache", a.sweep_cache]
    if a.function:
        cmd += ["--function", a.function]
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    # The comparison runs on the CPU: no GPU is visible to it.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    sys.exit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    main()
