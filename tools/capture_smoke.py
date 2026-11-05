# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Batch smoke test of tools/capture.py: capture each op, then ask ptxv whether it lifts
# (PTXV_LIFT_REPORT=1: the SASS instructions it does not lower), the verdict of `check`, and
# whether the simulator's output equals the captured GPU output bitwise. Writes
# <out>/results.json and prints a markdown table (docs/capture.md).
#   source env.sh; CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture_smoke.py [--out DIR]
#       [--only REGEX] [--ptxv PATH] [--max-states N] [--timeout S]
# ptxv runs under systemd-run with a 32 GB memory cap.
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EX = os.path.join(ROOT, "tools/capture_examples")

# name -> capture.py arguments
OPS = [
    ("softmax", ["--op", "torch.softmax(x, -1)", "--inputs", "x=randn(4,128,bf16)"]),
    (
        "log_softmax",
        ["--op", "torch.log_softmax(x, -1)", "--inputs", "x=randn(4,128,bf16)"],
    ),
    (
        "layer_norm",
        ["--op", "F.layer_norm(x, (128,), w, b)"]
        + ["--inputs", "x=randn(4,128,bf16),w=randn(128,bf16),b=randn(128,bf16)"],
    ),
    (
        "rms_norm",
        [
            "--op",
            "F.rms_norm(x, (128,), w)",
            "--inputs",
            "x=randn(4,128,bf16),w=randn(128,bf16)",
        ],
    ),
    ("sum_dim", ["--op", "x.sum(-1)", "--inputs", "x=randn(4,128,f32)"]),
    ("max_dim", ["--op", "x.amax(-1)", "--inputs", "x=randn(4,128,f32)"]),
    ("mean_dim", ["--op", "x.mean(-1)", "--inputs", "x=randn(4,128,f32)"]),
    ("cumsum", ["--op", "torch.cumsum(x, -1)", "--inputs", "x=randn(4,128,f32)"]),
    ("gelu", ["--op", "F.gelu(x)", "--inputs", "x=randn(4,128,bf16)"]),
    ("silu", ["--op", "F.silu(x)", "--inputs", "x=randn(4,128,bf16)"]),
    (
        "sdpa",
        ["--op", "F.scaled_dot_product_attention(q, k, v)"]
        + [
            "--inputs",
            "q=randn(1,2,64,64,bf16),k=randn(1,2,64,64,bf16),v=randn(1,2,64,64,bf16)",
        ],
    ),
    (
        "embedding",
        [
            "--op",
            "F.embedding(i, w)",
            "--inputs",
            "i=randint(0,64,16),w=randn(64,32,f32)",
        ],
    ),
    (
        "index_add",
        ["--op", "x.index_add(0, i, s)"]
        + ["--inputs", "x=zeros(16,32,f32),i=randint(0,16,8),s=randn(8,32,f32)"],
    ),
    (
        "matmul_bf16",
        ["--op", "a @ b", "--inputs", "a=randn(64,64,bf16),b=randn(64,64,bf16)"],
    ),
    # Several kernels from different modules (cuBLAS + libtorch_cuda).
    (
        "matmul_relu",
        [
            "--op",
            "torch.relu(a @ b)",
            "--inputs",
            "a=randn(64,64,bf16),b=randn(64,64,bf16)",
        ],
    ),
    ("sort", ["--op", "torch.sort(x, -1)", "--inputs", "x=randn(4,64,f32)"]),
    ("triton_softmax", ["--triton", f"{EX}/triton_softmax.py:softmax_kernel"]),
    ("triton_layernorm", ["--triton", f"{EX}/triton_layernorm.py:layernorm_kernel"]),
    ("triton_cumsum", ["--triton", f"{EX}/triton_cumsum.py:cumsum_kernel"]),
    ("triton_cat", ["--triton", f"{EX}/triton_cat.py:cat_kernel"]),
    ("cute_add", ["--file", f"{EX}/cute_add.py"]),
    # cuBLAS GEMM + libtorch copy from the middle of an allocation + Triton: three modules.
    ("mixed_3_modules", ["--file", f"{EX}/mixed.py"]),
    # Simple integer / move-only kernels, to find what lifts today.
    ("add_f32", ["--op", "x + y", "--inputs", "x=randn(4,128,f32),y=randn(4,128,f32)"]),
    (
        "transpose_copy",
        ["--op", "x.t().contiguous()", "--inputs", "x=randn(64,32,f32)"],
    ),
    (
        "index_select",
        [
            "--op",
            "x.index_select(0, i)",
            "--inputs",
            "x=randn(16,32,f32),i=randint(0,16,8)",
        ],
    ),
    (
        "cat",
        [
            "--op",
            "torch.cat([x, y], 0)",
            "--inputs",
            "x=randn(8,32,f32),y=randn(8,32,f32)",
        ],
    ),
    ("mul_scalar_f32", ["--op", "x * 2.5", "--inputs", "x=randn(4,128,f32)"]),
    (
        "where",
        [
            "--op",
            "torch.where(x > 0, x, y)",
            "--inputs",
            "x=randn(4,128,f32),y=randn(4,128,f32)",
        ],
    ),
    ("to_bf16", ["--op", "x.to(torch.bfloat16)", "--inputs", "x=randn(4,128,f32)"]),
    ("flip", ["--op", "x.flip(0)", "--inputs", "x=randn(16,32,f32)"]),
    (
        "gather",
        [
            "--op",
            "torch.gather(x, 1, i)",
            "--inputs",
            "x=randn(8,32,f32),i=randint(0,32,8,4)",
        ],
    ),
    ("fill", ["--op", "torch.full_like(x, 3.0)", "--inputs", "x=randn(4,128,f32)"]),
    (
        "add_i32",
        [
            "--op",
            "x + y",
            "--inputs",
            "x=randint(0,100,256,i32),y=randint(0,100,256,i32)",
        ],
    ),
]


def sh(cmd, env=None, timeout=None):
    try:
        r = subprocess.run(
            cmd, env=env, timeout=timeout, capture_output=True, text=True, cwd=ROOT
        )
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return "timeout", (e.stdout or b"").decode() + (e.stderr or b"").decode()


def capped(cmd):
    return (
        ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=32G"]
        + [
            "-p",
            "MemorySwapMax=0",
        ]
        + cmd
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ptxv_capture_smoke")
    ap.add_argument("--only")
    ap.add_argument(
        "--ptxv",
        default=os.path.join(os.environ.get("CARGO_TARGET_DIR", ""), "release/ptxv"),
    )
    ap.add_argument("--max-states", type=int, default=200000)
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res_path = f"{a.out}/results.json"
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for name, args in OPS:
        if a.only and not re.search(a.only, name):
            continue
        case = f"{a.out}/{name}"
        r = {"name": name, "args": args}
        code, log = sh(
            [sys.executable, f"{ROOT}/tools/capture.py", "--out", case] + args,
            timeout=600,
        )
        open(f"{a.out}/{name}.capture.log", "w").write(log)
        r["capture"] = code == 0
        if code != 0:
            r["capture_error"] = (log.strip().splitlines() or ["?"])[-1][:200]
            results[name] = r
            print(name, "capture failed:", r["capture_error"])
            continue
        meta = json.load(open(f"{case}/capture.json"))
        r["kernels"] = [la["name"] for la in meta["launches"]]
        r["cubins"] = sorted({la["cubin"] for la in meta["launches"]})
        r["ptx"] = meta["ptx"]
        r["warnings"] = meta["warnings"]
        r["unresolved"] = len(meta["unresolved"])
        # SASS front (the captured cubins, or ptxas's for a PTX case) with PTXV_LIFT_REPORT=1:
        # every launch's kernel reports what it does not lower, and a fully lowered case goes
        # on to the check.
        r["fronts"] = {}
        for f in ["sass"] + (["ptx"] if meta["ptx"] else []):
            env = dict(os.environ, PTXV_LIFT_REPORT="1")
            code, log = sh(
                capped(
                    [a.ptxv, "check", case, "--front", f, "--certify"]
                    + ["--max-states", str(a.max_states), "--out", f"{case}.out"]
                ),
                env=env,
                timeout=a.timeout,
            )
            open(f"{a.out}/{name}.{f}.check.log", "w").write(log)
            fr = {}
            if f == "sass":
                reps = re.findall(r"^lift report for (\S+): (\d+) kinds", log, re.M)
                miss = []
                for _n, w in re.findall(r"^\s*(\d+) (.+)$", log, re.M):
                    if not w.startswith("e.g.") and w not in miss:
                        miss.append(w)
                r["missing"] = miss
                r["lifted_kernels"] = sum(1 for _, n in reps if n == "0")
                r["lifts"] = len(reps) == len(r["kernels"]) and r[
                    "lifted_kernels"
                ] == len(reps)
            m = re.search(
                r"^(PASS|FAIL) \S+\s+\S+\s+(\[[^\]]*\]).*?outputs=(\d+)\s*(.*)$",
                log,
                re.M,
            )
            if m:
                fr["verdict"] = m.group(2)
                fr["complete"] = "INCOMPLETE" not in m.group(4)
                fr["distinct_outputs"] = int(m.group(3))
            elif f == "sass" and not r["lifts"]:
                fr["lifts"] = False
            else:
                panic = re.search(r"panicked at (\S+):\n(.*)", log)
                fr["error"] = (
                    "timeout"
                    if code == "timeout"
                    else (
                        f"panic at {panic.group(1)}: {panic.group(2)}"
                        if panic
                        else (log.strip().splitlines() or ["?"])[-1].strip()[:240]
                    )
                )
            ex = re.findall(r"expect \(output \d+\): (.*)$", log, re.M)
            fr["bitwise"] = None if not ex else all("bitwise equal" in e for e in ex)
            fr["expect"] = ex[:4]
            r["fronts"][f] = fr
        results[name] = r
        print(name, json.dumps({k: r[k] for k in ("lifts", "fronts")}))
        json.dump(results, open(res_path, "w"), indent=1)
    json.dump(results, open(res_path, "w"), indent=1)
    print(table(results))


def short(k):
    """Demangled kernel name without template arguments or parameters."""
    try:
        k = (
            subprocess.run(
                ["c++filt", k], capture_output=True, text=True
            ).stdout.strip()
            or k
        )
    except OSError:
        pass
    k = k.replace("(anonymous namespace)::", "")
    depth, out = 0, ""
    for ch in k:
        if ch in "<(" and out:
            if depth == 0 and ch == "(":
                break
            depth += 1
        elif ch in ">)" and depth:
            depth -= 1
        elif depth == 0:
            out += ch
    out = out.split(" ")[-1]  # return type of a template function
    return out[:60]


def norm_missing(ws):
    """One entry per kind: register numbers and operand modifiers folded."""
    out = []
    for w in ws:
        w = w.replace("unsupported SASS opcode ", "").strip("`")
        w = re.sub(r"\bU?R\d+(\.\w+)*", "R", w)
        w = re.sub(r"\bU?P\d\b", "P", w)
        if w not in out:
            out.append(w)
    return out


def table(results):
    rows = [
        "| op | kernels (cubins) | lifts (SASS) | missing SASS instructions | verdict | sim = GPU bitwise |",
        "|---|---|---|---|---|---|",
    ]
    for name, _ in OPS:
        r = results.get(name)
        if not r:
            continue
        if not r["capture"]:
            rows.append(f"| {name} | capture failed: {r['capture_error']} | | | | |")
            continue
        ks = "<br>".join(short(k) for k in r["kernels"]) + f" ({len(r['cubins'])})"
        miss = ", ".join(f"`{m}`" for m in norm_missing(r["missing"]))
        verdicts, bits = [], []
        for f, fr in r["fronts"].items():
            if "verdict" in fr:
                verdicts.append(
                    f"{f}: {fr['verdict']}"
                    + ("" if fr["complete"] else " (incomplete)")
                )
            elif "error" in fr:
                verdicts.append(f"{f}: error: {fr['error'][:120]}")
            if fr.get("bitwise") is not None:
                bits.append(f"{f}: {'yes' if fr['bitwise'] else 'NO'}")
        lifts = (
            "yes"
            if r["lifts"]
            else f"no ({r['lifted_kernels']}/{len(r['kernels'])} kernels)"
        )
        rows.append(
            f"| {name} | {ks} | {lifts} | {miss} | "
            f"{'; '.join(verdicts) or '-'} | {'; '.join(bits) or '-'} |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
