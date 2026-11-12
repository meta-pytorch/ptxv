# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torch.compile (Inductor) vs eager PyTorch, kernel by kernel (docs/compile-vs-eager.md).
#
#   source env.sh
#   # 1. Record eager + compiled forward passes of a model, pair every compiled kernel with the
#   #    eager kernel sub-sequence computing the same tensor, and write one ref/cand pair of
#   #    capture files per compiled kernel output:
#   CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/compile_vs_eager.py record --dtype f32 --out /tmp/cve_f32
#   # 2. Capture both sides of every pair on the GPU and run `ptxv compare` on it:
#   $PTXV_PYTHON tools/compile_vs_eager.py run /tmp/cve_f32 [--only REGEX] [--gpu 3]
#   # 3. One markdown table over one or more recordings:
#   $PTXV_PYTHON tools/compile_vs_eager.py report /tmp/cve_f32 /tmp/cve_bf16 --md OUT.md
#
# How the pairing works (all automatic):
# * Eager: the model runs under a TorchDispatchMode that records every ATen op (what eager
#   actually launches, after CompositeImplicit decompositions), its tensor arguments
#   (data_ptr / size / stride: tensor identity) and a copy of every output.
# * Compiled: torch.compile(mode="default"); the Inductor output code module (the same code
#   TORCH_LOGS=output_code prints) is found in PyCodeCache, its graph inputs are taken from
#   `call`'s frame, and `call` is run again with every Triton kernel's `run` wrapped (the
#   kernel's in_ptr / out_ptr / in_out_ptr arguments, before and after the launch) and the
#   extern ATen ops (mm, SDPA) recorded by the same dispatch mode.
# * For every output buffer of a compiled kernel, the eager tensor holding the same values
#   (same dtype and element count, smallest relative difference; graph inputs and weights
#   match exactly) is its counterpart. Each input buffer is matched the same way. The eager
#   sub-sequence is the backward slice from the matched output to the matched inputs (or
#   graph leaves): every eager op in between, views included, replayed on the compiled
#   kernel's own input values, so both sides read identical bytes.
# * The source FX nodes Inductor lists for the kernel ("Topologically Sorted Source Nodes")
#   are kept in the pair metadata for the report.
#
# Each pair directory holds data.pt (inputs, by name: the same names on both sides, which is
# what makes equivalence leaves line up), spec.pt, ref.py and cand.py (capture.py --file
# targets) and, after `run`, the compare report.
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# --------------------------------------------------------------------------- replay


def _aten(name):
    import torch

    ns, op, ov = name.split(".")
    return getattr(getattr(getattr(torch.ops, ns), op), ov)


def _subst(x, f):
    if isinstance(x, dict) and "$T" in x:
        return f(x)
    if isinstance(x, (list, tuple)):
        return type(x)(_subst(v, f) for v in x)
    if isinstance(x, dict):
        return {k: _subst(v, f) for k, v in x.items()}
    return x


def make(pair_file, side):  # noqa: C901 (builds one model per pair kind; sequential script)
    """(inputs, run) for capture.py --file: `side` is "ref" or "cand"."""
    import torch

    d = os.path.dirname(os.path.abspath(pair_file))
    spec = torch.load(f"{d}/spec.pt", weights_only=False)
    data = torch.load(f"{d}/data.pt", weights_only=False)
    s = spec[side]

    def inputs():
        return {n: t.cuda() for n, t in data.items()}

    def view(inp, t):
        if "input" in t:
            return torch.as_strided(
                inp[t["input"]], t["size"], t["stride"], t["offset"]
            )
        return None

    if s["kind"] == "ops":

        def run(**inp):
            env = {}

            def get(t):
                v = view(inp, t)
                if v is not None:
                    return v
                if "new" in t:
                    n, dt = t["new"]
                    base = torch.empty(n, dtype=getattr(torch, dt), device="cuda")
                    env[t["$T"]] = torch.as_strided(
                        base, t["size"], t["stride"], t["offset"]
                    )
                return env[t["$T"]]

            for op in s["ops"]:
                out = _aten(op["func"])(
                    *_subst(op["args"], get), **_subst(op["kwargs"], get)
                )
                flat = out if isinstance(out, (list, tuple)) else [out]
                for tid, o in zip(op["outs"], flat):
                    if tid is not None:
                        env[tid] = o
            o = s["output"]
            if "input" in o:  # an in-place output: capture finds the changed input
                return None
            r = env[o["$T"]]
            return torch.as_strided(r, (o["numel"],), (1,), o["offset"])

        return inputs, run

    # A Triton kernel of the Inductor output code module.
    import importlib.util

    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    ms = importlib.util.spec_from_file_location("inductor_code", s["module"])
    mod = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(mod)
    k = getattr(mod, s["kernel"])
    if s.get("config") and len(getattr(k, "configs", None) or []) > 1:
        keep = [c for c in k.configs if str(c) == s["config"]]
        if keep:
            k.configs = keep

    def run(**inp):
        from torch._C import _cuda_getCurrentRawStream as raw_stream

        outs = {}
        args = []
        for a in s["args"]:
            if isinstance(a, dict) and "$T" in a:
                if "input" in a:
                    args.append(view(inp, a))
                else:
                    n, dt = a["new"]
                    outs[a["$T"]] = torch.empty(
                        n, dtype=getattr(torch, dt), device="cuda"
                    )
                    args.append(outs[a["$T"]][a["offset"] :])
            else:
                args.append(a)
        k.run(*args, stream=raw_stream(0))
        o = s["output"]
        if "input" in o:
            return None
        return outs[o["$T"]]

    return inputs, run


# --------------------------------------------------------------------------- the model


def llama_block(dtype, d=128, h=2, f=256, T=64, B=1):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.randn(d) * 0.1 + 1)

        def forward(self, x):
            xf = x.float()
            return (
                xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            ).type_as(x) * self.weight

    def rope(x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = h
            self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
            self.wqkv = nn.Linear(d, 3 * d, bias=False)
            self.wo = nn.Linear(d, d, bias=False)
            self.w1 = nn.Linear(d, f, bias=False)
            self.w3 = nn.Linear(d, f, bias=False)
            self.w2 = nn.Linear(f, d, bias=False)

        def forward(self, x, cos, sin):
            B, T, D = x.shape
            hd = D // self.h
            q, k, v = self.wqkv(self.n1(x)).split(D, -1)
            q = q.view(B, T, self.h, hd).transpose(1, 2)
            k = k.view(B, T, self.h, hd).transpose(1, 2)
            v = v.view(B, T, self.h, hd).transpose(1, 2)
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.wo(a.transpose(1, 2).reshape(B, T, D))
            hh = self.n2(x)
            return x + self.w2(F.silu(self.w1(hh)) * self.w3(hh))

    torch.manual_seed(0)
    m = Block().cuda().to(dtype).eval()
    x = torch.randn(B, T, d, device="cuda", dtype=dtype)
    hd = d // h
    t = torch.arange(T, device="cuda").float()[:, None] * (
        1e4 ** (-torch.arange(0, hd, 2, device="cuda").float() / hd)
    )
    leaves = {"x": x, "cos": t.cos().to(dtype), "sin": t.sin().to(dtype)}
    for n, p in m.named_parameters():
        leaves[n.replace(".", "_")] = p.data
    return m, (leaves["x"], leaves["cos"], leaves["sin"]), leaves


# --------------------------------------------------------------------------- recording

VIEW_OPS = {
    "aten.view.default",
    "aten._unsafe_view.default",
    "aten.t.default",
    "aten.transpose.int",
    "aten.permute.default",
    "aten.slice.Tensor",
    "aten.split.Tensor",
    "aten.unsqueeze.default",
    "aten.squeeze.dim",
    "aten.expand.default",
    "aten.as_strided.default",
    "aten.select.int",
    "aten.alias.default",
    "aten.reshape.default",
    "aten.detach.default",
    "aten._reshape_alias.default",
    "aten.unbind.int",
    "aten.split_with_sizes.default",
}
NO_KERNEL = VIEW_OPS | {"aten.empty_strided.default", "aten.empty.memory_format"}


def tensor_raw(t):
    """The dense memory region of a tensor as a 1-D tensor, or None if it has holes."""
    import torch

    n = t.numel()
    if n == 0 or any(s < 0 for s in t.stride()):
        return None
    ext = 1 + sum((sz - 1) * st for sz, st in zip(t.shape, t.stride()))
    if ext != n:
        return None
    return torch.as_strided(t, (n,), (1,), t.storage_offset())


def storage_1d(t):
    import torch

    n = t.untyped_storage().nbytes() // t.element_size()
    return torch.empty(0, dtype=t.dtype, device=t.device).set_(
        t.untyped_storage(), 0, (n,), (1,)
    )


class Recorder:
    """TorchDispatchMode recording ATen ops with tensor identity."""

    def __init__(self, leaves=None):
        from torch.utils._python_dispatch import TorchDispatchMode

        rec = self
        self.ops = []  # dicts: func, args, kwargs, outs (tids)
        self.tensors = {}  # tid -> info
        self.key2tid = {}
        self.keep = []
        self.suspend = False
        self.events = None  # compiled run: kernel events appended here too
        for n, t in (leaves or {}).items():
            tid = self.tid(t)
            self.tensors[tid]["leaf"] = n

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if rec.suspend:
                    return func(*args, **kwargs)
                return rec.on_op(func, args, kwargs)

        self.mode = Mode()

    def tid(self, t):
        key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype))
        if key not in self.key2tid:
            tid = len(self.tensors)
            self.key2tid[key] = tid
            self.keep.append(t)
            self.tensors[tid] = {
                "key": key,
                "size": list(t.shape),
                "stride": list(t.stride()),
                "offset": t.storage_offset(),
                "dtype": str(t.dtype).replace("torch.", ""),
                "data_ptr": t.data_ptr(),
                "esize": t.element_size(),
                "storage": (
                    t.untyped_storage().data_ptr(),
                    t.untyped_storage().nbytes(),
                ),
                "producer": None,
                "leaf": None,
                "value": None,
            }
        return self.key2tid[key]

    def templ(self, x):
        import torch

        if isinstance(x, torch.Tensor):
            return {"$T": self.tid(x)}
        if isinstance(x, (list, tuple)):
            return type(x)(self.templ(v) for v in x)
        if isinstance(x, dict):
            return {k: self.templ(v) for k, v in x.items()}
        if isinstance(
            x, (torch.dtype, torch.device, torch.memory_format, torch.layout)
        ):
            return x
        return x

    def on_op(self, func, args, kwargs):
        import torch

        name = str(func)
        snap = None
        if self.events is not None and name not in NO_KERNEL:
            snap = self.snapshot(list(args) + list(kwargs.values()))
        out = func(*args, **kwargs)
        self.suspend = True
        try:
            idx = len(self.ops)
            flat = out if isinstance(out, (list, tuple)) else [out]
            outs = []
            for o in flat:
                if isinstance(o, torch.Tensor):
                    t = self.tid(o)
                    info = self.tensors[t]
                    if info["producer"] is None and info["leaf"] is None:
                        info["producer"] = idx
                        r = tensor_raw(o)
                        info["value"] = r.clone() if r is not None else None
                    outs.append(t)
                else:
                    outs.append(None)
            self.ops.append(
                {
                    "func": name,
                    "args": self.templ(list(args)),
                    "kwargs": self.templ(kwargs),
                    "outs": outs,
                }
            )
            if snap is not None:
                outk = kwargs.get("out")
                outt = (
                    [outk]
                    if isinstance(outk, torch.Tensor)
                    else [o for o in flat[:1] if isinstance(o, torch.Tensor)]
                )
                after = self.snapshot(outt)
                ins = [a for a in snap if outk is None or a[0] != self.tid(outk)]
                self.events.append(
                    {
                        "kind": "aten",
                        "name": name,
                        "op": idx,
                        "inputs": ins,
                        "outputs": after,
                        "args": self.ops[idx]["args"],
                        "kwargs": self.ops[idx]["kwargs"],
                    }
                )
        finally:
            self.suspend = False
        return out

    def snapshot(self, xs):
        """[(tid, storage values 1-D clone)] for every tensor in xs (nested)."""
        import torch

        res = []

        def walk(x):
            if isinstance(x, torch.Tensor):
                res.append((self.tid(x), storage_1d(x).clone()))
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)

        prev = self.suspend
        self.suspend = True
        try:
            walk(xs)
        finally:
            self.suspend = prev
        return res


def wrap_kernel(rec, name, k):
    import torch

    orig = k.run
    argnames = list(k.fn.arg_names)

    def run(*args, **kw):
        prev = rec.suspend
        rec.suspend = True
        try:
            names = argnames[: len(args)]
            role = {
                n: (
                    "inout"
                    if n.startswith("in_out_ptr")
                    else "out"
                    if n.startswith("out_ptr")
                    else "in"
                    if n.startswith("in_ptr")
                    else "scalar"
                )
                for n in names
            }
            before = [
                (n, rec.tid(a), storage_1d(a).clone())
                for n, a in zip(names, args)
                if isinstance(a, torch.Tensor) and role[n] in ("in", "inout")
            ]
            r = orig(*args, **kw)
            torch.cuda.synchronize()
            after = [
                (n, rec.tid(a), storage_1d(a).clone())
                for n, a in zip(names, args)
                if isinstance(a, torch.Tensor) and role[n] in ("out", "inout")
            ]
            targs = []
            for n, a in zip(names, args):
                if isinstance(a, torch.Tensor):
                    targs.append({"$T": rec.tid(a), "arg": n, "role": role[n]})
                else:
                    targs.append(a)
            cfg = None
            if getattr(k, "launchers", None):
                cfg = str(k.launchers[0].config)
            rec.events.append(
                {
                    "kind": "triton",
                    "name": name,
                    "args": targs,
                    "inputs": before,
                    "outputs": after,
                    "config": cfg,
                }
            )
            return r
        finally:
            rec.suspend = prev

    k.run = run


def source_nodes(code):
    """kernel name -> the FX source nodes Inductor lists for it."""
    res = {}
    lines = code.splitlines()
    for i, l in enumerate(lines):
        m = re.match(r"\s*(triton_\w+)\.run\(", l)
        if m:
            for j in range(i - 1, max(0, i - 4), -1):
                s = re.search(r"Topologically Sorted Source Nodes: \[(.*?)\]", lines[j])
                if s:
                    res.setdefault(m.group(1), s.group(1))
                    break
    return res


def record(a):  # noqa: C901 (one linear pass matching compiled to eager events)
    import torch
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.utils import run_and_get_code

    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    m, args, leaves = llama_block(dtype, T=a.seq)

    # Eager.
    erec = Recorder(leaves)
    with torch.no_grad(), erec.mode:
        y_eager = m(*args)
    torch.cuda.synchronize()

    # Compiled.
    torch._dynamo.reset()
    with torch.no_grad():
        cm = torch.compile(m, mode="default")
        y_comp, codes = run_and_get_code(cm, *args)
    code = codes[-1]
    open(f"{out}/inductor_code.py", "w").write(code)
    mods = [
        md
        for md in (
            PyCodeCache.modules
            if isinstance(PyCodeCache.modules, list)
            else PyCodeCache.modules.values()
        )
        if hasattr(md, "call")
        and "def call(" in open(md.__file__).read()
        and open(md.__file__).read().strip() == code.strip()
    ]
    if not mods:
        sys.exit("could not find the Inductor module in PyCodeCache")
    mod = mods[-1]
    # Graph inputs from call's frame (assert_size_stride is called first on each of them).
    grabbed = {}
    orig_assert = mod.assert_size_stride

    def grab(*xa, **xk):
        if not grabbed:
            f = sys._getframe(1)
            for n, v in f.f_locals.items():
                mm = re.fullmatch(r"arg(\d+)_1", n)
                if mm and isinstance(v, torch.Tensor):
                    grabbed[int(mm.group(1))] = v
        return orig_assert(*xa, **xk)

    mod.assert_size_stride = grab
    with torch.no_grad():
        cm(*args)
    mod.assert_size_stride = orig_assert
    gargs = [grabbed[i] for i in sorted(grabbed)]

    crec = Recorder()
    crec.events = []
    kernels = {
        n: v
        for n, v in vars(mod).items()
        if n.startswith("triton_") and hasattr(v, "run") and hasattr(v, "fn")
    }
    for n, k in kernels.items():
        wrap_kernel(crec, n, k)
    with torch.no_grad(), crec.mode:
        res = mod.call(list(gargs))
    torch.cuda.synchronize()
    print(
        f"eager vs compiled output: max |diff| = {(res[0].float() - y_eager.float()).abs().max().item():.3e}"
    )
    nodes = source_nodes(code)

    # ------------------------------------------------------------- pairing
    eager_vals = [
        (tid, info)
        for tid, info in erec.tensors.items()
        if (info["value"] is not None) or info["leaf"]
    ]
    for tid, info in eager_vals:
        if info["leaf"] and info["value"] is None:
            info["value"] = tensor_raw(erec.keep[tid]).clone()

    def best(val, exact_only=False):
        """The eager tensor whose dense values best match `val` (1-D)."""
        cands = []
        vf = val.float()
        scale = vf.abs().max().item() + 1e-30
        for tid, info in eager_vals:
            ev = info["value"]
            if ev is None or ev.numel() != val.numel() or ev.dtype != val.dtype:
                continue
            if torch.equal(ev, val):
                score = 0.0
            else:
                score = (ev.float() - vf).abs().max().item() / scale
            is_view = (
                info["producer"] is not None
                and erec.ops[info["producer"]]["func"] in VIEW_OPS
            )
            cands.append((score, is_view, info["producer"] or -1, tid))
        cands.sort()
        if not cands:
            return None, None
        s, _, _, tid = cands[0]
        thr = 0.0 if exact_only else (2e-2 if val.dtype == torch.float32 else 8e-2)
        return (tid, s) if s <= thr else (None, s)

    def in_region(t, reg):
        lo, n, es = reg
        return lo <= t["data_ptr"] < lo + n * es

    pairs = []
    pdir = f"{out}/pairs"
    os.makedirs(pdir, exist_ok=True)
    for ei, ev in enumerate(crec.events):
        # Inputs: compiled tid -> (name, eager tid, value)
        ins = {}
        for item in ev["inputs"]:
            ctid, val = item[-2], item[-1]
            st = crec.tensors[ctid]["storage"]
            if any(
                crec.tensors[c]["storage"] == st and v.numel() == val.numel()
                for c, (_, _, v) in ins.items()
            ):
                # Another view of the same storage: the same input.
                same = [c for c in ins if crec.tensors[c]["storage"] == st]
                ins[ctid] = ins[same[0]]
                continue
            etid, score = best(val)
            if etid is not None:
                info = erec.tensors[etid]
                if info["leaf"]:
                    nm = info["leaf"]
                else:
                    nm = f"{erec.ops[info['producer']]['func'].split('.')[1]}_{info['producer']}"
            else:
                nm = f"in{len(ins)}"
            ins[ctid] = (nm, etid, val)
        outs_ = ev["outputs"]
        if ev["kind"] == "aten":
            outs_ = outs_[:1]
        for _oi, item in enumerate(outs_):
            ctid, val = item[-2], item[-1]
            argname = item[0] if len(item) == 3 else "out"
            short = ev["name"] if ev["kind"] == "triton" else ev["name"].split(".")[1]
            pname = f"{len(pairs):02d}_{short[:60]}" + (
                f"_{argname}" if len(outs_) > 1 else ""
            )
            meta = {
                "name": pname,
                "event": ei,
                "kind": ev["kind"],
                "kernel": ev["name"],
                "output_arg": argname,
                "source_nodes": nodes.get(ev["name"]),
                "dtype": a.dtype,
            }
            etid, score = best(val)
            meta["match_score"] = score
            if etid is None:
                meta["status"] = (
                    f"no eager tensor matches output {argname} (best relative difference {score})"
                )
                pairs.append(meta)
                continue
            einfo = erec.tensors[etid]
            if einfo["leaf"]:
                meta["status"] = f"output matches graph input {einfo['leaf']}"
                pairs.append(meta)
                continue
            # Eager regions of the matched inputs (and their names).
            regions = []  # (region, name, eager tid)
            for _c, (nm, etid_in, v) in ins.items():
                if etid_in is not None:
                    inf = erec.tensors[etid_in]
                    regions.append(((inf["data_ptr"], v.numel(), inf["esize"]), nm))
            for _tid, info in erec.tensors.items():
                if info["leaf"]:
                    r = (info["data_ptr"], info["value"].numel(), info["esize"])
                    if not any(rr[0] == r for rr in regions):
                        regions.append((r, info["leaf"]))
            # Backward slice.
            need, extra = set(), set()
            stack = [einfo["producer"]]

            def ext_of(t):
                for r, nm in regions:  # noqa: B023 (closure used only within this iteration)
                    if in_region(t, r):
                        return r, nm
                return None

            ext_args = {}
            while stack:
                oi_ = stack.pop()
                if oi_ in need:
                    continue
                need.add(oi_)
                op = erec.ops[oi_]

                def visit(x):
                    if isinstance(x, dict) and "$T" in x:
                        t = erec.tensors[x["$T"]]
                        if (
                            t["producer"] is not None
                            and t["producer"] not in need  # noqa: B023 (closure used only within this iteration)
                            and ext_of(t) is None
                        ):
                            stack.append(t["producer"])  # noqa: B023 (closure used only within this iteration)
                        elif t["producer"] is None or ext_of(t) is not None:
                            e = ext_of(t)
                            if e is None:
                                extra.add(x["$T"])  # noqa: B023 (closure used only within this iteration)
                            else:
                                ext_args[x["$T"]] = e  # noqa: B023 (closure used only within this iteration)
                    elif isinstance(x, (list, tuple)):
                        for v in x:
                            visit(v)
                    elif isinstance(x, dict):
                        for v in x.values():
                            visit(v)

                visit(op["args"])
                visit(op["kwargs"])
            order = sorted(need)
            kern_ops = [
                erec.ops[i]["func"]
                for i in order
                if erec.ops[i]["func"] not in NO_KERNEL
            ]

            # The inputs (by name: identical in both cases).
            data = {}
            name_val = {nm: v for (nm, _, v) in ins.values()}
            for _tid, (_r, nm) in ext_args.items():
                if nm in name_val:
                    data[nm] = name_val[nm].cpu()
                else:  # a graph leaf the compiled kernel does not read
                    lf = [i for i, inf in erec.tensors.items() if inf["leaf"] == nm][0]
                    data[nm] = erec.tensors[lf]["value"].cpu()
                    meta.setdefault("extra_inputs", []).append(nm)
            for tid_ in extra:
                meta.setdefault("unresolved", []).append(tid_)

            def ref_t(x):
                t = erec.tensors[x["$T"]]
                if x["$T"] in ext_args:  # noqa: B023 (closure used only within this iteration)
                    (lo, _, es), nm = ext_args[x["$T"]]  # noqa: B023 (closure used only within this iteration)
                    return {
                        "$T": x["$T"],
                        "input": nm,
                        "size": t["size"],
                        "stride": t["stride"],
                        "offset": (t["data_ptr"] - lo) // es,
                    }
                return {"$T": x["$T"]}

            ref_ops = [
                {
                    "func": erec.ops[i]["func"],
                    "args": _subst(erec.ops[i]["args"], ref_t),
                    "kwargs": _subst(erec.ops[i]["kwargs"], ref_t),
                    "outs": erec.ops[i]["outs"],
                }
                for i in order
            ]
            ref = {
                "kind": "ops",
                "ops": ref_ops,
                "output": {
                    "$T": etid,
                    "numel": einfo["value"].numel(),
                    "offset": einfo["offset"],
                },
            }

            # Candidate.
            def cand_t(x):
                t = crec.tensors[x["$T"]]
                c = x["$T"]
                if c in ins:  # noqa: B023 (closure used only within this iteration)
                    nm = ins[c][0]  # noqa: B023 (closure used only within this iteration)
                    if nm not in data:  # noqa: B023 (closure used only within this iteration)
                        data[nm] = ins[c][2].cpu()  # noqa: B023 (closure used only within this iteration)
                    base = crec.tensors[c]["storage"][0]
                    return {
                        "$T": c,
                        "input": nm,
                        "size": t["size"],
                        "stride": t["stride"],
                        "offset": (t["data_ptr"] - base) // t["esize"],
                    }
                n = t["storage"][1] // t["esize"]
                return {
                    "$T": c,
                    "new": (n, t["dtype"]),
                    "size": t["size"],
                    "stride": t["stride"],
                    "offset": t["offset"],
                }

            if ev["kind"] == "triton":
                cargs = [cand_t(x) if isinstance(x, dict) else x for x in ev["args"]]
                for x, c in zip(ev["args"], cargs):
                    if isinstance(x, dict):
                        c["arg"] = x["arg"]
                if ctid in ins:
                    cout = {"$T": ctid, "input": ins[ctid][0]}
                else:
                    cout = {"$T": ctid}
                cand = {
                    "kind": "triton",
                    "module": f"{out}/inductor_code.py",
                    "kernel": ev["name"],
                    "args": cargs,
                    "output": cout,
                    "config": ev.get("config"),
                }
            else:
                op = crec.ops[ev["op"]]
                cand = {
                    "kind": "ops",
                    "ops": [
                        {
                            "func": op["func"],
                            "args": _subst(op["args"], cand_t),
                            "kwargs": _subst(op["kwargs"], cand_t),
                            "outs": op["outs"],
                        }
                    ],
                    "output": {"$T": ctid, "numel": val.numel(), "offset": 0},
                }
                # out= tensors are fresh; their full storage is the output.
                ct = crec.tensors[ctid]
                if ct["offset"] != 0 or ct["size"] != [val.numel()]:
                    cand["output"]["offset"] = 0
            meta.update(
                eager_ops=kern_ops,
                eager_kernel_ops=len(kern_ops),
                inputs=sorted(data),
                status="paired",
            )
            d = f"{pdir}/{pname}"
            os.makedirs(d, exist_ok=True)
            torch.save(data, f"{d}/data.pt")
            torch.save({"ref": ref, "cand": cand}, f"{d}/spec.pt")
            for side in ("ref", "cand"):
                open(f"{d}/{side}.py", "w").write(
                    f"# {side} of compile-vs-eager pair {pname} (tools/compile_vs_eager.py)\n"
                    f"import sys\nsys.path.insert(0, {os.path.join(ROOT, 'tools')!r})\n"
                    f"from compile_vs_eager import make\n"
                    f"inputs, run = make(__file__, {side!r})\n"
                )
            json.dump(meta, open(f"{d}/meta.json", "w"), indent=1)
            pairs.append(meta)
    json.dump(pairs, open(f"{out}/pairs.json", "w"), indent=1)
    for p in pairs:
        print(
            f"{p['name']}: {p['status']}; eager: {' '.join(o.split('.')[1] for o in p.get('eager_ops', []))}"
        )


# --------------------------------------------------------------------------- run


def run_pairs(a):
    out = os.path.abspath(a.dir)
    pairs = json.load(open(f"{out}/pairs.json"))
    py = os.environ.get("PTXV_PYTHON", sys.executable)
    for p in pairs:
        if p["status"] != "paired" or (a.only and not re.search(a.only, p["name"])):
            continue
        d = f"{out}/pairs/{p['name']}"
        if a.skip_done and os.path.exists(f"{d}/cmp/report.json"):
            continue
        mem = [
            "systemd-run",
            "--user",
            "--scope",
            "-q",
            "-p",
            "MemoryMax=32G",
            "-p",
            "MemorySwapMax=0",
        ]
        if a.recompare and os.path.exists(f"{d}/cmp/cand/case.toml"):
            # The captures as they are; only `ptxv compare` again (no GPU).
            ptxv = os.path.join(
                os.environ.get("CARGO_TARGET_DIR", f"{ROOT}/target"), "release/ptxv"
            )
            cmd = mem + [
                ptxv,
                "compare",
                f"{d}/cmp/ref",
                f"{d}/cmp/cand",
                "--out",
                f"{d}/cmp",
                "--max-states",
                str(a.max_states),
                "--trials",
                str(a.trials),
                "--ref-label",
                f"{d}/ref.py",
                "--cand-label",
                f"{d}/cand.py",
            ]
        else:
            cmd = mem + [
                py,
                f"{ROOT}/tools/compare.py",
                "--ref",
                f"{d}/ref.py",
                "--cand",
                f"{d}/cand.py",
                "--out",
                f"{d}/cmp",
                "--gpu",
                str(a.gpu),
                "--max-states",
                str(a.max_states),
                "--trials",
                str(a.trials),
            ]
        print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        env = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS="1")
        if a.recompare:
            env["CUDA_VISIBLE_DEVICES"] = ""
        r = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=a.timeout if a.timeout else None,
        )
        open(f"{d}/cmp.log", "w").write(r.stdout + r.stderr)
        print("\n".join((r.stdout + r.stderr).strip().splitlines()[-4:]), flush=True)


# --------------------------------------------------------------------------- report


def short(s, n=70):
    return s if len(s) <= n else s[: n - 3] + "..."


def classify(b):
    """Blocker class of one report blocker line (docs/compile-vs-eager.md)."""
    lo = b.lower()
    if (
        "hmma" in lo
        or "utcmma" in lo
        or "utchmma" in lo
        or "tensor-core" in lo
        or "tf32" in lo
    ):
        return "MMA (tensor-core) numerics / lifting"
    if "does not lift" in lo or "unsupported" in lo or "not modeled" in lo:
        return "lift gap"
    if "data-dependent" in lo:
        return "data-dependent control"
    if (
        "no dataflow history" in lo
        or "without a bound" in lo
        or "not the same function" in lo
    ):
        return "missing normal form / numerics tracking"
    return "other"


def equiv_text(r):
    h = r.get("headline", "")
    h = h.split("; GPU:")[0]
    return h


def report(a):  # noqa: C901 (report printer: one branch per pair status)
    rows = []
    for dd in a.dirs:
        pairs = json.load(open(f"{dd}/pairs.json"))
        nper = {}
        for p in pairs:
            nper[p["event"]] = nper.get(p["event"], 0) + 1
        for p in pairs:
            r = None
            rp = f"{dd}/pairs/{p['name']}/cmp/report.json"
            if os.path.exists(rp):
                r = json.load(open(rp))
            rows.append((p, r, nper[p["event"]] > 1))
    s = (
        "| # | dtype | compiled kernel | eager counterpart (kernel ops) | eager verdict | compiled verdict "
        "| equivalence | GPU max \\|C-R\\| (bitwise =) | blockers |\n|---|---|---|---|---|---|---|---|---|\n"
    )
    counts = {}
    classes = {}
    for p, r, multi in rows:
        eops = ", ".join(o.split(".")[1] for o in p.get("eager_ops", [])) or "-"
        kn = p["kernel"] if p["kind"] == "triton" else f"extern {p['kernel']}"
        kn = f"`{short(kn, 70)}`" + (f" ({p['output_arg']})" if multi else "")
        num = p["name"][:2]
        if r is None:
            st = p["status"] if p["status"] != "paired" else "not run"
            s += f"| {num} | {p['dtype']} | {kn} | {eops} | - | - | {st} | - | - |\n"
            counts["not run"] = counts.get("not run", 0) + 1
            continue

        def vtxt(sd):
            if not sd.get("lifted"):
                return "not lifted"
            c = sd.get("check") or {}
            v = c.get("verdict", "-")
            v = ", ".join(v) if isinstance(v, list) else str(v)
            if c.get("certified"):
                v += " (cert.)"
            sim = " ".join(sd.get("sim_vs_gpu") or [])
            if "MISMATCH" in sim:
                v += ", sim != GPU"
            elif "within one ulp" in sim:
                # Its line ends with "the rest bitwise equal": not sim = GPU.
                v += ", sim within 1 ulp of GPU"
            elif "bitwise equal" in sim:
                v += ", sim = GPU"
            return v

        eq = r.get("equivalence") or {}
        h = equiv_text(r)
        if h.startswith("PROVEN"):
            ew = eq.get("elementwise") or {}
            if ew.get("product"):
                kind = "EW (f(x)*y)"
            elif ew.get("max_ulps") is not None:
                kind = "EW"
            else:
                kind = eq.get("level", "?")
            counts["proven " + kind] = counts.get("proven " + kind, 0) + 1
            etxt = f"**proven {kind}**: " + short(h.replace("PROVEN ", ""), 150)
        else:
            counts["not proven"] = counts.get("not proven", 0) + 1
            etxt = short(h, 110)
        g = r.get("gpu") or {}
        gtxt = (
            f"{g['max_abs_diff']:.2e} ({g['bitwise_equal']}/{g['outputs']})"
            if g.get("outputs")
            else "-"
        )
        bl = r.get("blockers") or []
        if not h.startswith("PROVEN"):
            for b in bl:
                c = classify(b)
                classes.setdefault(c, []).append(f"{p['dtype']} {num}")
        btxt = "; ".join(short(x, 140) for x in bl[:3]) + (
            f"; +{len(bl) - 3} more" if len(bl) > 3 else ""
        )
        s += (
            f"| {num} | {p['dtype']} | {kn} | {eops} | {vtxt(r['reference']['side'])} | "
            f"{vtxt(r['candidate']['side'])} | {etxt.replace('|', chr(92) + '|')} | {gtxt} | {btxt.replace('|', chr(92) + '|') or '-'} |\n"
        )
    s += "\nCounts: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + "\n"
    s += (
        "\nBlocker classes (unproven pairs): "
        + "; ".join(
            f"{k}: {', '.join(sorted(set(v)))}" for k, v in sorted(classes.items())
        )
        + "\n"
    )
    if a.md:
        open(a.md, "w").write(s)
    print(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--dtype", choices=["f32", "bf16"], default="f32")
    r.add_argument("--out", required=True)
    r.add_argument(
        "--seq", type=int, default=64, help="sequence length (hidden 128, 2 heads)"
    )
    r2 = sub.add_parser("run")
    r2.add_argument("dir")
    r2.add_argument("--only")
    r2.add_argument("--gpu", type=int, default=3)
    r2.add_argument("--max-states", type=int, default=200000)
    r2.add_argument("--trials", type=int, default=20)
    r2.add_argument("--timeout", type=int, default=3600)
    r2.add_argument("--skip-done", action="store_true")
    r2.add_argument(
        "--recompare",
        action="store_true",
        help="reuse the captures, only run ptxv compare again",
    )
    r3 = sub.add_parser("report")
    r3.add_argument("dirs", nargs="+")
    r3.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "record":
        cuda = "/usr/local/cuda-13.0/bin"
        for tool in ("ptxas", "cuobjdump", "nvdisasm"):
            os.environ.setdefault(f"TRITON_{tool.upper()}_PATH", f"{cuda}/{tool}")
        record(a)
    elif a.cmd == "run":
        if str(a.gpu) in os.environ.get("PTXV_EXCLUDE_GPUS", "").split(","):
            sys.exit(f"GPU {a.gpu} is in PTXV_EXCLUDE_GPUS")
        run_pairs(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
