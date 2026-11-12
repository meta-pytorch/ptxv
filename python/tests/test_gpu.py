# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU tests (PTXV_TEST_GPU=3): in-process capture of tiny ops, checked by ptxv."""

import os

import numpy as np
import pytest
from conftest import GPU, needs_binary

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not GPU, reason="set PTXV_TEST_GPU=<gpu index>"),
    needs_binary,
]


@pytest.fixture(scope="module")
def torch():
    import ptxv  # noqa: F401  (the module hook before the first CUDA work)
    import torch

    torch.manual_seed(0)
    return torch


def test_capture_check_unperturbed(torch):
    import ptxv

    x = torch.randn(4, 128, device="cuda")
    ref = torch.cumsum(x, -1)
    torch.cuda.synchronize()
    with ptxv.capture() as cap:
        y = torch.cumsum(x, -1)
    assert torch.equal(y, ref)  # capture does not change results
    assert len(cap.kernels) == 1 and "scan" in cap.kernels[0].label
    k = cap.kernels[0]
    c = k.case()
    assert (
        c.outputs == ["y"]
        and c.buffer("x")["dtype"] == "f32"
        and c.buffer("x")["shape"] == [4, 128]
    )
    assert np.array_equal(c.expected()["y"], y.cpu().numpy())
    assert np.array_equal(c.inputs()["x"], x.cpu().numpy())
    v = k.check()
    assert v.clean and v.certified and v.sim_equals_gpu


def test_regions_dedupe_and_selection(torch):
    import ptxv

    x = torch.randn(8, 32, device="cuda")
    with ptxv.capture(ops=True) as cap:
        for _ in range(3):
            y = torch.cumsum(x, -1)
        z = torch.cat([x, y], 0)
    assert len(cap) == 4 and len(cap.kernels) == 2
    scan = cap.kernels["scan"]
    assert len(scan.instances) == 3 and any("cumsum" in o for o in scan.ops)
    cat = cap.kernels.filter(op="cat")[0]
    assert "Cat" in cat.label
    region = cap.select(index=[2, 3])  # the last cumsum, then cat: one two-launch case
    c = region.case()
    assert len(c.kernels) == 2 and "z" in c.outputs
    assert np.array_equal(c.expected()["z"], z.cpu().numpy())
    assert region.check().clean


def test_selected_launch_reads_region_allocation(torch):
    """A launch selected out of a region reads a buffer allocated in the region and written by
    an earlier launch: the buffer starts from that launch's pre-snapshot, not zeros, and is not
    an output."""
    import ptxv

    x = torch.randn(8, 32, device="cuda")
    with ptxv.capture() as cap:
        y = torch.cumsum(x, -1)  # y: allocated in the region, written by launch 0
        z = torch.cat([y, y], 0)
    k = cap.select(index=[1])
    c = k.case()
    assert c.outputs == ["z"]
    assert np.array_equal(c.inputs()["y"], y.cpu().numpy())
    assert np.array_equal(c.expected()["z"], z.cpu().numpy())
    v = k.check()
    assert v.clean and v.sim_equals_gpu


def test_compare_torch_vs_triton(torch):
    pytest.importorskip("triton")
    import sys

    import ptxv

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(__file__), "..", "..", "tools", "capture_examples"
        ),
    )
    import triton_cumsum as tc

    x = torch.randn(4, 128, device="cuda")
    eq = ptxv.compare("torch.cumsum(x, -1)", tc.run, inputs={"x": x})
    assert eq.level == "L1" and eq.proven
    c = tc.run(x)
    r = torch.cumsum(x, -1)
    chk = eq.bound.check(r, c, {"x": x})
    assert chk["violations"] == 0 and chk["evaluated"] == 512


def test_pointer_table_targets_are_captured(torch):
    """A kernel that loads tensor addresses from a pointer table (multi-tensor kernels): the
    table's targets are snapshotted, and the table's words become `ptrs` of the case."""
    triton = pytest.importorskip("triton")
    import ptxv
    import triton.language as tl

    @triton.jit
    def scale_list(table, out, n: tl.constexpr):
        i = tl.program_id(0)
        src = tl.load(table + i).to(tl.pointer_type(tl.float32))
        o = tl.arange(0, n)
        tl.store(out + i * n + o, 2.0 * tl.load(src + o))

    xs = [torch.randn(64, device="cuda") for _ in range(3)]
    table = torch.tensor([x.data_ptr() for x in xs], dtype=torch.int64, device="cuda")
    out = torch.empty(3, 64, device="cuda")
    scale_list[(3,)](table, out, 64)  # compile before the region
    out.zero_()
    torch.cuda.synchronize()
    with ptxv.capture(names={"table": table, "out": out}) as cap:
        scale_list[(3,)](table, out, 64)
    k = cap.kernels[0]
    c = k.case()
    toml = (c.path / "case.toml").read_text()
    assert toml.count("byte_offset = 0}") == 3, toml
    v = k.check()
    assert v.clean and v.sim_equals_gpu


@pytest.mark.parametrize("warps", [1, 4])
def test_module_choice_respects_reqntid(torch, warps):
    """Without a recorded module handle, of two same-name modules with the same parameter
    layout the case takes the one compiled for the launch's block (EIATTR_REQNTID), not the
    smallest."""
    triton = pytest.importorskip("triton")
    import ptxv
    import triton.language as tl
    from ptxv._cubin import reqntid

    @triton.jit
    def twice(x, out, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, 2.0 * tl.load(x + o))

    x = torch.randn(256, device="cuda")
    out = torch.empty(256, device="cuda")
    twice[(1,)](x, out, 256, num_warps=4)  # compile both block shapes before the region
    twice[(1,)](x, out, 256, num_warps=1)
    out.zero_()
    torch.cuda.synchronize()
    with ptxv.capture(names={"x": x, "out": out}) as cap:
        twice[(1,)](x, out, 256, num_warps=warps)
    k = cap.kernels[0]
    for la in k.launches:
        la.record.pop("module", None)
    c = k.case()
    cub = next(c.path.glob("*.cubin")).read_bytes()
    assert reqntid(cub, k.launches[0].name) == (32 * warps, 1, 1)
    v = k.check()
    assert v.clean and v.sim_equals_gpu


def test_vacuous_input_warnings(torch):
    """An all-zero input, and a launch outside the case that writes an input between its
    snapshot and a later launch of the case, are reported in capture.json warnings."""
    triton = pytest.importorskip("triton")
    import json

    import ptxv
    import triton.language as tl

    @triton.jit
    def addone(x, out, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, tl.load(x + o) + 1.0)

    x = torch.zeros(256, device="cuda")
    out = torch.empty(256, device="cuda")
    addone[(1,)](x, out, 256)
    out.zero_()
    torch.cuda.synchronize()
    with ptxv.capture(names={"x": x, "out": out}) as cap:
        addone[(1,)](x, out, 256)
    w = json.loads((cap.kernels[0].case().path / "capture.json").read_text())[
        "warnings"
    ]
    assert any("input x: all 1024 bytes are 0x00" in m for m in w), w

    x.normal_()
    out.zero_()
    torch.cuda.synchronize()
    with ptxv.capture(names={"x": x, "out": out}, snapshots="first") as cap:
        addone[(1,)](x, out, 256)
        x.fill_(2.0)  # not part of the addone case, after its snapshot
        addone[(1,)](x, out, 256)
    k = cap.select(name="addone")
    w = json.loads((k.case().path / "capture.json").read_text())["warnings"]
    assert any("missing from the initial contents" in m for m in w), w


def test_stale_small_name_takes_the_block_extent(torch):
    """A named tensor smaller than the live allocator block at its address (a stale name for
    a reused block) does not bound the buffer: the case takes the block, and warns."""
    triton = pytest.importorskip("triton")
    import json

    import ptxv
    import triton.language as tl
    from ptxv.capture import TensorInfo

    @triton.jit
    def fill(ws, n: tl.constexpr):
        i = tl.program_id(0)
        o = tl.arange(0, n)
        tl.store(ws + i * n + o, (i * n + o).to(tl.float32))

    fill[(4,)](torch.empty(1024, device="cuda"), 256)  # compile before the region
    torch.cuda.synchronize()
    with ptxv.capture(frame_locals=False) as cap:
        ws = torch.empty(4 * 1024 + 512, device="cuda")  # one 18 KiB block
        fill[(4,)](ws, 1024)
    p = ws.untyped_storage().data_ptr()
    cap.names = {"stale": TensorInfo("stale", p, 512, "f64", [64], True, p, False)}
    c = cap.kernels[0].case()
    meta = json.loads((c.path / "capture.json").read_text())
    assert any("smaller than the allocator block" in m for m in meta["warnings"]), meta
    assert next(b for b in meta["buffers"] if b["name"] == "stale")["size"] >= 16384, (
        meta
    )
    v = cap.kernels[0].check()
    assert v.clean and v.sim_equals_gpu


def test_freed_block_does_not_join_its_successors(torch):
    """A block freed in the region and its memory reused by two adjacent tensors: the stale
    block is not live at the launch, so the second tensor is its own buffer, not a pointer
    one past the end of the first."""
    triton = pytest.importorskip("triton")
    import ptxv
    import triton.language as tl

    @triton.jit
    def two_out(x, dx, dg, n: tl.constexpr):
        o = tl.arange(0, n)
        v = tl.load(x + o)
        tl.store(dx + o, v * 2)
        tl.store(dg + o, v * 3)

    x = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    dx, dg = torch.empty_like(x), torch.empty_like(x)
    two_out[(1,)](x=x, dx=dx, dg=dg, n=1024)  # compile before the region
    del dx, dg
    torch.cuda.synchronize()
    with ptxv.capture(names={"x": x}) as cap:
        big = torch.empty(2 * 4096, device="cuda", dtype=torch.bfloat16)
        big.fill_(1)
        del big
        dx, dg = torch.empty_like(x), torch.empty_like(x)
        assert dg.data_ptr() == dx.data_ptr() + dx.nbytes
        two_out[(1,)](x=x, dx=dx, dg=dg, n=1024)
    k = next(k for k in cap.kernels if k.name.startswith("two_out"))
    toml = (k.case().path / "case.toml").read_text()
    assert 'buffer = "dg"' in toml and "byte_offset = 8192" not in toml, toml
    v = k.check()
    assert v.clean and v.sim_equals_gpu


def test_pointer_table_targets_in_other_segments(torch):
    """A pointer table in torch's small-tensor segment (2 MiB, larger than the scan limit) whose
    targets live in other segments: the table is scanned from the parameter's address, and the
    targets are snapshotted (before: they started as zeros)."""
    triton = pytest.importorskip("triton")
    import json

    import ptxv
    import triton.language as tl

    @triton.jit
    def first_of(table, out, n: tl.constexpr):
        i = tl.program_id(0)
        src = tl.load(table + i).to(tl.pointer_type(tl.float32))
        o = tl.arange(0, n)
        tl.store(out + i * n + o, tl.load(src + o) + 1.0)

    xs = [
        torch.randn(1 << 19, device="cuda") for _ in range(2)
    ]  # 2 MiB each: own segments
    table = torch.tensor([x.data_ptr() for x in xs], dtype=torch.int64, device="cuda")
    out = torch.empty(2, 64, device="cuda")
    first_of[(2,)](table, out, 64)
    out.zero_()
    torch.cuda.synchronize()
    with ptxv.capture(names={"table": table, "out": out}) as cap:
        first_of[(2,)](table, out, 64)
    c = cap.kernels[0].case()
    w = json.loads((c.path / "capture.json").read_text())["warnings"]
    assert not any("no snapshot covers it" in m for m in w), w
    v = cap.kernels[0].check()
    assert v.clean and v.sim_equals_gpu


def test_same_name_kernels_use_the_launched_module(torch):
    """Two specializations of one Triton kernel share its name and parameter layout; the case
    takes the module the launch used (recorded by handle), not the smallest one."""
    triton = pytest.importorskip("triton")
    import ptxv
    import triton.language as tl

    @triton.jit
    def addc(x, out, C: tl.constexpr):
        o = tl.arange(0, 64)
        y = tl.load(x + o) + C
        if C == 2:  # a larger module: the smallest image is the wrong one
            for _ in tl.static_range(16):
                y = y * 1.5 + 0.25
        tl.store(out + o, y)

    x = torch.randn(64, device="cuda")
    out = torch.empty(64, device="cuda")
    addc[(1,)](x, out, 2)  # compile both specializations before the region
    addc[(1,)](x, out, 1)
    out.zero_()
    torch.cuda.synchronize()
    with ptxv.capture(names={"x": x, "out": out}) as cap:
        addc[(1,)](x, out, 2)
    k = cap.kernels[0]
    assert k.launches[0].record.get("module"), k.launches[0].record
    v = k.check()
    assert v.clean and v.sim_equals_gpu


def test_same_name_specializations_keep_their_own_cubin(torch):
    # Two specializations of one Triton kernel (constexpr S) share the entry name; each
    # launch's case cubin is its own module's, so both outputs match the GPU.
    import ptxv
    import triton
    import triton.language as tl

    @triton.jit
    def scale_kernel(x, y, n, S: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(y + i, tl.load(x + i, mask=m) * S, mask=m)

    def run(x):
        a, b = torch.empty_like(x), torch.empty_like(x)
        scale_kernel[(2,)](x, a, 256, S=2.0, BLOCK=128)
        scale_kernel[(2,)](x, b, 256, S=3.0, BLOCK=128)
        return a, b

    x = torch.randn(256, device="cuda")
    run(x)  # compile and load both specializations first
    torch.cuda.synchronize()
    with ptxv.capture() as cap:
        a, b = run(x)
        torch.cuda.synchronize()
    c = cap.all().case()
    assert np.array_equal(c.expected()["a"], (x * 2).cpu().numpy())
    assert np.array_equal(c.expected()["b"], (x * 3).cpu().numpy())
    v = cap.all().check()
    assert v.clean and v.sim_equals_gpu


def test_ptx_kept_for_every_kernel_of_a_case(torch, tmp_path):
    """Two Triton kernels from two modules in one case: their PTX (Triton's cache, matched to
    the captured machine code) is merged into one kernel.ptx and the PTX front checks it."""
    import subprocess

    import ptxv
    import triton
    import triton.language as tl

    @triton.jit
    def px_add(x, y, n, B: tl.constexpr):
        o = tl.program_id(0) * B + tl.arange(0, B)
        tl.store(y + o, tl.load(x + o, mask=o < n) + 1.0, mask=o < n)

    @triton.jit
    def px_mul(x, y, n, B: tl.constexpr):
        o = tl.program_id(0) * B + tl.arange(0, B)
        tl.store(y + o, tl.load(x + o, mask=o < n) * 2.0, mask=o < n)

    x = torch.randn(256, device="cuda")
    y = torch.empty_like(x)
    z = torch.empty_like(x)
    px_add[(2,)](x, y, 256, B=128)
    px_mul[(2,)](y, z, 256, B=128)
    torch.cuda.synchronize()
    with ptxv.capture() as cap:
        y = torch.empty_like(x)
        z = torch.empty_like(x)
        px_add[(2,)](x, y, 256, B=128)
        px_mul[(2,)](y, z, 256, B=128)
    d = tmp_path / "c"
    cap.select(index=[0, 1]).write(d, ptx=True)
    toml = (d / "case.toml").read_text()
    src = (d / "kernel.ptx").read_text()
    assert (
        'ptx = "kernel.ptx"' in toml
        and ".entry px_add" in src
        and ".entry px_mul" in src
    )
    r = subprocess.run(
        [
            os.environ.get("PTXV_BIN", "ptxv"),
            "check",
            str(d),
            "--variant",
            "good",
            "--front",
            "ptx",
            "--certify",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_scratch_workspace(torch):
    """A workspace allocated before the region and written before any read: listed in
    capture.json scratch_suggestions (not applied); with scratch=("ws",) it is init = "undef",
    scratch = true and not an output, and the case checks clean."""
    triton = pytest.importorskip("triton")
    import json
    import tomllib

    import ptxv
    import triton.language as tl

    @triton.jit
    def via_ws(x, ws, out, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(ws + o, tl.load(x + o) * 3.0)
        tl.debug_barrier()
        tl.store(out + o, tl.load(ws + o) + 1.0)

    x = torch.randn(256, device="cuda")
    ws = torch.full((256,), 7.0, device="cuda")
    out = torch.empty(256, device="cuda")
    via_ws[(1,)](x, ws, out, 256)  # compile before the region
    out.zero_()
    ws.fill_(7.0)
    torch.cuda.synchronize()
    with ptxv.capture(names={"x": x, "ws": ws, "out": out}) as cap:
        via_ws[(1,)](x, ws, out, 256)
    k = cap.kernels[0]
    meta = json.loads((k.case().path / "capture.json").read_text())
    assert [s["buffer"] for s in meta.get("scratch_suggestions", [])] == ["ws"], meta
    c = k.case(scratch=("ws",))
    man = tomllib.loads((c.path / "case.toml").read_text())
    b = {x["name"]: x for x in man["buffers"]}
    assert b["ws"]["init"] == "undef" and b["ws"]["scratch"], b["ws"]
    assert "ws" not in man["outputs"], man["outputs"]
    assert c.check().clean


def test_copies_between_launches_reach_the_case(torch):
    """MISS 173: buffers a device copy (clone) and a host upload fill between two launches of
    the group start with the bytes the launch that reads them saw (its snapshot), not those of
    the earliest snapshot of their allocator segment, and the case reproduces the GPU's output."""
    import json

    import ptxv

    torch.cuda.synchronize()
    with ptxv.capture() as cap:
        x = torch.arange(64, dtype=torch.float32, device="cuda")
        y = x.clone()
        t = torch.linspace(1.0, 2.0, 64).cuda()
        out = torch.addcmul(x, y, t)
    g = cap.all()
    c = g.case()
    ins = c.inputs()
    assert np.array_equal(ins["y"], y.cpu().numpy())
    assert np.array_equal(ins["t"], t.cpu().numpy())
    assert np.array_equal(c.expected()["out"], out.cpu().numpy())
    meta = json.loads((c.path / "capture.json").read_text())
    src = {b["name"]: b["initial_bytes"] for b in meta["buffers"]}
    assert src["y"].startswith("snapshot before launch") and src["t"] == src["y"], src
    v = g.check()
    assert v.clean and v.sim_equals_gpu


def test_overwrite_between_launches_refuses_the_case(torch):
    """MISS 173: a buffer two launches of the group reference, overwritten between them by a host
    upload, refuses the case: a case cannot express the copy."""
    import ptxv
    from ptxv._casewriter import CaseError

    torch.cuda.synchronize()
    with ptxv.capture() as cap:
        x = torch.arange(64, dtype=torch.float32, device="cuda")
        x.copy_(torch.linspace(1.0, 2.0, 64))
        y = x * 2
    assert torch.equal(y, torch.linspace(1.0, 2.0, 64, device="cuda") * 2)
    with pytest.raises(CaseError, match="MISS 173"):
        cap.all().case()
