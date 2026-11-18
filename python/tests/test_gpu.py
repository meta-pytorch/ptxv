# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU tests (PTXV_TEST_GPU=3): in-process capture of tiny ops, checked by ptxv."""

import os

import numpy as np
import pytest
from conftest import assert_clean_but_scoreboard_contracts, GPU, needs_binary

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
    # SASS: the scan kernel's E-HW-SB-QUEUE contract (a WAR on R21), nothing else.
    assert_clean_but_scoreboard_contracts(v)
    assert v.certified and v.sim_equals_gpu


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
    v = region.check()
    assert_clean_but_scoreboard_contracts(v)  # the scan kernel's queue contract
    assert v.sim_equals_gpu


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
    # The second launch (no snapshot: "first") writes out2, an allocator segment of its own
    # that no snapshot of the case covers. Were it to write `out` again, the case would be
    # refused: `out`'s latest snapshot would predate it (MISS 185).
    out2 = torch.zeros(1 << 20, device="cuda")
    torch.cuda.synchronize()
    with ptxv.capture(
        names={"x": x, "out": out, "out2": out2}, snapshots="first"
    ) as cap:
        addone[(1,)](x, out, 256)
        x.fill_(2.0)  # not part of the addone case, after its snapshot
        addone[(1,)](x, out2, 256)
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


def test_earlier_region_allocations_are_not_this_regions_blocks(torch):
    """A block live when a region starts and freed inside it, its memory then taken by two
    tensors of the region: the block stood as live at every launch (blocks live at entry had
    no lifetime) and joined both tensors into one u8 buffer named after the first
    (test_regions_dedupe_and_selection in a full run: outputs ["y"], u8[4096], no "z"). Here
    region a's frame-locals snapshot keeps w alive past `del w` until region b re-reads the
    frame's locals at entry, after its allocator snapshot. A block counts until its free."""
    import ptxv

    xb = torch.randn(8, 512, device="cuda")  # 16 KiB, before both regions
    torch.cumsum(xb, -1)
    torch.cat([xb, xb], 0)  # load both kernels before the regions
    torch.cuda.synchronize()
    with ptxv.capture() as a:
        w = torch.full((16384,), 1.0, device="cuda")  # 64 KiB, allocated in region a
    wp = w.data_ptr()
    del w  # the frame-locals snapshot of region a still holds it
    with ptxv.capture() as b:
        y = torch.cumsum(xb, -1)  # 16 KiB and 32 KiB: best fit in the freed 64 KiB block
        z = torch.cat([xb, y], 0)
    assert any(blk[0] == wp and blk[2] == "during" for blk in a.blocks)
    if y.data_ptr() != wp:
        pytest.skip("the allocator placed y outside the freed block: the layout does not arise")
    # w's block, if region b sees it (live at entry while a frame-locals snapshot held w, or an
    # entry of region a's history), ends at its free, before region b's launches.
    for i, blk in enumerate(b.blocks):
        if blk[0] == wp and blk[1] == 65536:
            assert (b.block_life.get(i) or (None, None))[1] is not None, (blk, b.block_life)
    c = b.select(index=[0, 1]).case()
    assert set(c.outputs) == {"y", "z"}, c.outputs
    assert c.buffer("z")["dtype"] == "f32" and c.buffer("z")["shape"] == [16, 512]
    assert np.array_equal(c.expected()["z"], z.cpu().numpy())
    assert np.array_equal(c.expected()["y"], y.cpu().numpy())


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


def test_expected_bytes_need_the_last_launch_that_reaches_a_buffer(torch):
    """MISS 185: a buffer's expected bytes are those after the last launch of the group that
    reaches it. Under snapshots="first" the second launch of a kernel and shape takes no
    snapshot, so the latest snapshot of the buffer predates that launch's writes. The case
    writer wrote those bytes as expected (a stored grouped GEMM output held its pre-launch
    bytes); it refuses the case instead. The first launch alone still makes a case."""
    import ptxv
    from ptxv.capture import Launches
    from ptxv._casewriter import CaseError

    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    with ptxv.capture(snapshots="first") as cap:
        y = x * 2.0
        y.mul_(3.0)
    assert torch.equal(y, x * 6.0)
    g = cap.all()
    assert [la.snap is None for la in g.launches] == [False, True], [
        (la.short_name, la.snap) for la in g.launches
    ]
    with pytest.raises(CaseError, match="MISS 185"):
        g.case()
    c = Launches(g.launches[:1], cap).case()
    assert np.array_equal(c.expected()["y"], (x * 2.0).cpu().numpy())


def test_expected_bytes_through_a_pointer_table(torch):
    """MISS 185: a launch that reaches a buffer only through a pointer table is one of the
    launches that reach it. The third launch writes `out` through the table after the latest
    snapshot of `out` (the second launch of the table kernel takes none under
    snapshots="first"): refused, instead of storing the bytes after the second launch. The
    first two launches alone give the bytes after the second (its snapshot follows the table)."""
    triton = pytest.importorskip("triton")
    import ptxv
    import triton.language as tl
    from ptxv.capture import Launches
    from ptxv._casewriter import CaseError

    @triton.jit
    def fill_ones(out, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, tl.full((n,), 1.0, tl.float32))

    @triton.jit
    def add_one_through(table, n: tl.constexpr):
        dst = tl.load(table).to(tl.pointer_type(tl.float32))
        o = tl.arange(0, n)
        tl.store(dst + o, tl.load(dst + o) + 1.0)

    out = torch.empty(64, device="cuda")
    table = torch.tensor([out.data_ptr()], dtype=torch.int64, device="cuda")
    fill_ones[(1,)](out, 64)
    add_one_through[(1,)](table, 64)  # compiled before the region
    out.zero_()
    torch.cuda.synchronize()
    with ptxv.capture(snapshots="first", names={"table": table, "out": out}) as cap:
        fill_ones[(1,)](out, 64)
        add_one_through[(1,)](table, 64)
        add_one_through[(1,)](table, 64)
    assert torch.equal(out, torch.full_like(out, 3.0))
    g = cap.all()
    assert [la.snap is None for la in g.launches] == [False, False, True], [
        (la.short_name, la.snap) for la in g.launches
    ]
    with pytest.raises(CaseError, match="MISS 185"):
        g.case()
    c = Launches(g.launches[:2], cap).case()
    assert np.array_equal(c.expected()["out"], np.full(64, 2.0, np.float32))


def _table_kernels():
    """Triton kernels writing `out` through a one-word pointer table (MISS 185 table tests),
    compiled on scratch tensors so the tables of a test hold 0 until the test fills them."""
    triton = pytest.importorskip("triton")
    import torch
    import triton.language as tl

    @triton.jit
    def fill_ones(out, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, tl.full((n,), 1.0, tl.float32))

    @triton.jit
    def add_one_through(table, n: tl.constexpr):
        dst = tl.load(table).to(tl.pointer_type(tl.float32))
        o = tl.arange(0, n)
        tl.store(dst + o, tl.load(dst + o) + 1.0)

    @triton.jit
    def put_ptr(table, v):
        tl.store(table, v)

    d_out = torch.empty(64, device="cuda")
    d_tab = torch.zeros(1, dtype=torch.int64, device="cuda")
    fill_ones[(1,)](d_out, 64)
    put_ptr[(1,)](d_tab, d_out.data_ptr())
    add_one_through[(1,)](d_tab, 64)
    torch.cuda.synchronize()
    return triton, fill_ones, add_one_through, put_ptr


def test_table_filled_by_a_launch_of_the_group(torch):
    """MISS 185: a pointer table that a launch of the group fills, after the earliest snapshot
    of its allocation, still makes the launches writing through it reach the target. Under
    snapshots="first" the last of them takes no snapshot: refused (before, `out` kept the bytes
    after the third launch, 2.0 where the GPU holds 3.0). With a snapshot of every launch, the
    expected bytes are those after the last one."""
    import ptxv
    from ptxv._casewriter import CaseError

    _, fill_ones, add_one_through, put_ptr = _table_kernels()
    for snaps in ("first", "all"):
        out = torch.zeros(64, device="cuda")
        table = torch.zeros(1, dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        with ptxv.capture(
            snapshots=snaps, names={"table": table, "out": out}, scratch=["table"]
        ) as cap:
            fill_ones[(1,)](out, 64)
            put_ptr[(1,)](table, out.data_ptr())
            add_one_through[(1,)](table, 64)
            add_one_through[(1,)](table, 64)
        assert torch.equal(out, torch.full_like(out, 3.0))
        g = cap.all()
        if snaps == "first":
            assert g.launches[3].snap is None
            with pytest.raises(CaseError, match="^out: .*MISS 185"):
                g.case()
        else:
            c = g.case()
            assert np.array_equal(c.expected()["out"], np.full(64, 3.0, np.float32))


def test_table_uploaded_in_the_region(torch):
    """MISS 185: a pointer table the host uploads inside the region, after the earliest
    snapshot of the segment it shares with `out`: the table's own initial bytes (the snapshot
    before the first launch that reads it) hold the word, so it becomes `ptrs` and the case
    runs without a fault (before, the word stayed the capture's address), and the launches
    reading the table reach `out` (snapshots="first": the last takes no snapshot, refused)."""
    import json

    import ptxv
    from ptxv._casewriter import CaseError

    _, fill_ones, add_one_through, _ = _table_kernels()
    for snaps in ("first", "all"):
        out = torch.zeros(64, device="cuda")
        table = torch.zeros(1, dtype=torch.int64, device="cuda")
        host = torch.tensor([out.data_ptr()], dtype=torch.int64)
        torch.cuda.synchronize()
        with ptxv.capture(snapshots=snaps, names={"table": table, "out": out}) as cap:
            fill_ones[(1,)](out, 64)
            table.copy_(host)  # a host upload, not a launch
            add_one_through[(1,)](table, 64)
            add_one_through[(1,)](table, 64)
        assert torch.equal(out, torch.full_like(out, 3.0))
        g = cap.all()
        if snaps == "first":
            with pytest.raises(CaseError, match="^out: .*MISS 185"):
                g.case()
            continue
        c = g.case()
        assert np.array_equal(c.expected()["out"], np.full(64, 3.0, np.float32))
        toml = (c.path / "case.toml").read_text()
        assert 'ptrs = [{offset = 0, buffer = "out", byte_offset = 0}]' in toml, toml
        meta = json.loads((c.path / "capture.json").read_text())
        assert any("pointer words" in w for w in meta["warnings"])
        v = g.check()
        assert v.clean and v.sim_equals_gpu


@pytest.mark.parametrize("form", ["order", "deep", "window"])
def test_a_two_level_table_counts_its_launch(torch, form):
    """MISS 185: y (8 MiB, a segment of its own) is reached through a two-level pointer table:
    mid holds y's address, top holds mid's, and launch 1 adds 1 to y through top and mid. The
    capture copies one level of table targets, from the window it scans from each parameter
    pointer; with mid allocated before top it never reads mid's word, so launch 1 has no
    snapshot of y and, as y's last reacher, refuses the case. Before, y kept the bytes after
    launch 0, 2.0 where the GPU holds 3.0 (the check passed with a mismatch). order: launch 0
    adds 1 through mid, and one pass over the tables in address order carried launch 1 to mid
    but not on to y; deep: launch 0 adds 1 to y directly, and mid, which the group reaches only
    through top, was not read as a table; window: mid allocated after top (control), so
    launch 1's snapshots cover y: written with 3.0."""
    triton = pytest.importorskip("triton")
    import ptxv
    import triton.language as tl
    from ptxv._casewriter import CaseError

    @triton.jit
    def add_one(out, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, tl.load(out + o) + 1.0)

    @triton.jit
    def add_one_through(table, n: tl.constexpr):
        dst = tl.load(table).to(tl.pointer_type(tl.float32))
        o = tl.arange(0, n)
        tl.store(dst + o, tl.load(dst + o) + 1.0)

    @triton.jit
    def add_one_via2(top, n: tl.constexpr):
        mid = tl.load(top).to(tl.pointer_type(tl.int64))
        dst = tl.load(mid).to(tl.pointer_type(tl.float32))
        o = tl.arange(0, n)
        tl.store(dst + o, tl.load(dst + o) + 1.0)

    torch.cuda.empty_cache()
    y = torch.ones(1 << 21, device="cuda")
    if form == "window":
        top = torch.zeros(1, dtype=torch.int64, device="cuda")
        mid = torch.zeros(1, dtype=torch.int64, device="cuda")
    else:
        mid = torch.zeros(1, dtype=torch.int64, device="cuda")
        top = torch.zeros(1, dtype=torch.int64, device="cuda")
    if (mid.data_ptr() > top.data_ptr()) != (form == "window"):
        pytest.skip("mid and top are not in allocation order")
    mid.fill_(y.data_ptr())
    top.fill_(mid.data_ptr())
    q = torch.zeros(64, device="cuda")  # compile every kernel outside the region
    qm = torch.tensor([q.data_ptr()], dtype=torch.int64, device="cuda")
    qt = torch.tensor([qm.data_ptr()], dtype=torch.int64, device="cuda")
    add_one[(1,)](q, 64)
    add_one_through[(1,)](qm, 64)
    add_one_via2[(1,)](qt, 64)
    torch.cuda.synchronize()
    with ptxv.capture(snapshots="all", names={"y": y, "mid": mid, "top": top}) as cap:
        if form == "deep":
            add_one[(1,)](y, 64)  # y: 1 -> 2
        else:
            add_one_through[(1,)](mid, 64)  # y: 1 -> 2
        add_one_via2[(1,)](top, 64)  # y: 2 -> 3
    torch.cuda.synchronize()
    assert torch.equal(y[:64].cpu(), torch.full((64,), 3.0))
    g = cap.all()
    if form == "window":
        c = g.case()
        assert np.array_equal(c.expected()["y"][:64], np.full(64, 3.0, np.float32))
        return
    try:
        c = g.case()
    except CaseError as e:
        assert str(e).startswith(
            "y: launch 1, the last of the group to reach it, has no snapshot covering it"
        ), str(e)
        assert "MISS 185" in str(e) and "snapshots='all'" not in str(e), str(e)
        return
    e_ = c.expected()["y"][:64]
    if np.array_equal(e_, np.full(64, 3.0, np.float32)):
        pytest.skip("launch 1's snapshots cover y (a word in the window from top points there)")
    pytest.fail(f"written with y = {float(e_[0])} where the GPU holds 3.0")


def test_graph_captured_launch_that_does_not_run_once(torch):
    """MISS 185: a launch recorded under CUDA graph capture runs when its graph is launched.
    With no graph launch after it in the region it did not run there, and with two it may have
    run twice, while the case holds it once. A buffer it reaches then has no known bytes after
    the group, even with a snapshotted launch after it: refused, with the reason (before, the
    case was written with the later launch's bytes)."""
    import ptxv
    from ptxv._casewriter import CaseError

    y = torch.arange(1 << 16, dtype=torch.float32, device="cuda")
    y.mul_(1.0)
    y.add_(0.0)  # warm-up of both kernels
    for replays, why in ((0, "did not run there"), (2, "may have run more than once")):
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with ptxv.capture(names={"y": y}) as cap:
            with torch.cuda.graph(g):
                y.mul_(2.0)  # recorded into the graph
            for _ in range(replays):
                g.replay()
            y.add_(1.0)  # with a snapshot
        torch.cuda.synchronize()
        group = cap.all()
        assert any(la.capturing for la in group.launches)
        with pytest.raises(CaseError, match=f"^y: .*graph capture, and .*{why}"):
            group.case()


def test_graph_captured_launch_runs_at_its_graph_launch(torch):
    """MISS 185: a launch recorded under CUDA graph capture runs when its graph is launched. In
    y.add_(0.5); [graph capture: y.mul_(2)]; y.add_(1); replay, the mul runs last, at a graph
    launch after every snapshot covering y: refused (before, y's expected bytes were those
    after the second add, without the mul). A snapshotted launch after the replay that reaches
    y gives its bytes after the group."""
    import ptxv
    from ptxv._casewriter import CaseError

    y = torch.arange(1 << 16, dtype=torch.float32, device="cuda")
    y.mul_(1.0)
    y.add_(0.0)  # warm-up of both kernels
    for tail in (False, True):
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        y0 = y.clone()
        torch.cuda.synchronize()
        with ptxv.capture(names={"y": y}) as cap:
            y.add_(0.5)
            with torch.cuda.graph(g):
                y.mul_(2.0)  # recorded into the graph, runs at the replay
            y.add_(1.0)
            g.replay()
            if tail:
                y.add_(2.0)  # after the replay, with a snapshot
        torch.cuda.synchronize()
        assert torch.equal(y, (y0 + 1.5) * 2.0 + (2.0 if tail else 0.0))
        group = cap.all()
        assert any(la.capturing for la in group.launches)
        if not tail:
            with pytest.raises(CaseError, match="^y: .*graph capture.*MISS 185"):
                group.case()
        else:
            c = group.case()
            assert np.array_equal(c.expected()["y"], y.cpu().numpy())


def test_read_only_last_reacher_refusal_names_outputs(torch):
    """MISS 185: y's last reacher (z = y * 3, no snapshot under snapshots="first"; y, z and w
    in segments of their own) only reads it, which the writer cannot tell from a write: the
    refusal says so and names outputs=[...] without y as the way to keep the case, which then
    compares z and w with the GPU's bytes."""
    import ptxv
    from ptxv._casewriter import CaseError

    x = torch.arange(1 << 22, dtype=torch.float32, device="cuda")
    (x * 2.0).add_(1.0)  # warm-up of both kernels
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    with ptxv.capture(snapshots="first", names={"x": x}) as cap:
        y = x * 2.0
        z = y * 3.0
        w = z + 1.0
    torch.cuda.synchronize()
    g = cap.all()
    with pytest.raises(CaseError) as e:
        g.case()
    msg = str(e.value)
    assert msg.startswith("y: ") and "MISS 185" in msg, msg
    assert "only reads y" in msg and "case(outputs=[...]) without y" in msg, msg
    c = g.case(outputs=["z", "w"])
    assert np.array_equal(c.expected()["z"], z.cpu().numpy())
    assert np.array_equal(c.expected()["w"], w.cpu().numpy())


def test_change_after_the_last_reacher_is_not_expected(torch):
    """MISS 219: a buffer's expected bytes are its last reacher's own snapshot after it. y and u
    share an allocator segment; y.add_(1), a host copy restoring y, u.add_(1): the later
    snapshot (u's) holds y's restored bytes, which made y no output (launch 0's write never
    compared), or with outputs=[y, u] stored 0 where launch 0 wrote 1. Now y's expected bytes
    are launch 0's, with a warning naming the change after it."""
    import json

    import ptxv

    y = torch.zeros(64, device="cuda")
    u = torch.zeros(64, device="cuda")
    y.add_(0.0)  # warm-up of the kernel
    host0 = torch.zeros(64)
    torch.cuda.synchronize()
    with ptxv.capture(snapshots="all", names={"y": y, "u": u}) as cap:
        y.add_(1.0)
        y.copy_(host0)  # a host copy, not a launch
        u.add_(1.0)
    torch.cuda.synchronize()
    g = cap.all()
    for outs in (None, ["y", "u"]):
        c = g.case(outputs=outs)
        assert np.array_equal(c.expected()["y"], np.ones(64, np.float32))
        assert np.array_equal(c.expected()["u"], np.ones(64, np.float32))
        w = json.loads((c.path / "capture.json").read_text())["warnings"]
        assert any(
            m.startswith("y: changed after launch 0, the last of the group") and "MISS 219" in m
            for m in w
        ), w


def test_a_writer_the_case_writer_cannot_follow_counts_by_its_own_snapshots(torch):
    """MISS 219: a launch that writes y through an address the case writer cannot resolve (y's
    address plus 2^44, undone in the kernel) but whose snapshots, before and after it, cover y
    (it shares u's segment) counts as y's last reacher: its after-snapshot gives y's expected
    bytes. Under snapshots="first" its second launch takes none, and a later snapshot shows y
    changed again: that launch may have written it, so a case comparing y is refused."""
    triton = pytest.importorskip("triton")
    import ptxv
    import triton.language as tl
    from ptxv._casewriter import CaseError

    @triton.jit
    def add_u_and_hidden(u, hidden, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(u + o, tl.load(u + o) + 1.0)
        p = (hidden - (1 << 44)).to(tl.pointer_type(tl.float32))
        tl.store(p + o, tl.load(p + o) + 1.0)

    hide = 1 << 44
    scratch = torch.zeros(64, device="cuda")
    add_u_and_hidden[(1,)](scratch, scratch.data_ptr() + hide, 64)  # compile
    for snaps in ("all", "first"):
        y = torch.zeros(64, device="cuda")
        u = torch.zeros(64, device="cuda")
        w = torch.ones(64, device="cuda")
        y.add_(0.0)
        w.mul_(1.0)  # warm-up
        segs = {
            next(
                s["address"]
                for s in torch.cuda.memory_snapshot()
                if s["address"] <= t_.data_ptr() < s["address"] + s["total_size"]
            )
            for t_ in (y, u, w)
        }
        if len(segs) != 1:
            pytest.skip("y, u and w are not in one allocator segment")
        torch.cuda.synchronize()
        with ptxv.capture(snapshots=snaps, names={"y": y, "u": u, "w": w}) as cap:
            y.add_(1.0)
            add_u_and_hidden[(1,)](u, y.data_ptr() + hide, 64)
            if snaps == "first":
                add_u_and_hidden[(1,)](u, y.data_ptr() + hide, 64)  # no snapshot
                w.mul_(2.0)  # a snapshot of the segment
        torch.cuda.synchronize()
        g = cap.all()
        if snaps == "all":
            assert torch.equal(y, torch.full_like(y, 2.0))
            assert np.array_equal(g.case().expected()["y"], y.cpu().numpy())
        else:
            assert torch.equal(y, torch.full_like(y, 3.0))
            # outputs=["y"]: u's last reacher (the second hidden writer) has no snapshot either.
            with pytest.raises(CaseError, match="^y: it changed after launch 1.*MISS 219"):
                g.case(outputs=["y"])


def _one_segment(torch, *ts):
    """Whether the tensors share one caching-allocator segment (a snapshot of one covers all)."""
    return (
        len(
            {
                next(
                    s["address"]
                    for s in torch.cuda.memory_snapshot()
                    if s["address"] <= t_.data_ptr() < s["address"] + s["total_size"]
                )
                for t_ in ts
            }
        )
        == 1
    )


@pytest.mark.parametrize("graph", [False, True], ids=["launch", "graph"])
def test_a_change_outside_the_group_does_not_hide_a_later_write(torch, graph):
    """MISS 219: y's last reacher writes 1.0 and a host copy sets 2.0 (a change outside the
    group: a warning); then a launch without a snapshot adds 1 to y through w plus an offset (an
    address the writer resolves to w): the kernel's second launch under snapshots="first"
    (launch), or a launch recorded under CUDA graph capture and run by its replay (graph); a
    later snapshot shows 3.0. The after-check stopped at the first change, so the case was
    written with y = 1.0 where the GPU holds 3.0. Now later snapshots are compared with the
    bytes the outside change left: refused. Without the host copy (the control) the case was
    refused already. With the host copy and a launch without a snapshot that does not write y
    (it adds 1 to w), the case is written with y's bytes after its last reacher and the
    warning: the later snapshot still holds what the copy left."""
    triton = pytest.importorskip("triton")
    import json

    import ptxv
    import triton.language as tl
    from ptxv._casewriter import CaseError

    @triton.jit
    def fill_ones(out, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, tl.full((n,), 1.0, tl.float32))

    @triton.jit
    def mul_one(w, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(w + o, tl.load(w + o) * 1.0)

    @triton.jit
    def scale(w, s, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(w + o, tl.load(w + o) * s)

    @triton.jit
    def add_one_at(base, off, n: tl.constexpr):
        o = off + tl.arange(0, n)
        tl.store(base + o, tl.load(base + o) + 1.0)

    q = torch.zeros(64, device="cuda")
    fill_ones[(1,)](q, 64)  # compile every kernel outside the region
    mul_one[(1,)](q, 64)
    scale[(1,)](q, 1.0, 64)
    add_one_at[(1,)](q, 0, 64)
    twos = torch.full((64,), 2.0)
    for outside, hidden in ((True, True), (False, True), (True, False)):
        y = torch.zeros(64, device="cuda")
        w = torch.zeros(64, device="cuda")
        if not _one_segment(torch, y, w):
            pytest.skip("y and w are not in one allocator segment")
        off = (y.data_ptr() - w.data_ptr()) // 4 if hidden else 0
        g = torch.cuda.CUDAGraph() if graph else None
        torch.cuda.synchronize()
        snaps = "all" if graph else "first"
        with ptxv.capture(snapshots=snaps, names={"y": y, "w": w}) as cap:
            if not graph:
                add_one_at[(1,)](w, 0, 64)  # the kernel's first launch takes the snapshot
            fill_ones[(1,)](y, 64)  # y's last reacher
            if outside:
                y.copy_(twos)  # a host copy, not a launch
            mul_one[(1,)](w, 64)  # a snapshot of the segment
            if graph:
                with torch.cuda.graph(g):
                    add_one_at[(1,)](w, off, 64)  # recorded; the replay adds 1 to y
                g.replay()
            else:
                add_one_at[(1,)](w, off, 64)  # no snapshot; adds 1 to y (or to w)
            scale[(1,)](w, 1.0, 64)  # a snapshot of the segment
        torch.cuda.synchronize()
        assert torch.equal(y, torch.full_like(y, 1.0 + outside + hidden))
        if not hidden:
            c = cap.all().case()
            assert np.array_equal(c.expected()["y"], np.ones(64, np.float32))
            w_ = json.loads((c.path / "capture.json").read_text())["warnings"]
            assert any(m.startswith("y: changed after launch") for m in w_), w_
            continue
        with pytest.raises(CaseError, match="^y: it changed after .*MISS 219"):
            cap.all().case()


def test_a_hidden_write_inside_a_run_of_buffers_counts(torch):
    """MISS 219: the launches whose own snapshots show a buffer changed are found by comparing,
    for each run of buffers a launch does not reach, each buffer's own slice of the snapshots
    before and after it. Here y, written through an address the writer cannot resolve, follows v
    (written by an earlier launch) in its run, so its slice starts inside the run: compared at
    the run's start, the launch would leave y's expected bytes at 1.0 where the GPU holds 2.0."""
    triton = pytest.importorskip("triton")
    import ptxv
    import triton.language as tl

    @triton.jit
    def add_u_and_hidden(u, hidden, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(u + o, tl.load(u + o) + 1.0)
        p = (hidden - (1 << 44)).to(tl.pointer_type(tl.float32))
        tl.store(p + o, tl.load(p + o) + 1.0)

    hide = 1 << 44
    scratch = torch.zeros(64, device="cuda")
    add_u_and_hidden[(1,)](scratch, scratch.data_ptr() + hide, 64)  # compile
    v, y, u = sorted(
        (torch.zeros(64, device="cuda") for _ in range(3)), key=lambda t_: t_.data_ptr()
    )
    if not _one_segment(torch, v, y, u):
        pytest.skip("v, y and u are not in one allocator segment")
    v.add_(0.0)  # warm-up
    torch.cuda.synchronize()
    with ptxv.capture(snapshots="all", names={"v": v, "y": y, "u": u}) as cap:
        v.add_(1.0)
        y.add_(1.0)
        add_u_and_hidden[(1,)](u, y.data_ptr() + hide, 64)  # y after v in the run
    torch.cuda.synchronize()
    assert torch.equal(y, torch.full_like(y, 2.0))
    c = cap.all().case()
    assert np.array_equal(c.expected()["y"], y.cpu().numpy())
    assert np.array_equal(c.expected()["v"], np.ones(64, np.float32))
