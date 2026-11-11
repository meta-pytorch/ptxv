# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scratch (dead workspace) buffers from the capture side: `ptxv first-access` names buffers
whose first access is a write or a discard, the case writer lists them as suggestions (never
applied), and a declared scratch buffer is `init = "undef"`, `scratch = true`, not an output.
"""

import json
import os
import subprocess
from pathlib import Path

from conftest import needs_binary

# ws: written (or L2-discarded) before any read; x: read first.
PTX = """.version 8.8
.target sm_103a
.address_size 64

.visible .entry k(.param .u64 x, .param .u64 ws, .param .u64 out)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [ws];
    ld.param.u64 %rd3, [out];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.u32 %r2, [%rd5];
    st.global.u32 [%rd6], %r2;
    ld.global.u32 %r3, [%rd6];
    add.u32 %r4, %r3, 1;
    st.global.u32 [%rd7], %r4;
    DISCARD
    ret;
}
"""


def write_case(d: Path, discard: bool, ws_init: str = 'init = "zeros"') -> Path:
    d.mkdir(parents=True)
    (d / "expected").mkdir()
    ptx = PTX.replace(
        "DISCARD",
        "bar.sync 0;\n    @%p1 discard.global.L2 [%rd2], 128;" if discard else "",
    )
    if discard:
        ptx = ptx.replace(
            ".reg .b32 %r<6>;", ".reg .b32 %r<6>;\n    .reg .pred %p1;"
        ).replace(
            "add.u32 %r4, %r3, 1;", "add.u32 %r4, %r3, 1;\n    setp.eq.u32 %p1, %r1, 0;"
        )
    (d / "kernel.ptx").write_text(ptx)
    (d / "expected" / "out.bin").write_bytes(
        b"".join((1).to_bytes(4, "little") for _ in range(32))
    )
    buf = lambda n, extra: (  # noqa: E731
        f'[[buffers]]\nname = "{n}"\ndtype = "u32"\nshape = [32]\n{extra}\n'
    )
    (d / "case.toml").write_text(
        'name = "scratch_probe"\ndescription = """test"""\nptx = "kernel.ptx"\n'
        'entry = "k"\narch = "sm_103a"\ngrid = [1, 1, 1]\nblock = [32, 1, 1]\n'
        'outputs = ["out"]\n\n'
        + buf("x", 'init = "zeros"')
        + buf("ws", ws_init)
        + buf("out", 'init = "zeros"\nexpect = "expected/out.bin"')
        + "".join(
            f'[[params]]\nkind = "ptr"\nbuffer = "{n}"\nbyte_offset = 0\n\n'
            for n in ("x", "ws", "out")
        )
        + '[[variants]]\nid = "good"\nsummary = "as captured"\nexpect = ["clean"]\n'
    )
    return d


def ptxv(*args):
    from ptxv import _binary

    return subprocess.run(
        [str(_binary.find(build=False)), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PTXV_QUIET": "1"},
    )


@needs_binary
def test_first_access_names_written_and_discarded_buffers(tmp_path):
    for discard in (False, True):
        d = write_case(tmp_path / f"c{int(discard)}", discard)
        r = ptxv("first-access", str(d), "--front", "ptx", "--out", str(tmp_path / "o"))
        assert r.returncode == 0, r.stderr
        first = json.loads(r.stdout)
        assert first["x"]["first"] == "read", first
        assert first["ws"]["first"] == "write" and first["ws"]["read_back"], first
        assert first["out"]["first"] == "write" and not first["out"]["read_back"], first


@needs_binary
def test_suggestions_are_listed_not_applied(tmp_path):
    from ptxv._casewriter import scratch_suggestions

    d = write_case(tmp_path / "c", discard=True)
    before = (d / "case.toml").read_text()
    sug = scratch_suggestions(d, exclude={"out"})
    assert [s["buffer"] for s in sug] == ["ws"], sug
    assert (d / "case.toml").read_text() == before
    # A buffer already declared scratch is not suggested again.
    d2 = write_case(
        tmp_path / "c2", discard=True, ws_init='init = "undef"\nscratch = true'
    )
    assert [s["buffer"] for s in scratch_suggestions(d2, exclude={"out"})] == []


@needs_binary
def test_a_declared_scratch_buffer_checks_clean(tmp_path):
    # The discarded workspace declared dead (S-INIT-DEAD): undefined from the start, the
    # discard leaves undefined data in it at the end, a note and not a violation. The same
    # case with ws as an input buffer reports it (uninit-gmem-exit). PTX front: the SASS
    # front does not model CCTL.E.RML2 yet.
    d = write_case(
        tmp_path / "c", discard=True, ws_init='init = "undef"\nscratch = true'
    )
    r = ptxv("check", str(d), "--front", "ptx", "--json", "--out", str(tmp_path / "o1"))
    row = json.loads(r.stdout)["rows"][0]
    assert row["got"] == ["clean"], row
    assert any(
        f.get("kind") == "note" and "dead-undef-exit" in (f.get("sig") or "")
        for f in row["verdict"]["findings"]
    ), row
    d2 = write_case(tmp_path / "c2", discard=True)
    r = ptxv(
        "check", str(d2), "--front", "ptx", "--json", "--out", str(tmp_path / "o2")
    )
    assert "leaves undefined data in global byte" in r.stderr, r.stderr
    assert "declared dead workspace" not in r.stderr, r.stderr
