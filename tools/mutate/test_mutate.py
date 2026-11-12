# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the mutation tool and the harness's pure parts (no ptxv, no GPU).

$PTXV_PYTHON -m pytest -q tools/mutate
"""

import json
import os
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
import mutate  # noqa: E402
import srcmut  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

PTX = """\
.version 9.0
.target sm_100a
.address_size 64
.visible .entry k(.param .u64 p)
{
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
    mbarrier.init.shared::cta.b64 [%r1], 1;
    fence.mbarrier_init.release.cluster;
    bar.sync 0;                         // all
WAIT:
    mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 %p1, [%r1], %r2;
    @!%p1 bra WAIT;
L2: mbarrier.try_wait.parity.shared::cta.b64 %p2, [%r1], 1;
    @!%p2 bra L2;
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r1], 128;
    mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%r3];
    fence.proxy.async.shared::cta;
    fence.acq_rel.gpu;
    membar.gl;
    tcgen05.fence::after_thread_sync;
    tcgen05.wait::ld.sync.aligned;
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group 0;
    tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r1];
    ld.acquire.gpu.global.u32 %r4, [%rd1];
    st.release.sys.global.u32 [%rd1], %r4;
    atom.relaxed.gpu.global.add.u32 %r5, [%rd1], 1;
    barrier.cluster.arrive.release.aligned;
    barrier.cluster.wait.aligned;
    // bar.sync 1;   a comment, not an instruction
    ret;
}
"""


def by_id(ms):
    return {m.id: m for m in ms}


def test_parse_insts_skips_comments_and_directives():
    ins = mutate.parse_insts(PTX.split("\n"))
    ops = [i.opcode for i in ins]
    assert (
        "bar.sync" in ops and ops.count("bar.sync") == 1
    )  # the commented one is skipped
    assert not any(o.startswith(".") for o in ops)
    l2 = next(i for i in ins if i.label == "L2")
    assert l2.opcode.startswith("mbarrier.try_wait")
    assert l2.text.startswith(
        "mbarrier.try_wait"
    )  # the label is not part of the find text


def test_operators():
    ms = by_id(mutate.ptx_mutants(PTX))
    ops = {m.op for m in ms.values()}
    assert ops >= {
        "del-bar",
        "del-wait",
        "parity",
        "relax",
        "scope",
        "drop-expect-tx",
        "del-arrive",
        "del-fence",
        "del-commit",
        "weaken-wait",
    }
    # wait loop: the try_wait succeeds at once
    assert ms["del-wait-L12"].edits[0].replace == "mov.pred %p1, 1;"
    # immediate parity flipped, register parity flipped through a scoped temp
    assert (
        ms["parity-L14"].edits[0].replace
        == "mbarrier.try_wait.parity.shared::cta.b64 %p2, [%r1], 0;"
    )
    assert "xor.b32 %ptxv_mut_par, %r2, 1" in ms["parity-L12"].edits[0].replace
    # acquire -> relaxed, cluster -> cta only on the scope (not on ::cluster state spaces)
    assert ".relaxed.cluster" in ms["relax-L12"].edits[0].replace
    sc = ms["scope-L17"].edits[0].replace
    assert ".release.cta.shared::cluster" in sc
    # expect_tx dropped keeps the arrive
    assert (
        ms["drop-expect-tx-L16"].edits[0].replace
        == "mbarrier.arrive.shared::cta.b64 _, [%r1];"
    )
    assert ms["weaken-wait-L24"].edits[0].replace == "cp.async.bulk.wait_group 1;"
    assert (
        ms["weaken-wait-L24.1"].edits[0].replace == "cp.async.bulk.wait_group.read 0;"
    )
    assert ms["scope-L27"].edits[0].replace.startswith("st.release.gpu.global")
    assert ms["relax-L26"].edits[0].replace.startswith("ld.relaxed.gpu.global")
    assert ms["scope-L28"].edits[0].replace.startswith("atom.relaxed.cta.global")
    assert ms["relax-L29"].edits[0].replace == "barrier.cluster.arrive.relaxed.aligned;"
    # fence.proxy and mbarrier_init fences are not rescoped; acq_rel.gpu is
    assert not any(m.op == "scope" and m.lines == [18] for m in ms.values())
    assert not any(m.op == "scope" and m.lines == [9] for m in ms.values())
    assert ms["scope-L19"].edits[0].replace == "fence.acq_rel.cta;"
    assert ms["scope-L20"].edits[0].replace == "membar.cta;"
    # ids are unique
    assert len(ms) == len(mutate.ptx_mutants(PTX))


def test_order_and_atomicity_operators():
    src = """\
.visible .entry k(.param .u64 p)
{
    st.global.u32 [%rd2], %r3;
    fence.acq_rel.gpu;
    atom.relaxed.gpu.global.add.u32 %r4, [%rd3], 1;
    red.relaxed.gpu.global.add.u32 [%rd3], 1;
    ld.acquire.gpu.global.u32 %r5, [%rd3];
L1: fence.acq_rel.gpu;
    ret;
}
"""
    ms = by_id(mutate.ptx_mutants(src))
    # the fence after the next access, and before the previous one
    after = ms["move-fence-L4"]
    assert [e.replace for e in after.edits] == [
        "atom.relaxed.gpu.global.add.u32 %r4, [%rd3], 1;",
        "fence.acq_rel.gpu;",
    ]
    before = ms["move-fence-L3"]
    assert [e.line for e in before.edits] == [2, 3]
    assert before.edits[0].replace == "fence.acq_rel.gpu;"
    # a labelled fence is not moved
    assert not any(m.op == "move-fence" and 8 in m.lines for m in ms.values())
    # acquire fences after relaxed RMWs, not after an acquiring load
    acq = {m.lines[0] for m in ms.values() if m.op == "acq-fence"}
    assert acq == {5, 6}
    assert ms["acq-fence-L5"].edits[0].replace.endswith("1; fence.acq_rel.gpu;")
    # the used atomic add split into ld / add / st
    sp = ms["atom-plain-L5"].edits[0].replace
    assert "ld.relaxed.gpu.global.u32 %r4, [%rd3];" in sp
    assert "add.u32 %ptxv_mut_t, %r4, 1;" in sp
    assert "st.relaxed.gpu.global.u32 [%rd3], %ptxv_mut_t;" in sp


def test_strip_variants_keeps_rest():
    t = (ROOT / "fixtures/e_bararv/case.toml").read_text()
    s = mutate.strip_variants(t)
    m = tomllib.loads(s)
    assert "variants" not in m
    assert m["name"] == "e_bararv" and m["buffers"] and m["params"]
    # a header-looking line inside a multi-line string is not a header
    t2 = 'name = "x"\ndescription = """\n[[variants]]\n"""\nentry = "k"\n[[variants]]\nid = "good"\nexpect = []\n'
    m2 = tomllib.loads(mutate.strip_variants(t2))
    assert m2["entry"] == "k" and "[[variants]]" in m2["description"]


def test_write_mutant_roundtrip(tmp_path):
    case = tmp_path / "c"
    case.mkdir()
    (case / "kernel.ptx").write_text(PTX)
    (case / "case.toml").write_text(
        'name = "c"\ndescription = ""\nptx = "kernel.ptx"\nentry = "k"\narch = "sm_100a"\n'
        'grid = [1,1,1]\nblock = [32,1,1]\noutputs = []\n\n[[variants]]\nid = "good"\nexpect = ["clean"]\n'
        '\n[[variants]]\nid = "fix"\nmarker = "all"\nfind = "bar.sync 0;"\nreplace = "bar.sync 0;  bar.sync 0;"\n'
        'expect = ["clean"]\n'
    )
    (case / "data").mkdir()
    ds = mutate.generate(case, tmp_path / "m", base="fix", arch="sm_103a")
    assert ds
    for d in ds:
        man = tomllib.loads((d / "case.toml").read_text())
        assert man["arch"] == "sm_103a"
        assert [v["id"] for v in man["variants"]] == ["good", "mut"]
        assert (d / "data").is_symlink()
        # Relative links: a copy of the runs and case trees side by side still resolves.
        assert not os.path.isabs(os.readlink(d / "data"))
        assert (d / "data").resolve() == (case / "data").resolve()
        src = (d / "kernel.ptx").read_text()
        assert ".target sm_103a" in src
        good = mutate.apply_variant_ptx(src, man["variants"][0])
        mut = mutate.apply_variant_ptx(src, man["variants"][1])
        assert "bar.sync 0;  bar.sync 0;" in good  # the base variant is materialized
        assert good != mut
        assert len(good.split("\n")) == len(
            mut.split("\n")
        )  # line numbers kept (for --delay)
        mj = json.loads((d / "mutant.json").read_text())
        assert mj["base_variant"] == "fix"


def test_sass_mutants():
    dis = """
        .section .text.k,"ax",@progbits
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0010*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;
        /*0020*/              @!P0 SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [UR4], R3 ;
        /*0030*/                   MEMBAR.ALL.GPU ;
        /*0040*/                   FENCE.VIEW.ASYNC.S ;
        /*0050*/                   ERRBAR ;
        /*0060*/                   CGAERRBAR ;
        /*0070*/                   EXIT ;
"""
    ks = mutate.cubin_kernels(dis)
    ms = mutate.sass_mutants(ks["k"])
    assert [m.patch[0]["at"] for m in ms] == [
        "@0x0010",
        "@0x0020",
        "@0x0030",
        "@0x0040",
        "@0x0050",
        "@0x0060",
    ]
    assert [m.op for m in ms[4:]] == ["sass-del-fence", "sass-del-fence"]
    assert all(m.patch[0]["nop"] for m in ms)
    assert (
        mutate._toml_patch(ms[0].patch)
        == '[{ at = "@0x0010", nop = true, wbar = 7, rbar = 7, expect = "NOP" }]'
    )


def test_classify_check():
    row = lambda got, complete=True: json.dumps(
        {
            "rows": [
                {
                    "got": got,
                    "verdict": {
                        "complete": complete,
                        "states": 3,
                        "findings": [],
                        "assumptions": ["E-WELLSYNC"],
                    },
                }
            ]
        }
    )
    assert harness.classify_check(0, row(["clean"]), "", False)["status"] == "clean"
    assert harness.classify_check(1, row(["race"]), "", False)["status"] == "caught"
    assert (
        harness.classify_check(1, row(["contract"]), "", False)["status"] == "contract"
    )
    assert (
        harness.classify_check(1, row(["clean"], False), "", False)["status"]
        == "budget"
    )
    assert (
        harness.classify_check(1, row(["deadlock"], False), "", False)["status"]
        == "caught"
    )
    assert (
        harness.classify_check(1, '{"error": "ptxas"}', "", False)["status"]
        == "rejected"
    )
    assert harness.classify_check(124, "", "", True)["status"] == "budget"
    # The certificate run's race (printed by ptxv before it explores) decides an exploration
    # that ran out of budget or time.
    err = (
        "certificate for launch (LDS/STS plain) does not hold (25000 steps, 9.0s); exploring instead. Reasons:\n"
        "  - the run reported findings (race); the explorer decides them: first: race on ...\n"
        "certificate run race (launch (LDS/STS strong)): race:proxy:123:456: proxy race on smem byte 0x1a00\n"
    )
    r = harness.classify_check(1, row(["clean"], False), err, False)
    assert (r["status"], r["cert_run"]) == ("caught", True), r
    assert "race:proxy:123:456" in r["why"]
    r = harness.classify_check(124, "", err, True)
    assert (r["status"], r["cert_run"]) == ("caught", True), r
    # A run's findings alone (no standing race printed) do not count.
    only = "\n".join(err.splitlines()[:2]) + "\n"
    assert (
        harness.classify_check(1, row(["clean"], False), only, False)["status"]
        == "budget"
    )
    assert (
        harness.cert_run_race("certificate run race (cluster 3): race:x: y\n")
        == "cluster 3: race:x"
    )
    # A complete clean exploration stays clean whatever the certificate run said.
    assert harness.classify_check(0, row(["clean"]), err, False)["status"] == "clean"


def test_delay_plans_hit_sync_lines():
    lines = PTX.split("\n")
    sites = harness.sync_lines(PTX)
    assert sites and all(
        not lines[s - 1].strip().startswith((".", "{", "}")) for s in sites
    )
    plans = harness.delay_plans(PTX, [12], 10, seed=1)
    assert len(plans) == 10
    for p in plans:
        for d in p:
            l, ns = d.split(":")
            assert int(l) in sites and int(ns) > 0
    assert plans == harness.delay_plans(PTX, [12], 10, seed=1)  # reproducible


def test_hw_line_parse():
    m = harness._HW_LINE.match(
        "hw mut   3/200 runs differ from the correct output (200 runs, 4.1s) distinct"
    )
    assert m and m.group(1) == "mut" and m.group(3) == "200"


def test_mutant_of_materialized_base_has_own_files(tmp_path):
    """Mutants of a written case (e.g. a materialized base variant) get their own mutant.json
    and kernel, not symlinks into the base."""
    case = tmp_path / "c"
    case.mkdir()
    (case / "kernel.ptx").write_text(PTX)
    (case / "case.toml").write_text(
        'name = "c"\ndescription = ""\nptx = "kernel.ptx"\nentry = "k"\n'
        'grid = [1,1,1]\nblock = [32,1,1]\noutputs = []\n\n[[variants]]\nid = "good"\nexpect = ["clean"]\n'
    )
    base = mutate.write_mutant(
        case,
        tmp_path / "base",
        mutate.Mutant(id="base", op="none", summary=""),
        PTX,
        None,
    )
    ds = mutate.generate(base, tmp_path / "m")
    ids = set()
    for d in ds:
        assert (
            not (d / "mutant.json").is_symlink() and not (d / "kernel.ptx").is_symlink()
        )
        ids.add(json.loads((d / "mutant.json").read_text())["id"])
    assert len(ids) == len(ds)
    assert json.loads((base / "mutant.json").read_text())["id"] == "base"
    # the reach probe of a mutant: a trap at the mutated line
    p = mutate.write_probe(ds[0], tmp_path / "probe")
    man = tomllib.loads((p / "case.toml").read_text())
    assert man["variants"][1]["replace"].endswith("trap;")


def test_emit_and_ingest_runner_tasks(tmp_path):
    """check --emit-jobs writes one task per pending run; --ingest records finished ones."""
    runs = tmp_path / "runs"
    d = runs / "mutants" / "c" / "m1"
    d.mkdir(parents=True)
    (d / "mutant.json").write_text(json.dumps({"op": "del-bar", "summary": "s"}))
    (runs / "baseline.jsonl").write_text(
        json.dumps(
            {
                "mode": "cert",
                "status": "clean",
                "case": "c",
                "path": str(tmp_path),
                "src": str(tmp_path),
                "base": "good",
                "front": "ptx",
                "got": ["clean"],
            }
        )
        + "\n"
    )
    jobs = tmp_path / "jobs.txt"
    harness.main(
        [
            "check",
            "--runs",
            str(runs),
            "--modes",
            "cert",
            "--emit-jobs",
            str(jobs),
            "--job-prefix",
            "P",
        ]
    )
    line = jobs.read_text().strip()
    assert line.startswith(f"P/c/m1/ptx-cert {d} ")
    assert "--certify" in line and "{case}" in line
    res = tmp_path / "res" / "P" / "c" / "m1"
    res.mkdir(parents=True)
    doc = {
        "rows": [
            {
                "got": ["race"],
                "verdict": {"complete": True, "findings": [], "assumptions": []},
            }
        ]
    }
    (res / "ptx-cert.json").write_text(json.dumps(doc) + "\n")
    (res / "ptx-cert.log").write_text("")
    (res / "ptx-cert.exit").write_text("1\n")
    harness.main(
        [
            "check",
            "--runs",
            str(runs),
            "--modes",
            "cert",
            "--ingest",
            str(tmp_path / "res"),
            "--job-prefix",
            "P",
        ]
    )
    recs = [json.loads(x) for x in (runs / "check.jsonl").read_text().splitlines()]
    assert [(r["mutant"], r["status"], r["runner"]) for r in recs] == [
        ("m1", "caught", True)
    ]



def test_srcmut_check_anchors(tmp_path, monkeypatch, capsys):
    """srcmut.py --check-anchors prints every mutant whose find text does not occur exactly once
    in its file, or occurs once but only inside a comment, and exits 1 if there is one; it needs
    no --out and builds and runs nothing."""
    (tmp_path / "a.rs").write_text("fn a() {\n    keep(x);\n}\n")
    b = tmp_path / "b.rs"
    good = srcmut.SrcMutant("good", "a.rs", "    keep(x);\n", "", [])
    broken = srcmut.SrcMutant("broken", "b.rs", "    drop(y);\n", "", [])
    monkeypatch.setattr(srcmut, "ROOT", tmp_path)
    monkeypatch.setattr(srcmut, "MUTANTS", [good, broken])

    def never(*args, **kwargs):
        raise AssertionError("--check-anchors built or ran a mutant")

    monkeypatch.setattr(srcmut, "run_mutant", never)
    monkeypatch.setattr(srcmut, "build", never)

    def check(*args):
        rc = srcmut.main(["--check-anchors", *args])
        out = capsys.readouterr()
        return rc, [ln.split(None, 1) for ln in out.out.splitlines()], out.err.strip()

    b.write_text("    drop(y);\n    drop(y);\n")
    assert check() == (1, [["broken", "b.rs: find text occurs 2 times"]], "1 of 2 anchors broken")
    assert check("--only", "good") == (0, [], "0 of 1 anchors broken")
    b.write_text("    keep(y);\n")
    assert check() == (1, [["broken", "b.rs: find text occurs 0 times"]], "1 of 2 anchors broken")
    b.unlink()
    assert check() == (1, [["broken", "b.rs: no such file"]], "1 of 2 anchors broken")
    b.write_text("    drop(y);\n")
    assert check() == (0, [], "0 of 2 anchors broken")
    # A find text that occurs once but only inside a comment is no anchor either: the mutant
    # would edit the comment. One in a string literal, or next to a comment, is code; so is a
    # comment marker inside a literal.
    rs = srcmut.SrcMutant("cmt", "c.rs", "    keep(z);\n", "", [])
    py = srcmut.SrcMutant("pcmt", "p.py", "    keep(z)\n", "", [])
    monkeypatch.setattr(srcmut, "MUTANTS", [good, broken, rs, py])
    (tmp_path / "p.py").write_text("def p():\n    keep(z)\n")
    clean = (0, [], "0 of 4 anchors broken")
    rs_cmt = (1, [["cmt", "c.rs: find text occurs once, inside a comment"]], "1 of 4 anchors broken")
    for text, want in [
        ("fn c() {\n    keep(z);\n}\n", clean),
        ("/*\n    keep(z);\n*/\nfn c() {\n    keep(y);\n}\n", rs_cmt),
        ("fn c() {\n//    keep(z);\n    keep(y);\n}\n", rs_cmt),
        ("/* outer /* inner */\n    keep(z);\n*/\nfn c() {}\n", rs_cmt),
        ('fn c() {\n    let u = "http://x";\n    keep(z);\n}\n', clean),
        ("fn c() {\n    let q = '\"'; let s = \"/*\";\n    keep(z);\n}\n", clean),
        ('fn c() {\n    let r = r#"/* "quoted" */"#;\n    keep(z);\n}\n', clean),
        ("fn c<'a>(x: &'a str) {\n    keep(z);\n    // why\n}\n", clean),
    ]:
        (tmp_path / "c.rs").write_text(text)
        assert check() == want, text
    py_cmt = (1, [["pcmt", "p.py: find text occurs once, inside a comment"]], "1 of 4 anchors broken")
    for text, want in [
        ("def p():\n    #    keep(z)\n    pass\n", py_cmt),
        ('def p():\n    s = "# no"\n    keep(z)\n', clean),
        ("def p():\n    '''# no\n    '''\n    keep(z)\n    # yes\n", clean),
    ]:
        (tmp_path / "p.py").write_text(text)
        assert check() == want, text
    monkeypatch.setattr(srcmut, "MUTANTS", [good, broken])
    # --root checks the files of another tree (the anchors test's positive control plants a
    # broken anchor in a scratch tree); a run takes no --root.
    other = tmp_path / "other"
    other.mkdir()
    (other / "a.rs").write_text("    keep(x);\n    keep(x);\n")
    (other / "b.rs").write_text("    drop(y);\n")
    assert check("--root", str(other)) == (1, [["good", "a.rs: find text occurs 2 times"]], "1 of 2 anchors broken")
    assert check("--root", str(other), "--only", "broken") == (0, [], "0 of 1 anchors broken")
    with pytest.raises(SystemExit):
        srcmut.main(["--root", str(other)])
    capsys.readouterr()  # argparse's usage message
    # No mutant chosen is an error (exit 2), not "0 of 0 anchors broken": an --only name that is
    # not a mutant, or no mutants at all.
    assert check("--only", "no-such") == (2, [], "srcmut: --only names no mutant that runs: no-such")
    assert check("--only", "good,no-such")[0] == 2
    # An --only that is given but names nothing exits 2 too (it chose every mutant).
    for names in ["", ",", ",,", " ", " , "]:
        assert check("--only", names) == (2, [], "srcmut: --only names nothing"), names
    assert check("--only=") == (2, [], "srcmut: --only names nothing")
    monkeypatch.setattr(srcmut, "MUTANTS", [])
    assert check() == (2, [], "srcmut: no mutant is chosen")
    # A run still needs --out.
    monkeypatch.setattr(srcmut, "MUTANTS", [good, broken])
    with pytest.raises(SystemExit):
        srcmut.main([])


def test_srcmut_anchors_whole_list(tmp_path, capsys):
    """--check-anchors over the whole mutant list, without --only, in a scratch tree that holds
    every file the mutants name (--root): exit 1 if some find text is not an anchor (it does not
    occur exactly once, or only inside a comment), naming exactly those mutants, before and after
    the anchors of the last mutant and of a mutant whose file an earlier mutant names are broken
    (their find texts written twice). A check that drops a tail of the list, or checks one mutant
    per file, passes test_srcmut_check_anchors' short lists; it fails here."""
    ms = srcmut.MUTANTS
    for rel in {m.file for m in ms}:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((srcmut.ROOT / rel).read_text())

    def check():
        rc = srcmut.main(["--check-anchors", "--root", str(tmp_path)])
        out = capsys.readouterr()
        return rc, sorted(ln.split(None, 1)[0] for ln in out.out.splitlines()), out.err.strip()

    def want():
        # The find-text count of each mutant, read here, independently of srcmut; a find text
        # that occurs once is still broken when it is kept only inside a comment (srcmut's comment
        # rule, whose cases test_srcmut_check_anchors runs), as in a tree whose anchor was
        # commented out.
        bad, texts = [], {}
        for m in ms:
            if m.file not in texts:
                text = (tmp_path / m.file).read_text()
                texts[m.file] = (text, srcmut.comment_spans(text, m.file) or [])
            text, spans = texts[m.file]
            if text.count(m.find) != 1 or srcmut.anchor_problem(text, m.file, m.find, spans):
                bad.append(m.name)
        return 1 if bad else 0, sorted(bad), f"{len(bad)} of {len(ms)} anchors broken"

    assert check() == want()
    last = ms[-1]
    first = {}
    for i, m in enumerate(ms):
        first.setdefault(m.file, i)
    shared = next(m for i, m in enumerate(ms[:-1]) if first[m.file] < i and m.file != last.file)
    for m in (shared, last):
        f = tmp_path / m.file
        f.write_text(f.read_text().replace(m.find, m.find * 2))
    w = want()
    assert {shared.name, last.name} <= set(w[1]), (shared.name, last.name, w)
    assert check() == w


def test_srcmut_only_names_a_mutant_that_runs(tmp_path):
    # An unknown or excluded name in --only fails before anything runs (it used to run nothing
    # and exit 0).
    import subprocess
    srcmut = Path(__file__).resolve().parent / "srcmut.py"
    for names in ["no-such-mutant", "atom-sources-off", "dd-refuse-off,no-such-mutant"]:
        r = subprocess.run([sys.executable, str(srcmut), "--out", str(tmp_path / "o"), "--only", names],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 2, (names, r.stdout, r.stderr)
        assert "--only names no mutant that runs" in r.stderr, r.stderr
    # An --only that is given but names nothing fails the same way (it ran every mutant).
    for args in (["--only", ""], ["--only", ","], ["--only", ",,"], ["--only", " "], ["--only="]):
        r = subprocess.run([sys.executable, str(srcmut), "--out", str(tmp_path / "o"), *args],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 2, (args, r.stdout, r.stderr)
        assert "--only names nothing" in r.stderr, r.stderr
    assert not (tmp_path / "o").exists()
