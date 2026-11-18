# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""tools/capture.py's cluster lines (`launch_cluster`, `cluster_note`, `cluster_toml`), taken
from the script's source (it runs a capture when imported). A launchcap record with
`cluster_attr` is a cluster launch even at 1 x 1 x 1: the kernel description names its cluster,
and the first launch and every [[launches]] entry give `cluster = [1, 1, 1]`, so st.async runs in
the model as it ran on GB300 (MISS 259; outside a cluster launch it faults, MISS 209). launchcap
writes `cluster_attr` for the cluster dimension attribute (MISS 259) and for a cluster shape the
kernel requires (the RequiredCluster function attributes, recorded as the launch's `cluster`;
MISS 281), and `required_cluster` where the kernel requires a shape other than the launch
attribute's: the description names it. A record without either and without a preferred cluster
gives no `cluster` line. No GPU. unittest cases: tools/mutate/srcmut.py runs a mutant's python
tests with unittest, which collects no plain test functions (pytest runs these too)."""

import ast
import pathlib
import unittest
from types import SimpleNamespace

SRC = pathlib.Path(__file__).resolve().parents[2] / "tools" / "capture.py"
TREE = ast.parse(SRC.read_text())
NAMES = ("launch_cluster", "cluster_note", "cluster_toml")


def helpers():
    keep = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    assert sorted(n.name for n in keep) == sorted(NAMES)
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"), ns)
    return SimpleNamespace(**{k: ns[k] for k in NAMES})


def rec(cluster, **kw):
    return {"cluster": cluster, **kw}


class CaptureCluster(unittest.TestCase):
    def test_attribute_1x1x1_is_a_cluster_launch(self):
        h = helpers()
        assert h.launch_cluster(rec([1, 1, 1], cluster_attr=1))
        assert h.cluster_toml(rec([1, 1, 1], cluster_attr=1)) == "cluster = [1, 1, 1]\n"
        assert h.cluster_note(rec([1, 1, 1], cluster_attr=1)) == " cluster [1, 1, 1]"

    def test_no_attribute_gives_no_cluster_line(self):
        h = helpers()
        for r in (rec([1, 1, 1]), rec([1, 1, 1], cluster_attr=0)):
            assert not h.launch_cluster(r)
            assert h.cluster_toml(r) == ""
            assert h.cluster_note(r) == ""

    def test_shapes_and_preferred_clusters_as_before(self):
        h = helpers()
        assert h.launch_cluster(rec([2, 1, 1]))
        assert h.cluster_toml(rec([2, 1, 1])) == "cluster = [2, 1, 1]\n"
        assert h.cluster_toml(rec([2, 1, 1], cluster_attr=1)) == "cluster = [2, 1, 1]\n"
        assert not h.launch_cluster(rec([1, 1, 1], preferred_cluster=[2, 1, 1]))
        assert h.cluster_toml(rec([1, 1, 1], preferred_cluster=[2, 1, 1])) == (
            "cluster = [1, 1, 1]\npreferred_cluster = [2, 1, 1]\n"
        )

    def test_required_cluster_conflict_is_named(self):
        # MISS 281: the launch attribute's shape is the launch's (launchcap records it as `cluster`)
        # and the kernel's other required shape is named in the description; the lines are the
        # launch's.
        h = helpers()
        r = rec([2, 1, 1], cluster_attr=1, required_cluster=[1, 1, 1])
        assert h.cluster_toml(r) == "cluster = [2, 1, 1]\n"
        assert h.cluster_note(r) == (
            " cluster [2, 1, 1] (the kernel requires [1, 1, 1]; the launch attribute is taken)"
        )
        assert h.cluster_note(rec([2, 1, 1], cluster_attr=1)) == " cluster [2, 1, 1]"

    def test_every_site_uses_the_helpers(self):
        # The kernel description, the first launch and each [[launches]] entry: no other test of
        # a record's cluster fields decides a line.
        text = SRC.read_text()
        assert 'f" cluster {r[\'cluster\']}" if launch_cluster(r) else ""' in text
        assert "+ cluster_note(r)\n" in text
        assert "+ cluster_toml(r0)" in text
        assert "s += cluster_toml(r)" in text
        calls = [
            n
            for n in ast.walk(TREE)
            if isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Constant)
            and n.slice.value == "cluster"
            and isinstance(n.value, ast.Name)
            and n.value.id in ("r", "r0")
        ]
        gets = [
            n
            for n in ast.walk(TREE)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "get"
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == "cluster_attr"
        ]
        # launch_cluster's own read of cluster_attr is the only one.
        assert len(gets) == 1, [ast.unparse(g) for g in gets]
        assert calls, "no record cluster reads"
