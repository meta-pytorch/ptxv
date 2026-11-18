# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The case writer's cluster lines (_casewriter._cluster_toml) and description (_cluster_note).
A launch with `cluster_attr` in its record (python/ptxv/_native/ptxvcap.c) is a cluster launch
even at 1 x 1 x 1: its case gives `cluster = [1, 1, 1]`, so st.async runs in the model as it ran
on GB300 (MISS 259; outside a cluster launch it faults, MISS 209). ptxvcap.c writes `cluster_attr`
for the cluster dimension attribute (MISS 259) and for a cluster shape the kernel requires (the
RequiredCluster function attributes, recorded as the launch's `cluster`; MISS 281), and
`required_cluster` where the kernel requires a shape other than the launch attribute's: the
description names it. A launch without either and without a preferred cluster gives no `cluster`
line. No GPU. unittest cases: tools/mutate/srcmut.py runs a mutant's python tests with unittest,
which collects no plain test functions (pytest runs these too)."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptxv._casewriter import _cluster_note, _cluster_toml  # noqa: E402


def launch(cluster, **record):
    return SimpleNamespace(cluster=cluster, record=record)


class CasewriterCluster(unittest.TestCase):
    def test_attribute_1x1x1_is_a_cluster_launch(self):
        assert _cluster_toml(launch((1, 1, 1), cluster_attr=1)) == "cluster = [1, 1, 1]\n"

    def test_no_attribute_gives_no_cluster_line(self):
        assert _cluster_toml(launch((1, 1, 1))) == ""
        assert _cluster_toml(launch((1, 1, 1), cluster_attr=0)) == ""
        assert _cluster_note(launch((1, 1, 1))) == ""

    def test_shapes_and_preferred_clusters_as_before(self):
        assert _cluster_toml(launch((2, 1, 1), cluster_attr=1)) == "cluster = [2, 1, 1]\n"
        assert _cluster_toml(launch((2, 1, 1))) == "cluster = [2, 1, 1]\n"
        assert _cluster_toml(launch((1, 1, 1), preferred_cluster=[2, 1, 1])) == (
            "cluster = [1, 1, 1]\npreferred_cluster = [2, 1, 1]\n"
        )
        assert _cluster_toml(launch((2, 1, 1), preferred_cluster=[4, 1, 1], cluster_attr=1)) == (
            "cluster = [2, 1, 1]\npreferred_cluster = [4, 1, 1]\n"
        )

    def test_required_cluster_conflict_is_named(self):
        # MISS 281: the launch attribute's shape is the launch's (ptxvcap.c records it as `cluster`)
        # and the kernel's other required shape is named in the description; the lines are the
        # launch's.
        la = launch((2, 1, 1), cluster_attr=1, required_cluster=[1, 1, 1])
        assert _cluster_toml(la) == "cluster = [2, 1, 1]\n"
        assert _cluster_note(la) == (
            " cluster [2, 1, 1] (the kernel requires [1, 1, 1]; the launch attribute is taken)"
        )
        assert _cluster_note(launch((1, 1, 1), cluster_attr=1)) == " cluster [1, 1, 1]"
        assert _cluster_note(launch((2, 1, 1))) == " cluster [2, 1, 1]"
