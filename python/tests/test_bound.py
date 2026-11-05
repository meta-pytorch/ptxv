# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess

import numpy as np
import ptxv
import pytest
from conftest import FIX, needs_binary
from ptxv import _binary


def equiv(toml, tmp_path):
    b = tmp_path / "bound.json"
    r = subprocess.run(
        [
            str(_binary.find(build=False)),
            "equiv",
            str(toml),
            "--json",
            "--no-hw",
            "--emit-bound",
            str(b),
            "--out",
            str(tmp_path / "eq"),
        ],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=600,
    )
    doc = json.loads(r.stdout.strip().splitlines()[-1])
    return doc, json.loads(b.read_text())


@needs_binary
def test_sum_bound_matches_ptxv(tmp_path):
    doc, bd = equiv(FIX / "c_equiv_cumsum" / "equiv.toml", tmp_path)
    assert doc["command"] == "equiv" and doc["schema"] == 1 and doc["ok"]
    b = ptxv.Bound(bd)
    assert b.level == "L1" and b.used_inputs == ["x"] and b.output_shape == (4, 128)
    x = ptxv.Case(FIX / "c_cumsum").inputs()["x"]
    t = b(x=x)
    ref = np.array(doc["candidates"][0]["captured"]["bound_abs"])
    assert np.array_equal(
        t.ravel(), ref
    )  # the same float64 evaluation as ptxv's bound_at
    tb = b({"x": np.stack([x, -2 * x])})  # batch; tau is homogeneous in |x|
    assert tb.shape == (2, 4, 128) and np.allclose(tb[1], 2 * t)
    torch = pytest.importorskip("torch")
    tt = b(x=torch.from_numpy(x))
    assert isinstance(tt, torch.Tensor) and np.array_equal(tt.numpy(), t)


@needs_binary
def test_softmax_bound_matches_ptxv(tmp_path):
    doc, bd = equiv(FIX / "c_equiv_softmax" / "equiv.toml", tmp_path)
    b = ptxv.Bound(bd[1])  # the online-softmax candidate shares the reference's inputs
    x = ptxv.Case(FIX / "c_triton_softmax_online").inputs()["x"]
    ref = np.array(doc["candidates"][1]["captured"]["bound_abs"])
    assert np.allclose(b(x=x).ravel(), ref, rtol=1e-14, atol=0)


def synthetic(forms):
    """A two-output document: out 0 = 2 |x0 x1| + 0.5; out 1 = |x2| + 3 (`forms[1]`)."""
    return {
        "format": "ptxv-bound",
        "version": 1,
        "u": 2.0**-24,
        "level": "L1",
        "proven": True,
        "output": {"name": "o", "count": 2, "buffer": {"dtype": "f32", "shape": [2]}},
        "inputs": [{"name": "x", "buffer": {"dtype": "f32", "shape": [3]}}],
        "per_output": {
            "level": ["L1", "L1"],
            "form": forms,
            "bound_u": [1.0, 1.0],
            "abs_extra": [0.5, 0.0],
            "round_rel": [None, 0.25 if forms[1] == "round" else None],
            "no_bound": [None, None],
        },
        "terms": {
            "out": [0, 1, 1],
            "g": [2.0, 1.0, 3.0],
            "coef": [1.0, 1.0, -1.0],
            "deg": [2, 1, 0],
            "leaf_buf": [0, 0, 0],
            "leaf_idx": [0, 1, 2],
        },
        "softmax": {
            "out": [],
            "row": [],
            "leaf_buf": [],
            "leaf_idx": [],
            "rel_u": [],
            "add_u": [],
            "rows": [],
        },
    }


def test_sum_and_round_forms():
    x = np.array([2.0, -3.0, 5.0])
    t = ptxv.Bound(synthetic(["sum", "sum"]))(x=x)
    # out 1: |x2| * 1 + (empty product) * 3
    assert np.allclose(t, [2 * 6 + 0.5, 5.0 + 3.0], rtol=1e-11)
    t = ptxv.Bound(synthetic(["sum", "round"]))(x=x)
    y = 5.0 * 1.0 + 1.0 * -1.0  # sum_m c_m prod x_m
    assert np.allclose(t[1], 0.25 * abs(y) + 5.0 + 3.0, rtol=1e-11)
    b = ptxv.Bound(synthetic(["sum", "none"]))
    assert np.isnan(b(x=x)[1]) and b.evaluable.tolist() == [True, False]
