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
