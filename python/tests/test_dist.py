# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from conftest import FIX, needs_binary
from ptxv import _errdist_adapter as A, dist as D


def bf16_ref(x):
    """bf16 RNE through the bit pattern (f32 input)."""
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32).astype(np.float64)


def test_round_bf16_matches_bits():
    x = (
        np.random.default_rng(0).normal(size=10000)
        * 10.0 ** np.random.default_rng(1).integers(-30, 30, 10000)
    ).astype(np.float32)
    assert np.array_equal(D.round_to(x.astype(np.float64), "bf16"), bf16_ref(x))


def test_e4m3_grid_and_saturation():
    v = D.round_to(
        np.array([0.0, 1.0625, 1.07, 447.0, 470.0, 1000.0, -(2**-9), 2**-10]), "e4m3"
    )
    assert v.tolist() == [0.0, 1.0, 1.125, 448.0, 448.0, 448.0, -(2**-9), 0.0]


def test_e2m1_values():
    x = np.linspace(-7, 7, 141)
    q = D.round_to(x, "e2m1")
    assert set(np.abs(q).tolist()) <= {0, 0.5, 1, 1.5, 2, 3, 4, 6}


def test_mx_floor_scale_and_ceil():
    x = np.zeros((1, 32))
    x[0, 0] = 300.0  # floor(log2 300) = 8, emax(e4m3) = 8: scale 1
    x[0, 1] = 1.3
    q = D.quantize(x, "mxf8", block=32)
    assert q[0, 0] == 288.0 and q[0, 1] == 1.25
    qc = D.quantize(
        x, "mxf8", block=32, scale_rule="ceil"
    )  # scale 2^ceil(log2(300/448)) = 1
    assert qc[0, 0] == 288.0


def test_nvfp4_shape_and_grid():
    x = np.random.default_rng(0).normal(size=(4, 64))
    q = D.quantize(x, "nvfp4")
    assert q.shape == x.shape
    assert np.abs(q - x).max() < np.abs(x).max() / 2


def test_samplers_and_spec_roundtrip():
    rng = np.random.default_rng(0)
    ds = [
        D.Normal(1, 2),
        D.Uniform(-1, 3),
        D.Laplace(0, 0.5),
        D.StudentT(3),
        D.LogNormal(0, 0.5),
        D.Discrete([1, 2, 3], [0.2, 0.3, 0.5]),
        D.Integers(0, 10),
        D.Const(2.5),
        D.Empirical(np.arange(10.0)),
        D.Mixture([D.Normal(0, 1), D.Laplace(0, 10)], [0.9, 0.1]),
        D.Normal(0, 1) * 3 + 1,
        D.Normal().per_channel(D.LogNormal(0, 1), axis=-1),
        D.Normal().outliers(0.1, 20.0, axis=0),
        D.Normal().quantize("mxf8", scales="SFA"),
        D.Normal().clip(-1, 1),
    ]
    for d in ds:
        s = d.sample((8, 64), rng)
        assert s.shape == (8, 64) and np.isfinite(s).all()
        assert D.from_spec(d.spec()).spec() == d.spec()
    assert abs(D.Normal(1, 2).sample((200000,), rng).mean() - 1) < 0.02
    assert set(np.unique(D.Integers(0, 3).sample((1000,), rng))) == {0, 1, 2}


def test_per_channel_scales_are_per_channel():
    d = D.Const(1.0).per_channel(D.LogNormal(0, 1), axis=-1)
    s = d.sample((5, 7), 0)
    assert np.allclose(s, s[0:1, :])  # same scale down each column


def test_adapter_maps_to_t_spec(tmp_path):
    args = {
        "inputs": {
            "A": D.Normal(0, 1).quantize("mxf8", scales="SFA").spec(),
            "B": D.Laplace(0, 0.5)
            .per_channel(D.LogNormal(0, 0.5), axis=0)
            .outliers(0.01, 20.0, axis=1)
            .quantize("mxf8", scales="SFB")
            .spec(),
            "c": D.Integers(0, 4).spec(),
            "e": D.Empirical(np.ones(5)).spec(),
        },
        "seed": 7,
        "mc_inputs": 100,
    }
    s = A.t_spec(args, tmp_path)
    assert s["version"] == 1 and s["seed"] == 7 and s["engine"] == {"mc_inputs": 100}
    assert s["tensors"]["A"] == {
        "quant": {
            "kind": "mx",
            "elem": "e4m3",
            "block": 32,
            "scale_rule": "floor",
            "scales": "SFA",
        },
        "dist": {"kind": "normal", "mean": 0.0, "std": 1.0},
    }
    b = s["tensors"]["B"]
    assert b["outliers"] == {"axis": 1, "prob": 0.01, "factor": 20.0}
    assert b["scales"] == [
        {"axis": 0, "dist": {"kind": "log_normal", "mu": 0.0, "sigma": 0.5}}
    ]
    assert b["dist"] == {"kind": "laplace", "mean": 0.0, "scale": 0.5}
    assert s["tensors"]["c"]["dist"] == {"kind": "integers", "lo": 0, "hi": 3}
    f = tmp_path / s["tensors"]["e"]["dist"]["file"]
    assert np.fromfile(f, np.float32).tolist() == [1.0] * 5
    with pytest.raises(A.AdapterError):
        A.t_spec({"inputs": {"x": D.Normal().quantize("fp8").spec()}}, tmp_path)
    with pytest.raises(A.AdapterError):
        A.t_spec({"inputs": {"x": D.Normal().clip(0, 1).spec()}}, tmp_path)


def test_report_to_errdist():
    rep = {
        "version": 1,
        "kind": "ptxv-errdist",
        "mode": "pair",
        "quantity": "C - R",
        "best": 0,
        "estimates": [
            {
                "tier": 3,
                "method": "monte carlo",
                "n": 1000,
                "outputs": 4,
                "abs": {
                    "mean": 0.0,
                    "var": 1e-12,
                    "std": 1e-6,
                    "mean_abs": 8e-7,
                    "p_zero": 0.3,
                    "quantiles": [[0.0, -3e-6], [0.5, 0.0], [1.0, 3e-6]],
                    "max_abs": 3e-6,
                },
                "error_bar": {"kind": "dkw", "notes": []},
            }
        ],
        "out_of_domain": {"mass": 0.0, "by_cause": {}, "method": "mc"},
    }
    e = A.to_errdist(rep)
    assert e.summary_only and e.n == 1000 and e.max == 3e-6
    assert e.quantile(0.5) == 0.0 and abs(e.cdf(1.5e-6) - 0.75) < 1e-12
    assert e.sample(100, 0).shape == (100,)
    assert "ErrDist" in repr(e)


def test_model_argv(tmp_path):
    """Compile / query command lines of the compile-once, query-many split."""
    c = A.compile_argv("fixtures/r4_sum_torch", tmp_path / "m.json")
    assert c == [
        "errdist",
        "fixtures/r4_sum_torch",
        "--compile",
        str(tmp_path / "m.json"),
    ]
    q = A.query_argv(
        tmp_path / "m.json",
        {
            "default": D.Normal(0, 1).spec(),
            "inputs": {},
            "tier1": False,
            "mc_inputs": 100,
        },
        tmp_path / "q",
    )
    assert q[:3] == ["errdist", "--model", str(tmp_path / "m.json")]
    assert "--no-tier1" in q and q[q.index("--mc") + 1] == "100"
    assert (tmp_path / "q" / "spec.json").exists()


@needs_binary
def test_errdist_model_chain_and_nonchain():
    """One compiled model per case answers queries for chain (a reduction) and non-chain
    (softmax) programs, memoized per distribution."""
    import ptxv

    m = ptxv.errdist_model(str(FIX / "r4_sum_torch"))
    e = m(D.Normal(0, 1), tier1=False)
    assert 0.55 < e.meta["p_differ"][0] < 0.8  # tier 3: 0.674
    wc = e.worst_case  # the kernel's own bound, stored in the compiled model
    assert wc["available"] and wc["kind"] == "single" and wc["form"] == "sum"
    assert m(D.Normal(0, 1), tier1=False) is not None
    s = ptxv.errdist_model(str(FIX / "r4_softmax_torch"))
    e = s(D.Normal(0, 1), mc_inputs=500)
    assert 0.5 < e.meta["p_differ"][0] < 0.8  # tier 3: 0.643
    assert "nonlin" in __import__("json").loads(s.path.read_text())
