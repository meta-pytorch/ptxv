# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The proven equivalence bound tau(x) as a vectorized function of the inputs.

`ptxv equiv --emit-bound` / `ptxv compare --emit-bound` write the per-output bound as data
("ptxv-bound", docs/json.md). `Bound` evaluates it on numpy arrays or torch tensors:

    tau = eq.bound                      # from ptxv.compare(...)
    t = tau(x=x)                        # per-output bound, shaped like the output
    t = tau({"x": xb})                  # a leading batch dimension is allowed
    (abs(c - r) <= t).all()

Forms (per output): "sum" sum_m |prod x_m| g_m + abs_extra; "round" round_rel |y| + the sum
form, y = sum_m c_m prod x_m (the compact 16-bit form); "softmax" ((rel_u y + add_u) u +
abs_extra)(1 + 1e-9), y the exact softmax of the output's row; "none" no bound (NaN);
"opaque" a bound exists but is not exported (norm, attention: NaN, see `bound_u`).
The evaluation is float64 with the same relative slack ptxv's own `bound_at` applies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_SUM_SLACK = 1.0 + 1e-12
_SM_SLACK = 1.0 + 1e-9

_FLOAT = {"f32", "f64", "bf16", "f16"}


def _to_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):  # torch
        t = t.detach()
        if str(t.dtype) in ("torch.bfloat16", "torch.float16"):
            t = t.float()
        return t.cpu().numpy().astype(np.float64)
    return np.asarray(t, dtype=np.float64)


class Bound:
    """Per-output equivalence bound |C(x) - R(x)| <= tau(x), as data plus an evaluator."""

    def __init__(self, doc: Mapping[str, Any]) -> None:
        if doc.get("format") != "ptxv-bound":
            raise ValueError("not a ptxv-bound document")
        self.doc = dict(doc)
        self.u: float = doc["u"]
        self.level: str = doc.get("level", "?")
        self.proven: bool = bool(doc.get("proven"))
        out = doc["output"]
        self.n: int = out["count"]
        self.output_name: str | None = out.get("name")
        ob = out.get("buffer") or {}
        self.output_shape: tuple[int, ...] = tuple(ob.get("shape") or (self.n,))
        if int(np.prod(self.output_shape)) != self.n:
            self.output_shape = (self.n,)
        self.inputs: list[dict] = doc["inputs"]
        self.input_names: list[str] = [i["name"] for i in self.inputs]
        po = doc["per_output"]
        self.form = np.array(po["form"])
        self.bound_u = np.array(
            [np.nan if b is None else b for b in po["bound_u"]], float
        )
        self.abs_extra = np.array(po["abs_extra"], float)
        self.round_rel = np.array(
            [0.0 if r is None else r for r in po["round_rel"]], float
        )
        t = doc["terms"]
        self.t_out = np.asarray(t["out"], np.int64)
        self.t_g = np.asarray(t["g"], float)
        self.t_c = np.asarray(t["coef"], float)
        self.t_deg = np.asarray(t["deg"], np.int64)
        self.l_buf = np.asarray(t["leaf_buf"], np.int64)
        self.l_idx = np.asarray(t["leaf_idx"], np.int64)
        s = doc["softmax"]
        self.s_out = np.asarray(s["out"], np.int64)
        self.s_row = np.asarray(s["row"], np.int64)
        self.s_buf = np.asarray(s["leaf_buf"], np.int64)
        self.s_idx = np.asarray(s["leaf_idx"], np.int64)
        self.s_rel = np.asarray(s["rel_u"], float)
        self.s_add = np.asarray(s["add_u"], float)
        self.rows = [
            (np.asarray(r["leaf_buf"], np.int64), np.asarray(r["leaf_idx"], np.int64))
            for r in s["rows"]
        ]

    @classmethod
    def load(cls, path: str | Path) -> "Bound":
        return cls(json.loads(Path(path).read_text()))

    # -------------------------------------------------------------------------------
    @property
    def used_inputs(self) -> list[str]:
        """Names of the input buffers the bound depends on."""
        b = set(self.l_buf.tolist()) | set(self.s_buf.tolist())
        for rb, _ in self.rows:
            b |= set(rb.tolist())
        return [self.input_names[i] for i in sorted(b)]

    @property
    def evaluable(self) -> np.ndarray:
        """Per output: whether tau(x) can be evaluated from the exported data."""
        return np.isin(self.form, ["sum", "round", "softmax"])

    def max_u(self) -> float:
        """The largest per-output coefficient bound_u (units of u)."""
        if not self.n or np.all(np.isnan(self.bound_u)):
            return float("nan")
        return float(np.nanmax(self.bound_u))

    def _flat(
        self, inputs: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, int | None]:
        """All used inputs concatenated: (values [B, total] or [total], offsets, batch)."""
        offs = np.zeros(len(self.inputs) + 1, np.int64)
        arrays: list[np.ndarray | None] = []
        batch = None
        for i, inp in enumerate(self.inputs):
            name = inp["name"]
            buf = inp.get("buffer") or {}
            n = int(np.prod(buf.get("shape") or [0]))
            a = None
            if name in self.used_inputs:
                if name not in inputs:
                    raise KeyError(
                        f"bound needs input {name!r} ({buf.get('dtype')} {buf.get('shape')})"
                    )
                if buf.get("dtype") not in _FLOAT:
                    raise TypeError(
                        f"input {name!r} is {buf.get('dtype')}: leaves of byte formats "
                        "(FP8/FP4 codes) are not evaluable here"
                    )
                a = _to_numpy(inputs[name])
                if a.size == n:
                    a = a.reshape(n)
                elif a.ndim >= 1 and a.size % n == 0 and a.shape[0] * n == a.size:
                    b = a.shape[0]
                    if batch not in (None, b):
                        raise ValueError("inputs disagree on the batch size")
                    batch = b
                    a = a.reshape(b, n)
                else:
                    raise ValueError(
                        f"input {name!r}: {a.shape} does not match {buf.get('shape')}"
                    )
            arrays.append(a)
            offs[i + 1] = offs[i] + n
        total = int(offs[-1])
        X = np.zeros((batch or 1, total))
        for i, a in enumerate(arrays):
            if a is not None:
                X[:, offs[i] : offs[i + 1]] = a if a.ndim == 2 else a[None, :]
        return X, offs, batch

    def __call__(self, inputs: Mapping[str, Any] | None = None, /, **kw: Any) -> Any:
        """tau at the given inputs (by buffer name); returns the type of the inputs (numpy, or
        torch on the first torch input's device), shaped like the output (batch first)."""
        inputs = dict(inputs or {}, **kw)
        like = next((v for v in inputs.values() if hasattr(v, "detach")), None)
        r = self.eval_numpy(inputs)
        if like is not None:
            import torch

            return torch.from_numpy(r).to(like.device)
        return r

    def eval_numpy(self, inputs: Mapping[str, Any]) -> np.ndarray:
        X, offs, batch = self._flat(inputs)
        B = X.shape[0]
        out = np.full((B, self.n), np.nan)
        # Sum and round forms.
        if len(self.t_out):
            g = offs[self.l_buf] + self.l_idx
            vals = X[:, g]  # [B, leaves]
            nt = len(self.t_deg)
            starts = np.concatenate([[0], np.cumsum(self.t_deg)[:-1]])
            prod = np.ones((B, nt))
            nz = self.t_deg > 0
            if vals.shape[1]:
                p = np.multiply.reduceat(vals, starts[nz], axis=1) if nz.any() else None
                if p is not None:
                    prod[:, nz] = p
            mag = np.abs(prod) * self.t_g
            tb = np.zeros((B, self.n))
            np.add.at(tb, (slice(None), self.t_out), mag)
            y = np.zeros((B, self.n))
            np.add.at(y, (slice(None), self.t_out), prod * self.t_c)
            for f in ("sum", "round"):
                m = self.form == f
                if not m.any():
                    continue
                b = tb[:, m] + self.abs_extra[m]
                if f == "round":
                    b = b + self.round_rel[m] * np.abs(y[:, m])
                out[:, m] = b * _SUM_SLACK
        # Forms without terms (an L0 output with no term): abs_extra.
        m0 = np.isin(self.form, ["sum", "round"]) & ~np.isin(
            np.arange(self.n), self.t_out
        )
        out[:, m0] = self.abs_extra[m0] * _SUM_SLACK
        # Softmax form.
        if len(self.s_out):
            ys = np.empty((B, len(self.s_out)))
            cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for j, (_o, r) in enumerate(zip(self.s_out, self.s_row)):
                if r not in cache:
                    rb, ri = self.rows[r]
                    v = X[:, offs[rb] + ri]
                    mx = v.max(axis=1, keepdims=True)
                    cache[r] = (mx[:, 0], np.exp(v - mx).sum(axis=1))
                mx, s = cache[r]
                xi = X[:, offs[self.s_buf[j]] + self.s_idx[j]]
                ys[:, j] = np.exp(xi - mx) / s
            b = (
                (self.s_rel * ys + self.s_add) * self.u + self.abs_extra[self.s_out]
            ) * _SM_SLACK
            out[:, self.s_out] = b + 1e-300
        shape = self.output_shape
        return out.reshape((B,) + shape) if batch is not None else out.reshape(shape)

    def check(self, ref: Any, cand: Any, inputs: Mapping[str, Any]) -> dict:
        """|cand - ref| against tau(inputs): fraction of the bound used, violations."""
        t = np.asarray(self.eval_numpy(inputs))
        d = np.abs(_to_numpy(cand) - _to_numpy(ref)).reshape(t.shape)
        ok = np.isfinite(t)
        to, do = t[ok], d[ok]
        frac = np.zeros_like(do)
        pos = to > 0
        frac[pos] = do[pos] / to[pos]
        frac[~pos & (do > 0)] = np.inf
        return {
            "outputs": int(t.size),
            "evaluated": int(ok.sum()),
            "violations": int((d[ok] > t[ok]).sum()),
            "max_fraction": float(frac.max()) if frac.size else None,
        }

    def distribution(
        self, inputs: Mapping[str, Any], n: int = 1000, seed: int = 0, batch: int = 256
    ):
        """Distribution of tau(x) for x drawn from `inputs` (name -> `ptxv.dist` distribution
        or fixed array): an `ErrDist` over (sample, output)."""
        from .dist import Dist
        from .results import ErrDist

        if not self.n or not np.any(self.evaluable):
            raise ValueError(
                "no output has an evaluable bound (the equivalence is not proven, or its"
                " form is not exported); see Equivalence.blockers"
            )
        rng = np.random.default_rng(seed)
        shapes = {
            i["name"]: tuple((i.get("buffer") or {}).get("shape") or ())
            for i in self.inputs
        }
        chunks = []
        done = 0
        while done < n:
            b = min(batch, n - done)
            xs = {}
            for name in self.used_inputs:
                d = inputs.get(name)
                if isinstance(d, Dist):
                    xs[name] = np.stack([d.sample(shapes[name], rng) for _ in range(b)])
                elif d is None:
                    raise KeyError(f"no distribution for input {name!r}")
                else:
                    xs[name] = np.broadcast_to(_to_numpy(d), (b,) + shapes[name])
            chunks.append(self.eval_numpy(xs).reshape(b, -1))
            done += b
        return ErrDist(
            np.concatenate(chunks),
            metric="bound",
            source="bound",
            output_shape=self.output_shape,
        )

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(
            {"form": self.form, "bound_u": self.bound_u, "abs_extra": self.abs_extra}
        )

    @property
    def json(self) -> dict:
        return self.doc

    def __repr__(self) -> str:
        forms = {f: int((self.form == f).sum()) for f in np.unique(self.form)}
        return (
            f"Bound(level={self.level!r}, outputs={self.n}, inputs={self.used_inputs}, "
            f"forms={forms}, max={self.max_u():.3g}u{'' if self.proven else ', unproven'})"
        )
