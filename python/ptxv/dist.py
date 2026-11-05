# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Input distributions for error-distribution jobs (`ptxv.errdist`, `Kernel.errdist`).

    from ptxv import dist as D
    D.Normal(0, 1)
    D.StudentT(3) * 0.5 + 1
    D.Mixture([D.Normal(0, 1), D.Laplace(0, 10)], [0.99, 0.01])
    D.Normal(0, 1).per_channel(D.LogNormal(0, 1), axis=-1)   # one scale per channel
    D.Normal(0, 1).quantize("mxf8", block=32)                # MX: e4m3 + ue8m0 per 32
    D.Empirical(activations)                                 # resample captured values

Every distribution samples with numpy (`sample(shape, rng)`, float64, the dequantized values
a kernel reads) and serializes to a plain dict (`spec()`), which `_errdist_adapter` maps to
ptxv errdist's input format.
"""

from __future__ import annotations

import base64
import math
from typing import Any, Sequence

import numpy as np

# Element formats: (mantissa bits, minimum normal exponent, largest finite value).
FORMATS: dict[str, tuple[int, int, float]] = {
    "f32": (23, -126, float(np.finfo(np.float32).max)),
    "bf16": (7, -126, float.fromhex("0x1.fep127")),
    "f16": (10, -14, 65504.0),
    "e4m3": (3, -6, 448.0),
    "e5m2": (2, -14, 57344.0),
    "e3m2": (2, -2, 28.0),
    "e2m3": (3, 0, 7.5),
    "e2m1": (1, 0, 6.0),
}

# Quantization recipes: element format, block size (None: per tensor), scale kind.
RECIPES: dict[str, dict[str, Any]] = {
    "bf16": {"elem": "bf16", "block": None, "scale": None},
    "f16": {"elem": "f16", "block": None, "scale": None},
    "fp8": {"elem": "e4m3", "block": None, "scale": "f32"},
    "fp8_e4m3": {"elem": "e4m3", "block": None, "scale": "f32"},
    "fp8_e5m2": {"elem": "e5m2", "block": None, "scale": "f32"},
    "e4m3": {"elem": "e4m3", "block": None, "scale": "none"},
    "e5m2": {"elem": "e5m2", "block": None, "scale": "none"},
    "e2m1": {"elem": "e2m1", "block": None, "scale": "none"},
    "mxf8": {"elem": "e4m3", "block": 32, "scale": "ue8m0"},
    "mxf8_e5m2": {"elem": "e5m2", "block": 32, "scale": "ue8m0"},
    "mxf6": {"elem": "e2m3", "block": 32, "scale": "ue8m0"},
    "mxfp4": {"elem": "e2m1", "block": 32, "scale": "ue8m0"},
    "nvfp4": {"elem": "e2m1", "block": 16, "scale": "e4m3+f32"},
}


def round_to(x: np.ndarray, fmt: str, saturate: bool = True) -> np.ndarray:
    """Round to nearest even in an element format (subnormals kept; saturating by default)."""
    mbits, emin, vmax = FORMATS[fmt]
    x = np.asarray(x, np.float64)
    a = np.abs(x)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    e = np.maximum(e, emin)
    q = np.exp2(e - mbits)
    r = np.rint(x / q) * q
    if saturate:
        r = np.clip(r, -vmax, vmax)
    else:
        r = np.where(np.abs(r) > vmax, np.sign(r) * np.inf, r)
    return np.where(np.isfinite(x), r, x)


def quantize(
    x: np.ndarray,
    recipe: str,
    block: int | None = None,
    axis: int = -1,
    scale_rule: str = "floor",
    global_scale: bool = True,
) -> np.ndarray:
    """Quantize-dequantize `x` with a recipe (see RECIPES); returns float64 values.

    MX scales: "floor" 2^(floor(log2 amax) - emax_elem) (OCP MX v1.0) or "ceil"
    2^ceil(log2(amax / max_elem)). NVFP4: e4m3 block scales times a per-tensor f32 scale
    amax / (448 * 6) when `global_scale`."""
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r} (one of {sorted(RECIPES)})")
    r = RECIPES[recipe]
    elem, scale = r["elem"], r["scale"]
    block = block or r["block"]
    x = np.asarray(x, np.float64)
    if scale is None:
        return round_to(x, elem)
    if scale == "none":  # cast: no scale, saturating
        return round_to(x, elem)
    vmax = FORMATS[elem][2]
    if block is None:  # per-tensor scale (f32)
        amax = np.max(np.abs(x)) if x.size else 0.0
        s = amax / vmax if amax > 0 else 1.0
        return round_to(x / s, elem) * s
    xm = np.moveaxis(x, axis, -1)
    n = xm.shape[-1]
    pad = (-n) % block
    xp = np.pad(xm, [(0, 0)] * (xm.ndim - 1) + [(0, pad)])
    blocks = xp.reshape(xp.shape[:-1] + (-1, block))
    amax = np.max(np.abs(blocks), axis=-1, keepdims=True)
    if scale == "ue8m0":  # OCP MX: 2^(floor(log2 amax) - emax of the element format)
        emax = math.floor(math.log2(vmax))
        with np.errstate(divide="ignore"):
            if scale_rule == "ceil":
                k = np.ceil(np.log2(np.where(amax > 0, amax, vmax) / vmax))
            else:
                k = np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - emax
        s = np.exp2(np.clip(k, -127, 127))
        q = round_to(blocks / s, elem) * s
    elif scale == "e4m3+f32":  # NVFP4: e4m3 block scale times a per-tensor f32 scale
        gamax = float(np.max(np.abs(x))) if x.size else 0.0
        g = gamax / (448.0 * vmax) if gamax > 0 and global_scale else 1.0
        sb = round_to(amax / vmax / g, "e4m3")
        s = np.where(sb > 0, sb * g, 1.0)
        q = round_to(blocks / s, elem) * s
    else:  # pragma: no cover
        raise ValueError(scale)
    q = q.reshape(xp.shape)[..., :n]
    return np.moveaxis(q, -1, axis)


class Dist:
    """A distribution over tensors (elementwise iid unless a subclass says otherwise)."""

    def sample(
        self, shape: Sequence[int], rng: np.random.Generator | int | None = None
    ) -> np.ndarray:
        raise NotImplementedError

    def spec(self) -> dict:
        raise NotImplementedError

    # -- combinators -----------------------------------------------------------------
    def __mul__(self, k: float) -> "Dist":
        return Affine(self, scale=float(k))

    __rmul__ = __mul__

    def __add__(self, k: float) -> "Dist":
        return Affine(self, shift=float(k))

    __radd__ = __add__

    def __neg__(self) -> "Dist":
        return Affine(self, scale=-1.0)

    def quantize(
        self,
        recipe: str,
        block: int | None = None,
        axis: int = -1,
        *,
        scales: str | None = None,
        scale_rule: str = "floor",
        global_scale: bool = True,
    ) -> "Dist":
        """The values after quantize-dequantize with `recipe`: bf16, f16; e4m3 / e5m2 / e2m1
        (cast, saturating); fp8 / fp8_e4m3 / fp8_e5m2 (per-tensor f32 scale); mxf8, mxf8_e5m2,
        mxf6, mxfp4, nvfp4 (blocks along `axis`). `scales` names the case's scale buffer the
        recipe fills (block-scaled recipes, for ptxv errdist)."""
        return Quantized(
            self,
            recipe,
            block,
            axis,
            scales=scales,
            scale_rule=scale_rule,
            global_scale=global_scale,
        )

    def outliers(self, prob: float, factor: float, axis: int = -1) -> "Dist":
        """Each index along `axis` is an outlier channel with probability `prob`: its
        elements are multiplied by `factor`."""
        return Outliers(self, prob, factor, axis)

    def per_channel(self, scale: "Dist", axis: int = -1) -> "Dist":
        """Hierarchical: one scale drawn from `scale` per index along `axis`, times this."""
        return Scaled(self, scale, axis)

    def clip(self, lo: float, hi: float) -> "Dist":
        return Clipped(self, lo, hi)

    def __repr__(self) -> str:
        s = self.spec()
        k = s.pop("kind")
        return f"{k}({', '.join(f'{a}={v!r}' for a, v in s.items())})"


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


class Normal(Dist):
    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu, self.sigma = float(mu), float(sigma)

    def sample(self, shape, rng=None):
        return _rng(rng).normal(self.mu, self.sigma, tuple(shape))

    def spec(self):
        return {"kind": "normal", "mu": self.mu, "sigma": self.sigma}


class Const(Dist):
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def sample(self, shape, rng=None):
        return np.full(tuple(shape), self.value)

    def spec(self):
        return {"kind": "const", "value": self.value}


class Uniform(Dist):
    def __init__(self, lo: float = 0.0, hi: float = 1.0) -> None:
        self.lo, self.hi = float(lo), float(hi)

    def sample(self, shape, rng=None):
        return _rng(rng).uniform(self.lo, self.hi, tuple(shape))

    def spec(self):
        return {"kind": "uniform", "lo": self.lo, "hi": self.hi}


class Laplace(Dist):
    def __init__(self, mu: float = 0.0, b: float = 1.0) -> None:
        self.mu, self.b = float(mu), float(b)

    def sample(self, shape, rng=None):
        return _rng(rng).laplace(self.mu, self.b, tuple(shape))

    def spec(self):
        return {"kind": "laplace", "mu": self.mu, "b": self.b}


class StudentT(Dist):
    def __init__(self, df: float, mu: float = 0.0, scale: float = 1.0) -> None:
        self.df, self.mu, self.scale = float(df), float(mu), float(scale)

    def sample(self, shape, rng=None):
        return self.mu + self.scale * _rng(rng).standard_t(self.df, tuple(shape))

    def spec(self):
        return {"kind": "student_t", "df": self.df, "mu": self.mu, "scale": self.scale}


class LogNormal(Dist):
    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu, self.sigma = float(mu), float(sigma)

    def sample(self, shape, rng=None):
        return _rng(rng).lognormal(self.mu, self.sigma, tuple(shape))

    def spec(self):
        return {"kind": "lognormal", "mu": self.mu, "sigma": self.sigma}


class Discrete(Dist):
    """Values with probabilities (uniform when `probs` is None)."""

    def __init__(
        self, values: Sequence[float], probs: Sequence[float] | None = None
    ) -> None:
        self.values = [float(v) for v in values]
        p = np.ones(len(self.values)) if probs is None else np.asarray(probs, float)
        self.probs = (p / p.sum()).tolist()

    def sample(self, shape, rng=None):
        return _rng(rng).choice(
            np.asarray(self.values), size=tuple(shape), p=self.probs
        )

    def spec(self):
        return {"kind": "discrete", "values": self.values, "probs": self.probs}


class Integers(Dist):
    """Uniform integers in [lo, hi)."""

    def __init__(self, lo: int, hi: int) -> None:
        self.lo, self.hi = int(lo), int(hi)

    def sample(self, shape, rng=None):
        return _rng(rng).integers(self.lo, self.hi, tuple(shape)).astype(np.float64)

    def spec(self):
        return {"kind": "integers", "lo": self.lo, "hi": self.hi}


class Empirical(Dist):
    """Values resampled iid from a tensor (or the tensor itself when `exact` and the shape
    matches)."""

    def __init__(self, values: Any, exact: bool = False) -> None:
        if hasattr(values, "detach"):
            values = values.detach().float().cpu().numpy()
        self.values = np.asarray(values, np.float64)
        self.exact = exact

    def sample(self, shape, rng=None):
        if self.exact and tuple(shape) == self.values.shape:
            return self.values.copy()
        return _rng(rng).choice(self.values.ravel(), size=tuple(shape))

    def spec(self):
        v = self.values.astype(np.float32)
        return {
            "kind": "empirical",
            "shape": list(v.shape),
            "exact": self.exact,
            "f32_base64": base64.b64encode(v.tobytes()).decode(),
        }


class Mixture(Dist):
    """Elementwise mixture: each element from component i with probability weights[i]."""

    def __init__(
        self, components: Sequence[Dist], weights: Sequence[float] | None = None
    ) -> None:
        self.components = list(components)
        w = (
            np.ones(len(self.components))
            if weights is None
            else np.asarray(weights, float)
        )
        self.weights = (w / w.sum()).tolist()

    def sample(self, shape, rng=None):
        g = _rng(rng)
        shape = tuple(shape)
        pick = g.choice(len(self.components), size=shape, p=self.weights)
        out = np.empty(shape)
        for i, c in enumerate(self.components):
            m = pick == i
            if m.any():
                out[m] = c.sample(shape, g)[m]
        return out

    def spec(self):
        return {
            "kind": "mixture",
            "weights": self.weights,
            "components": [c.spec() for c in self.components],
        }


class Affine(Dist):
    def __init__(self, base: Dist, scale: float = 1.0, shift: float = 0.0) -> None:
        if isinstance(base, Affine):
            scale, shift = base.scale * scale, base.shift * scale + shift
            base = base.base
        self.base, self.scale, self.shift = base, scale, shift

    def sample(self, shape, rng=None):
        return self.base.sample(shape, rng) * self.scale + self.shift

    def spec(self):
        return {
            "kind": "affine",
            "scale": self.scale,
            "shift": self.shift,
            "base": self.base.spec(),
        }


class Clipped(Dist):
    def __init__(self, base: Dist, lo: float, hi: float) -> None:
        self.base, self.lo, self.hi = base, float(lo), float(hi)

    def sample(self, shape, rng=None):
        return np.clip(self.base.sample(shape, rng), self.lo, self.hi)

    def spec(self):
        return {"kind": "clip", "lo": self.lo, "hi": self.hi, "base": self.base.spec()}


class Scaled(Dist):
    """base * scale[channel]: one draw of `scale` per index along `axis` (per-channel
    activation scales, outlier channels)."""

    def __init__(self, base: Dist, scale: Dist, axis: int = -1) -> None:
        self.base, self.scale, self.axis = base, scale, axis

    def sample(self, shape, rng=None):
        g = _rng(rng)
        shape = tuple(shape)
        x = self.base.sample(shape, g)
        n = shape[self.axis]
        s = self.scale.sample((n,), g)
        bshape = [1] * len(shape)
        bshape[self.axis] = n
        return x * s.reshape(bshape)

    def spec(self):
        return {
            "kind": "per_channel",
            "axis": self.axis,
            "base": self.base.spec(),
            "scale": self.scale.spec(),
        }


class Outliers(Dist):
    def __init__(self, base: Dist, prob: float, factor: float, axis: int = -1) -> None:
        self.base, self.prob, self.factor, self.axis = (
            base,
            float(prob),
            float(factor),
            axis,
        )

    def sample(self, shape, rng=None):
        g = _rng(rng)
        shape = tuple(shape)
        x = self.base.sample(shape, g)
        n = shape[self.axis]
        f = np.where(g.random(n) < self.prob, self.factor, 1.0)
        bshape = [1] * len(shape)
        bshape[self.axis] = n
        return x * f.reshape(bshape)

    def spec(self):
        return {
            "kind": "outliers",
            "axis": self.axis,
            "prob": self.prob,
            "factor": self.factor,
            "base": self.base.spec(),
        }


class Quantized(Dist):
    def __init__(
        self,
        base: Dist,
        recipe: str,
        block: int | None = None,
        axis: int = -1,
        *,
        scales: str | None = None,
        scale_rule: str = "floor",
        global_scale: bool = True,
    ) -> None:
        if recipe not in RECIPES:
            raise ValueError(f"unknown recipe {recipe!r} (one of {sorted(RECIPES)})")
        if scale_rule not in ("floor", "ceil"):
            raise ValueError("scale_rule: 'floor' or 'ceil'")
        self.base, self.recipe, self.axis = base, recipe, axis
        self.block = block or RECIPES[recipe]["block"]
        self.scales, self.scale_rule, self.global_scale = (
            scales,
            scale_rule,
            global_scale,
        )

    def sample(self, shape, rng=None):
        return quantize(
            self.base.sample(shape, rng),
            self.recipe,
            self.block,
            self.axis,
            self.scale_rule,
            self.global_scale,
        )

    def spec(self):
        r = RECIPES[self.recipe]
        return {
            "kind": "quantize",
            "recipe": self.recipe,
            "element": r["elem"],
            "scale": r["scale"],
            "block": self.block,
            "axis": self.axis,
            "scales": self.scales,
            "scale_rule": self.scale_rule,
            "global_scale": self.global_scale,
            "base": self.base.spec(),
        }


def from_spec(s: dict) -> Dist:  # noqa: C901 (one branch per serialized distribution kind)
    """Inverse of `Dist.spec()` (Empirical needs its values in the spec)."""
    k = s["kind"]
    if k == "normal":
        return Normal(s["mu"], s["sigma"])
    if k == "uniform":
        return Uniform(s["lo"], s["hi"])
    if k == "laplace":
        return Laplace(s["mu"], s["b"])
    if k == "student_t":
        return StudentT(s["df"], s["mu"], s["scale"])
    if k == "lognormal":
        return LogNormal(s["mu"], s["sigma"])
    if k == "discrete":
        return Discrete(s["values"], s["probs"])
    if k == "integers":
        return Integers(s["lo"], s["hi"])
    if k == "empirical":
        v = np.frombuffer(base64.b64decode(s["f32_base64"]), np.float32)
        return Empirical(v.reshape(s["shape"]), s.get("exact", False))
    if k == "mixture":
        return Mixture([from_spec(c) for c in s["components"]], s["weights"])
    if k == "affine":
        return Affine(from_spec(s["base"]), s["scale"], s["shift"])
    if k == "clip":
        return Clipped(from_spec(s["base"]), s["lo"], s["hi"])
    if k == "per_channel":
        return Scaled(from_spec(s["base"]), from_spec(s["scale"]), s["axis"])
    if k == "quantize":
        return Quantized(
            from_spec(s["base"]),
            s["recipe"],
            s["block"],
            s["axis"],
            scales=s.get("scales"),
            scale_rule=s.get("scale_rule", "floor"),
            global_scale=s.get("global_scale", True),
        )
    if k == "outliers":
        return Outliers(from_spec(s["base"]), s["prob"], s["factor"], s["axis"])
    if k == "const":
        return Const(s["value"])
    raise ValueError(f"unknown distribution kind {k!r}")
