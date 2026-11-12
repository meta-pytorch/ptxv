# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Result objects: `Verdict` (check), `Equivalence` (compare), `ErrDist` (error distributions).

Each keeps the CLI's JSON document (`.json`) and exposes the common fields as attributes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .bound import Bound


# ptxv's bitwise-equal sentence for one compared output (`expect_report`), whole line; and its
# within-one-ulp sentence (N-MULTIMEM-ROUND16), which ends "the rest bitwise equal".
_BITWISE_LINE = re.compile(
    r"^output \d+: [^:]+: bitwise equal to the captured GPU output "
    r"(\(\d+ bytes(; \d+ differing bytes in device-address words excluded \(expect_ignore\))?\)"
    r"|outside the permuted rows)$"
)
_WITHIN_ULP_LINE = re.compile(
    r"^output \d+: [^:]+: within one ulp of the captured GPU output: "
)


@dataclass
class Finding:
    """One finding. For races and stale reads, `gb300` is the two-level classification
    (`ptxv check`'s second level, the GB300 hardware model E-HW-L1): "observable" (reachable
    on the hardware model), "latent" (a PTX-model bug not reachable on current GB300, even with
    outside fills; see `gb300_reason`), "latent-closed-world" (not reachable only while nothing
    outside the case fills the line, E-L1-CLOSED: observable with a co-resident kernel that
    does; `gb300_witness` is the havoc check's `stale-l1-havoc:<pc>`) or "undecided"; None
    when not classified. A latent finding is still a bug."""

    kind: str
    msg: str
    trace: list[str] = field(default_factory=list)
    sig: str = ""
    gb300: str | None = None
    gb300_reason: str = ""
    gb300_witness: str | None = None

    @classmethod
    def from_json(cls, f: dict) -> "Finding":
        g = f.get("gb300") or {}
        return cls(
            f["kind"],
            f.get("msg", ""),
            f.get("trace", []),
            f.get("sig", ""),
            g.get("class"),
            g.get("reason", ""),
            g.get("witness"),
        )

    def __repr__(self) -> str:
        first = self.msg.strip().splitlines()[0] if self.msg.strip() else ""
        g = f" [{self.gb300} on GB300]" if self.gb300 else ""
        return f"Finding({self.kind}{g}: {first[:100]})"


@dataclass(repr=False)
class Verdict:
    """Execution verdict of one case (`ptxv check`): every schedule explored or certified.

    classes: ["clean"] or the finding kinds ("race", "deadlock", "violation", "fault",
    "unmodeled", "nondeterministic"). certified: a complete verdict decided by the E-WELLSYNC
    certificate (one run stands for every schedule); an incomplete verdict is never certified,
    whatever its assumptions list. complete: the exploration finished (no state cap hit).
    sim_vs_gpu: the simulator's output against the captured GPU output (the report lines);
    gpu_compare: ptxv's result behind each line (`{output, part, result}`), read by
    `gpu_state` / `sim_equals_gpu`.
    refuted_opt_in: refuted assumptions the verdict relies on by explicit opt-in.
    gb300: the second level's summary (the hardware-model check: states, observable / latent
    counts, stale reads only it sees), or None when it did not run. `observable` / `latent` /
    `latent_closed_world` / `latent_needs_divergence` list the classified findings.
    nonfinite: with `check(..., nonfinite=True)` (or "strict"), the NaN / Inf scan of the
    outputs: per output buffer counts and the first element, input buffers holding NaN / Inf,
    and `births`, the instructions that produced a non-finite value from operands without
    one. A `nonfinite-output` finding (and class) when every float input element is finite.
    """

    classes: list[str]
    certified: bool
    complete: bool
    states: int
    findings: list[Finding]
    assumptions: list[str]
    notes: list[str]
    seconds: float
    sim_vs_gpu: list[str]
    passed: bool
    json: dict
    refuted_opt_in: list[str] = field(default_factory=list)
    gb300: dict | None = None
    nonfinite: dict | None = None
    gpu_compare: list | None = None

    @classmethod
    def from_json(cls, doc: dict) -> "Verdict":
        rows = doc.get("rows") or []
        if not rows:
            raise ValueError("check produced no rows")
        r = rows[0]
        v = r["verdict"]
        got = list(r.get("got") or [])
        return cls(
            classes=got,
            certified="E-WELLSYNC" in v.get("assumptions", []) and v.get("complete") is True,
            complete=bool(v.get("complete")),
            states=int(v.get("states", 0)),
            findings=[Finding.from_json(f) for f in v.get("findings", [])],
            assumptions=list(v.get("assumptions", [])),
            notes=list(v.get("notes", [])),
            seconds=float(v.get("seconds", 0.0)),
            sim_vs_gpu=list(r.get("expect_report", [])),
            passed=bool(r.get("pass")),
            json=doc,
            refuted_opt_in=list(r.get("refuted_opt_in") or []),
            gb300=r.get("gb300"),
            nonfinite=r.get("nonfinite"),
            gpu_compare=r.get("gpu_compare"),
        )

    @property
    def observable(self) -> list[Finding]:
        """Findings reachable under the GB300 hardware model."""
        return [f for f in self.findings if f.gb300 == "observable"]

    @property
    def latent(self) -> list[Finding]:
        """PTX-model findings not reachable on current GB300 hardware, even with outside
        fills of L1 lines (still bugs)."""
        return [f for f in self.findings if f.gb300 == "latent"]

    @property
    def latent_closed_world(self) -> list[Finding]:
        """PTX-model findings latent on GB300 only under E-L1-CLOSED: observable when a
        co-resident kernel fills the line (witness `gb300_witness`). Not in `latent`."""
        return [f for f in self.findings if f.gb300 == "latent-closed-world"]

    @property
    def latent_needs_divergence(self) -> list[Finding]:
        """Intra-warp PTX-model races between lanes of one warp that stays converged between
        the accesses (PTX front-end): GB300 orders them unless the warp diverges or the
        compiler reorders them (E-LOCKSTEP). Not in `latent`."""
        return [f for f in self.findings if f.gb300 == "latent-needs-divergence"]

    @property
    def clean(self) -> bool:
        return self.classes == ["clean"] and self.complete

    def to_pandas(self):
        """One row per finding (kind, first line of the message, trace length)."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "kind": f.kind,
                    "gb300": f.gb300,
                    "msg": f.msg.strip().splitlines()[0] if f.msg.strip() else "",
                    "trace": len(f.trace),
                }
                for f in self.findings
            ],
            columns=["kind", "gb300", "msg", "trace"],
        )

    @property
    def gpu_state(self) -> str | None:
        """The comparison with the captured GPU output, as kernel_db reads it: "bitwise" (every
        comparison bitwise equal), else "mismatch", "within_ulp" (differing elements one unit in
        the last place off or NaN on both sides, N-MULTIMEM-ROUND16), "permuted_equal" /
        "not_compared" (C-TICKET-ADDR), in that order; None when nothing was compared. From ptxv's
        per-comparison `gpu_compare`, never the report text. Outputs of binaries without it: a
        line is bitwise only when the whole line is ptxv's bitwise-equal sentence (the
        within-one-ulp line also says "the rest bitwise equal")."""
        if self.gpu_compare is not None:
            res = [
                x.get("result")
                for x in self.gpu_compare
                if x.get("result") != "no_capture"
            ]
        else:
            res = [
                "bitwise"
                if _BITWISE_LINE.match(s)
                else "within_ulp"
                if _WITHIN_ULP_LINE.match(s)
                else "mismatch"
                for s in self.sim_vs_gpu
            ]
        if not res:
            return None
        if all(r == "bitwise" for r in res):
            return "bitwise"
        for state in ("mismatch", "within_ulp", "permuted_equal", "not_compared"):
            if state in res:
                return state
        return "mismatch"  # a result this reader does not know

    @property
    def sim_equals_gpu(self) -> bool | None:
        """True if every captured output equals the simulator's bit for bit (`gpu_state`
        "bitwise"); False for any other comparison, within one ulp included; None: no GPU
        output in the case."""
        s = self.gpu_state
        return None if s is None else s == "bitwise"

    def __repr__(self) -> str:
        how = (
            "certified"
            if self.certified
            else ("complete" if self.complete else "INCOMPLETE")
        )
        gpu = {True: ", sim=GPU", False: f", sim!=GPU ({self.gpu_state})", None: ""}[
            self.sim_equals_gpu
        ]
        f = f", findings={self.findings}" if self.findings else ""
        if self.refuted_opt_in:
            f += f", assumes refuted {','.join(self.refuted_opt_in)}"
        return f"Verdict(classes={self.classes}, {how}, states={self.states}{gpu}{f})"


@dataclass(repr=False)
class Equivalence:
    """Equivalence of a candidate kernel to a reference (`ptxv compare`).

    level: "L0" (bitwise), "L1" (same function, bound per output), "L2", elementwise levels,
    "none" (a witness shows they differ), "not attempted"; proven: every precondition held.
    bound: the per-output bound as a callable `Bound` (None when not exported).
    """

    level: str
    proven: bool
    headline: str
    blockers: list[str]
    relies_on: list[str]
    witness: Any
    gpu: dict | None
    per_output_bound: dict | None
    bound: Bound | None
    reference: dict
    candidate: dict
    json: dict

    @classmethod
    def from_json(cls, doc: dict) -> "Equivalence":
        eq = doc.get("equivalence") or {}
        b = doc.get("bound")
        if isinstance(b, list):  # several candidates: compare has one
            b = b[0] if b else None
        return cls(
            level=eq.get("level", "not attempted"),
            proven=bool(eq.get("proven")),
            headline=doc.get("headline", ""),
            blockers=list(doc.get("blockers", [])),
            relies_on=list(doc.get("relies_on", [])),
            witness=eq.get("witness"),
            gpu=doc.get("gpu"),
            per_output_bound=eq.get("per_output_bound"),
            bound=Bound(b) if b else None,
            reference=(doc.get("reference") or {}).get("side", {}),
            candidate=(doc.get("candidate") or {}).get("side", {}),
            json=doc,
        )

    @property
    def markdown(self) -> str:
        return self.json.get("report_md", "")

    def to_pandas(self):
        """Per output: bound form and coefficient (units of u), and the bound at the captured
        inputs when the two cases share them."""
        import pandas as pd

        df = self.bound.to_pandas() if self.bound is not None else pd.DataFrame()
        cap = ((self.json.get("equivalence") or {}).get("captured") or {}).get(
            "bound_abs"
        )
        if cap is not None:
            df["bound_at_captured"] = cap
        return df

    def _repr_markdown_(self) -> str:
        return self.markdown or self.headline

    def __repr__(self) -> str:
        b = ""
        if self.bound is not None:
            b = f", bound=<tau(x): {self.bound.n} outputs, max {self.bound.max_u():.3g}u>"
        bl = f", blockers={len(self.blockers)}" if self.blockers else ""
        return f"Equivalence(level={self.level!r}, proven={self.proven}{b}{bl})"


class ErrDist:
    """An error distribution.

    From samples (shape (trials, outputs)): `ErrDist(samples, metric=...)`. From a summary
    (quantile points, moments: ptxv errdist's report): `ErrDist.from_summary(...)`; there,
    quantiles interpolate the reported points, `cdf` inverts them, and `sample` draws by
    inverse transform. metric: "abs", "u" (units of u S(x)), "ulp", or "bound" (the proven
    bound tau(x)); source: what produced it. Sample statistics pool all elements.
    """

    def __init__(
        self,
        samples: Any,
        metric: str = "abs",
        source: str = "",
        output_shape: Sequence[int] | None = None,
        meta: dict | None = None,
    ) -> None:
        s = np.asarray(samples, np.float64)
        self.samples: np.ndarray | None = (
            s.reshape(s.shape[0], -1) if s.ndim > 1 else s.reshape(-1, 1)
        )
        self.metric = metric
        self.source = source
        self.output_shape = (
            tuple(output_shape) if output_shape else (self.samples.shape[1],)
        )
        self.meta = dict(meta or {})
        self.report: dict | None = None
        self._flat = np.sort(self.samples[np.isfinite(self.samples)].ravel())
        self._q: np.ndarray | None = None  # summary mode: [[p, x]]
        self._moments: dict = {}
        self._n = int(self.samples.shape[0])

    @property
    def worst_case(self) -> dict | None:
        """ptxv errdist reports: the proven worst case beside the distribution (pairs: the
        equivalence bound |C - R| <= rel |y| + k_u u sum|terms| + abs; single kernels: the
        kernel's own bound against the exact value), with its distribution in output ulps
        over the Tier 3 inputs and the check of every observed difference against it
        (`checks`, `exceed` must be 0). None outside errdist reports."""
        return (self.report or {}).get("worst_case")

    @property
    def tier(self) -> int | None:
        """ptxv errdist reports: the headline's tier (0 proof, 1 opt-in residue chain, 2 CLT
        where its applicability check and the Tier 3 cross-check pass, else 3 Monte Carlo).
        None when no estimate passed its checks (without Tier 3, mc_inputs=0, only a Tier 0
        certificate can)."""
        r = self.report or {}
        e = r.get("estimates") or []
        b = r.get("best")
        return e[b]["tier"] if b is not None and b < len(e) else None

    def as_input(self) -> Any:
        """This kernel's output as the next kernel's input: a `ptxv.dist.Upstream` over the
        run's errstate (joint value / error pairs; needs `errstate=True`)."""
        from .dist import Upstream

        es = (self.report or {}).get("errstate")
        if not es:
            raise ValueError("no errstate in this report (run with errstate=True)")
        return Upstream(es)

    @property
    def crosscheck(self) -> list:
        """ptxv errdist reports: each analytic estimate against the run's Tier 3 (p, p3, its
        interval, the tolerance, `agrees`); an estimate that disagrees is never the
        headline. Empty without Tier 3 (mc_inputs=0)."""
        return (self.report or {}).get("crosscheck") or []

    @classmethod
    def from_summary(
        cls,
        quantiles: Sequence[tuple[float, float]],
        moments: dict,
        metric: str = "abs",
        source: str = "",
        n: int = 0,
        hist: Any = None,
        meta: dict | None = None,
        report: dict | None = None,
    ) -> "ErrDist":
        self = cls.__new__(cls)
        self.samples = None
        self.metric, self.source = metric, source
        self.output_shape = ()
        self.meta = dict(meta or {})
        self.report = report
        self._flat = np.empty(0)
        q = np.asarray(sorted(quantiles), float).reshape(-1, 2)
        self._q = q
        self._moments = dict(moments)
        self._n = int(n)
        self.hist = hist
        return self

    @property
    def summary_only(self) -> bool:
        return self.samples is None

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        if self.summary_only:
            return float(self._moments.get("mean", math.nan))
        return float(self._flat.mean()) if self._flat.size else math.nan

    @property
    def std(self) -> float:
        if self.summary_only:
            return float(self._moments.get("std", math.nan))
        return float(self._flat.std()) if self._flat.size else math.nan

    @property
    def max(self) -> float:
        if self.summary_only:
            m = self._moments.get("max_abs")
            return (
                float(m)
                if m is not None
                else (float(self._q[-1, 1]) if len(self._q) else math.nan)
            )
        return float(self._flat[-1]) if self._flat.size else math.nan

    def quantile(self, q: float | Sequence[float]) -> Any:
        if self.summary_only:
            if not len(self._q):
                return math.nan
            r = np.interp(q, self._q[:, 0], self._q[:, 1])
        else:
            r = np.quantile(self._flat, q) if self._flat.size else np.nan
        return float(r) if np.ndim(r) == 0 else np.asarray(r)

    @property
    def quantiles(self) -> dict[float, float]:
        qs = (0.5, 0.9, 0.99, 0.999, 1.0)
        return {q: self.quantile(q) for q in qs}

    def cdf(self, t: float | np.ndarray) -> Any:
        """P(err <= t) (summary mode: interpolated between reported quantiles)."""
        if self.summary_only:
            r = np.interp(t, self._q[:, 1], self._q[:, 0], left=0.0, right=1.0)
        else:
            r = np.searchsorted(self._flat, t, side="right") / max(self._flat.size, 1)
        return float(r) if np.ndim(r) == 0 else r

    def tail(self, t: float, confidence: float = 0.99) -> tuple[float, float]:
        """(P(err > t), one-sided Clopper-Pearson upper bound at `confidence`) over the pooled
        sample elements. Summary mode: (1 - cdf(t), nan)."""
        if self.summary_only:
            return 1.0 - self.cdf(t), math.nan
        n = self._flat.size
        k = int(n - np.searchsorted(self._flat, t, side="right"))
        return k / max(n, 1), _cp_upper(k, n, confidence)

    def per_output(self, stat: str = "max") -> np.ndarray:
        """A statistic per output element over trials, shaped like the output."""
        if self.summary_only:
            raise ValueError(
                "a summary has no per-output samples (ptxv errdist --per-output)"
            )
        f = {"max": np.nanmax, "mean": np.nanmean, "median": np.nanmedian}[stat]
        return f(self.samples, axis=0).reshape(self.output_shape)

    def sample(self, n: int, rng: Any = None) -> np.ndarray:
        """n draws: a bootstrap of the samples, or inverse transform of the quantiles."""
        g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        if self.summary_only:
            return np.interp(g.random(n), self._q[:, 0], self._q[:, 1])
        return g.choice(self._flat, size=n)

    def to_numpy(self) -> np.ndarray:
        if self.summary_only:
            return self._q.copy()
        return self.samples

    def to_pandas(self):
        import pandas as pd

        if self.summary_only:
            return pd.DataFrame({"p": self._q[:, 0], self.metric: self._q[:, 1]})
        t, o = np.meshgrid(
            np.arange(self.samples.shape[0]),
            np.arange(self.samples.shape[1]),
            indexing="ij",
        )
        return pd.DataFrame(
            {"trial": t.ravel(), "output": o.ravel(), self.metric: self.samples.ravel()}
        )

    @property
    def json(self) -> dict:
        if self.report is not None:
            return self.report
        return {
            "metric": self.metric,
            "source": self.source,
            "trials": self.n,
            "outputs": int(self.samples.shape[1]) if self.samples is not None else None,
            "mean": self.mean,
            "std": self.std,
            "max": self.max,
            "quantiles": {str(k): v for k, v in self.quantiles.items()},
            **({"meta": self.meta} if self.meta else {}),
        }

    def __repr__(self) -> str:
        q = self.quantiles
        shape = (
            f", outputs={self.samples.shape[1]}"
            if self.samples is not None
            else ", summary"
        )
        return (
            f"ErrDist(metric={self.metric!r}, n={self.n}{shape}, mean={self.mean:.3g}, "
            f"p50={q[0.5]:.3g}, p99={q[0.99]:.3g}, max={self.max:.3g}"
            + (f", source={self.source!r}" if self.source else "")
            + ")"
        )


def _cp_upper(k: int, n: int, conf: float) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial proportion (bisection on the
    regularized incomplete beta through the binomial CDF; no scipy)."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    alpha = 1.0 - conf
    if k > 2000:  # normal approximation with continuity correction (k large)
        z = math.sqrt(2) * _erfinv(1 - 2 * alpha)
        p = (k + 0.5) / n
        return min(1.0, p + z * math.sqrt(p * (1 - p) / n))

    def cdf(p: float) -> float:  # P(X <= k), X ~ Bin(n, p), in log space
        if p <= 0:
            return 1.0
        if p >= 1:
            return 0.0
        lp, lq = math.log(p), math.log1p(-p)
        terms = [
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * lp
            + (n - i) * lq
            for i in range(k + 1)
        ]
        m = max(terms)
        return math.exp(m) * sum(math.exp(t - m) for t in terms)

    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def _erfinv(y: float) -> float:
    """Inverse error function (Newton on math.erf)."""
    x = 0.0
    for _ in range(60):
        e = math.erf(x) - y
        x -= e / (2 / math.sqrt(math.pi) * math.exp(-x * x))
    return x
