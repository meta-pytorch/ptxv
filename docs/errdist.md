# Error distributions under an input distribution (`ptxv errdist`)

**Status (2026-11-05, agent T, phase 2).** Tiers 0-3, simulator / GB300 validation, the
JSON report, and a compile-once / query-many split (§1.1) are implemented. §2 (input spec,
version 1) and §3 (report, version 1) are fixed interfaces.
- **Tier 0** (exact) and **Tier 3** (Monte Carlo through the DAG, bit-exact per sample,
  matching GB300 on every run in §7) are the ground truth.
- **The default (§3.2)**: Tier 0 when it applies, else **Tier 2** (the DAG CLT: reductions,
  split-K, atomics, scans, FMA and MMA chains, 1x / 3xTF32; AG's linearized tiers on
  non-chain programs) where its applicability check passes, else Tier 3. Every run also
  runs a small Tier 3 (300 inputs) as a cross-check: a Tier 2 that disagrees with it beyond
  its interval is refused and the headline falls back to Tier 3.
- **Tier 1** (residue-exact Markov chain; AG's exact-site convolution on non-chain
  programs) is opt-in: `--tier1`, `tier1=True`.
- **Inputs that carry an error, and composition (§2.1, §10).** A tensor may carry an error
  (relative, absolute, a 16-bit rounding, or an upstream kernel's errstate). The kernel
  reads value + error, the reference is exact on the error-free value. The linearized tier
  carries J e_in + e_own, and refuses where the kernel is not linear at the error's scale.
  `--errstate FILE` writes a kernel's output error state for the next kernel, so
  multi-launch regions compose stage by stage (Python: `m1.then(m2, "h")`).

`ptxv equiv` proves a worst-case bound τ(x) on |C(x) − R(x)| for every input. `ptxv errdist`
estimates the *distribution* of the error for inputs drawn from a user-given distribution:

- one kernel: C − exact, the exact value being the kernel's own algorithm in real arithmetic
  on the same (quantized) inputs;
- a pair: C − R.

These are distributional estimates with stated error bars and modeling assumptions
(ASSUMPTIONS.md, N-ERRDIST-*), not worst-case proofs. The worst-case τ of `equiv` stays the
guarantee.

## 1. Usage

```
ptxv errdist <case-dir | equiv.toml> --inputs spec.json [--json out.json] [--html out.html]
             [--candidate N] [--mc N] [--sim-trials N] [--gpu-trials N] [--per-output]
             [--pre-round] [--uniform] [--explore] [--threads N] [--out DIR] [--tier1]
             [--errstate FILE]
ptxv errdist <case-dir | equiv.toml> [--candidate N] --compile [FILE]
ptxv errdist --model FILE --inputs spec.json [--json out.json] [--mc N] [--tier1]
```

- A case directory: one kernel, C − exact.
- An `equiv.toml`: the reference against candidate N (default 0), C − R.
- `--inputs` takes JSON or TOML (by extension). The text report goes to stdout, the JSON
  report to `--json`.
- `--pre-round` drops the final 16-bit rounding: the f32 accumulators are compared
  (diagnostics for the 16-bit flip model).
- `--uniform` adds the Tier 1 estimate under the uniform-residue heuristic, for comparison.
- `--explore` prints per-step local-error statistics from 50 sampled inputs.
- `--gpu-trials N` runs N sampled inputs on a free GB300 (`PTXV_GPUS` restricts
  the set). Keep N ≤ 1000 on shared GPUs.
- `--mc N`: Tier 3 inputs, default 300 (the cross-check and the fallback; `--mc 0` skips
  it, and the chain tiers then rest on their structural checks alone).
- `--threads N`: Tier 3 workers. The default is the number of cores, capped so the workers
  fit in 70% of the memory left: the cgroup's limit less its use (e.g. `systemd-run -p
  MemoryMax=48G`), else MemAvailable. A note gives the count and the estimate per worker.
  Tier 3 draws only the input elements the DAG's leaves read (whole MX / NVFP4 scale
  blocks), and loops over tracked outputs only. A one-tile cut of a 4096 x 4096 x 8192
  GEMM samples 2M of 67M elements per input.
  - Each element comes from its own stream, keyed (seed, sample, buffer, element), so a
    value does not depend on which other elements are drawn.
  - Tier 3 numbers moved within their seed-to-seed spread when this sampler replaced the
    sequential one.
- `--tier1` also runs Tier 1 (seconds to minutes; 15+ min on MXF8 random codes; about 13 min
  for a K = 960 bf16 chain, hours at K = 4096; a few GB, flat in K). With `--json`, a preliminary report is written before the analytic tiers
  run. `--no-tier1` is accepted and ignored.
- `--out DIR`: scratch directory; by default one per process under `$TMPDIR`, removed at the
  end (concurrent runs never share one).
- `--errstate FILE` writes the output's error state (§10) from Tier 3's (value, error)
  pairs: the next kernel's input `"error": {"kind": "pairs", "file": FILE}`.
- `--compile [FILE]` / `--model FILE`: the model file (§1.1).
- `--html FILE` writes one self-contained page (inline SVG, no scripts or external assets):
  P(differ) per tier and observed with 95% intervals, the PMFs over output ulps overlaid
  (predictions dashed, observations solid, log axis, |k| ≤ 24) with a table view, and the
  text summary.

Guards (never report from bad data):
- **The DAG must agree with the simulator.** With `--inputs`, the extracted numerics DAG is
  evaluated on two sampled inputs and compared bit for bit with the simulator's tracked
  outputs. A graph that folded an input-dependent value to a constant (bf16 ↔ f32 bit
  conversions, Kahan's compensation) reproduces the captured input only. On a mismatch the
  kernels are re-extracted in segment mode; if that graph disagrees too, errdist refuses.
  A DAG mismatch in the simulator validation is also a refusal.
- **NaN / Inf outputs are out of domain**, counted and reported: the `out_of_domain` cause
  `non-finite output (NaN / Inf)` (fraction of inputs) and a note with the fraction of
  outputs; a Tier 3 that compared no output says so. A validation run whose outputs have
  NaN / Inf that Tier 3 does not, or with nothing to compare, does not agree.

Examples (the runs of §7):

```
PTXV_GPUS=3 ptxv errdist fixtures/c_equiv_mm_bf16/equiv.toml --candidate 1 \
    --inputs fixtures/errdist/normal.json --mc 4000 --gpu-trials 1000 --json out.json
ptxv errdist fixtures/c_equiv_mxf8_triton/equiv.toml --candidate 1 \
    --inputs fixtures/errdist/mxf8_codes.json --mc 400
ptxv errdist fixtures/r3_cute_mxf8 --inputs fixtures/errdist/mxf8_gauss.json --mc 300 --uniform
```

The Python API (`python/ptxv`, agent U) runs `ptxv.errdist(target, inputs, ref=ref,
gpu_trials=N)`. It serializes `ptxv.dist` distributions to the §2 spec and reads the §3
report into an `ErrDist`.

### 1.1 Compile once, query many

The expensive part of a run is simulating the kernel(s) and extracting their numerics
programs. One **model file** holds everything a query needs, as a JSON object with two
sections:
- `programs` (agent T): both kernels' `Prog`s, including non-chain ops (instruction slices
  with their identified functions, MUFU ops, maxima), the input layouts, the kernel ids and
  labels, and each program's slice identifications;
- `nonlin` (agent AG, §9.6): the slice / MUFU / pair tables of non-chain programs, keyed by
  the programs' canonical hash (`nonlin::model_key`).

Each writer keeps the other's section. **One cache key**: the file is named and checked by
`case_key`, which hashes every file of the case directories, the equiv.toml, the candidate
index and the ptxv version; the nonlin section's canonical hash is a consistency check
inside it.

- `--compile [FILE]` simulates the kernel(s) and writes the programs section to FILE, or
  without FILE to the cache, `$PTXV_CACHE/errdist/<case key>.model.json` (else
  `~/.cache/ptxv/errdist`). Without `--inputs` it stops there. With `--inputs` it also runs
  the full report (Tier 3, simulator validation, analytic tiers) and writes the non-chain
  tables (AG's original `--compile FILE`).
- `--model FILE --inputs spec.json` is **the** query path for every program: no simulator,
  no validation runs, the default 300-input Tier 3 cross-check (`--mc N`; 0 skips it).
  Chain programs get Tiers 0 and 2 (§5), Tier 1 with `--tier1`.
  Non-chain programs get AG's tiers (§9) from the nonlin section, which is compiled at the
  first query (or extended when the spec reaches new binades) and written back, plus the
  DAG CLT where the DAG is a sum of products. Given a case path too, the case key must
  match the file's.
- `best` picks across all of them in tier order (§3).

Measured on a 144-core host. Compile takes 0.7-2.2 s for the small fixtures; it is the
simulator run, so minutes for large kernels. Query times:
- Tier 2: 0.03-2 s (sums, TF32, bf16);
- AG's tiers: 6-13 s (softmax);
- Tier 1: 8-13 s (bf16 normal), 25-200 s (MX single kernels; §7.1).

Tier 1's cost per query is its chain, which depends on the input law, so the compile stage
cannot precompute it; it is opt-in (`tier1=True` / `--tier1`).

Python: `m = ptxv.errdist_model(target, ref=None)` compiles once (cached under
`$PTXV_CACHE/errdist-models/<hash of the case trees and the binary>`) and returns `m` with
`m(inputs, metric="abs", tier1=False, mc_inputs=None) -> ErrDist` (`mc_inputs` None: the
default 300-input cross-check; 0 skips it). `ErrDist.tier` is the headline's tier and
`ErrDist.crosscheck` the cross-check entries. Queries are memoized per input distribution,
in memory and on disk:

```python
m = ptxv.errdist_model("fixtures/r4_sum_torch")
m(ptxv.dist.Normal(0, 1)).meta["p_differ"]     # 0.678 (Tier 3: 0.674), 0.9 s
m(ptxv.dist.Uniform(1, 2)).meta["p_differ"]    # 0.282 (Tier 3: 0.285), 0.03 s
```

## 2. Input spec, version 1

Schema: `crates/sim/src/errdist/spec.rs` (serde types; unknown fields are errors).

```json
{
  "version": 1,
  "seed": 1,
  "name": "gaussian activations, MXF8",
  "tensors": {
    "A": {"dist": {"kind": "normal", "std": 1.0},
          "quant": {"kind": "mx", "elem": "e4m3", "block": 32, "scales": "SFA"}},
    "B": {"dist": {"kind": "laplace", "scale": 0.5},
          "scales": [{"axis": 0, "dist": {"kind": "log_normal", "sigma": 0.5}}],
          "outliers": {"axis": 1, "prob": 0.01, "factor": 20.0},
          "quant": {"kind": "mx", "elem": "e4m3", "block": 32, "scales": "SFB"}}
  },
  "default": {"dist": {"kind": "normal"}},
  "engine": {"mc_inputs": 2000, "sim_trials": 20, "gpu_trials": 0}
}
```

**Top level.**

| field | type | meaning |
|---|---|---|
| `version` | 1 | required; other values are rejected |
| `seed` | u64 | base seed; sample t of buffer b uses the stream (seed, t, b), independent of order |
| `tensors` | map | per input buffer of the case (the case's buffer name) |
| `default` | tensor | for input buffers without an entry |
| `engine` | object | optional `mc_inputs`, `sim_trials`, `gpu_trials`, `per_output` (CLI flags override) |
| `name` | string | echoed in the report |

A scale buffer named by a recipe (`quant.scales`) is filled by that recipe and gets no entry.

**Tensor.**

| field | meaning |
|---|---|
| `dist` | the element distribution, before scales and quantization |
| `scales` | list of `{axis, dist}`: per input sample, one draw per index along `axis` multiplies every element with that index (per-row / per-channel scales) |
| `outliers` | `{axis, prob, factor}`: each index along `axis` is an outlier channel with probability `prob`, its elements times `factor` |
| `quant` | how real values become the buffer's codes (default `round`) |
| `shape` | logical row-major shape the axes refer to (default: the buffer's; e2m1 codes: last dimension doubled) |

Elements are independent given the axis draws (ASSUMPTION N-ERRDIST-IID): the spec defines the
input distribution, and only what it states is modeled.

**Distributions** (`dist`, tagged by `kind`):

| kind | fields | |
|---|---|---|
| `normal` | `mean` = 0, `std` = 1 | |
| `uniform` | `lo`, `hi` | |
| `laplace` | `mean` = 0, `scale` = 1 | |
| `student_t` | `df`, `mean` = 0, `scale` = 1 | |
| `log_normal` | `mu` = 0, `sigma` = 1, `signed` = false | random sign when `signed` |
| `mixture` | `components: [{weight, dist}]` | |
| `discrete` | `values`, `probs` (optional, uniform) | explicit PMF |
| `integers` | `lo`, `hi` | uniform on lo..=hi |
| `const` | `value` | |
| `empirical` | `file`, `dtype` = f32 | i.i.d. resampling of a captured tensor's finite values (f32, bf16, f16, or 1-byte codes of a narrow format) |
| `histogram` | `edges`, `counts` or `file` | piecewise-uniform density; file is JSON `{"edges", "counts"}` |
| `codes` | `format`, `range` = [lo, hi] or `values`, `probs` | codes drawn directly (with `quant: codes`); without range / values: every finite code |
| `affine` | `a` = 1, `b` = 0, `dist` | a X + b |

Formats: `f32`, `bf16`, `f16`, `e4m3`, `e5m2`, `e2m1`, `ue8m0`, `ue4m3`.

**Quantization recipes** (`quant`, tagged by `kind`):

| kind | fields | codes |
|---|---|---|
| `round` | | round to nearest even to the buffer's float dtype (f32 / bf16 / f16); overflow is out-of-domain |
| `codes` | `format` (default: what the kernel reads) | the distribution's values are the codes |
| `cast` | `format` | RNE to a narrow format, saturating (`cvt.rn.satfinite`), no scale |
| `mx` | `elem`, `block` = 32, `scales`, `scale_rule` = `ceil` | per block of `block` elements along K: ue8m0 scale 2^E with E = ceil(log2(amax / max_elem)) (`ceil`) or floor(log2 amax) − emax_elem (`floor`, OCP MX v1.0); elements RNE(x / 2^E), saturating |
| `nvfp4` | `block` = 16, `scales`, `global_scale` = false | e2m1 elements, ue4m3 scale RN(amax_b / (6 gs)), gs = amax_tensor / 2688 when `global_scale` (else 1); elements RNE(x / (s gs)). The kernel's output is in units of gs_A gs_B |

The joint law of (codes, scales) is derived from the recipe, not assumed independent: the block
maximum sets the scale, so it lands in the top binade of the element format. Which code
elements share a scale byte is read from the kernel's numerics DAG (the scale leaf each code
leaf is multiplied with), so the recipe does not need the scale layout (swizzled or not).

Equivalent forms in TOML are accepted (same field names).

### 2.1 Inputs that carry an error

A tensor's `error` field says the kernel does not read the spec's value x but
round_dtype(x + e). The reference stays the exact computation on x, so the report is the
kernel's output error against the exact result on error-free inputs: the upstream error
pushed through the kernel, plus the kernel's own. Elements' errors are independent
(N-ERRDIST-INERR). The tensor needs `quant: round` (a float buffer).

| kind | fields | e given x |
|---|---|---|
| `relative` | `std`, `mean` = 0 | x (mean + std Z) |
| `absolute` | `std`, `mean` = 0 | mean + std Z |
| `round` | `format` (bf16 / f16 / ...) | RNE(x) to `format` minus x: an f32 value that went through a 16-bit tensor |
| `linear` | `alpha`, `beta`, `s0`, `s1` | N(alpha + beta x, s0^2 + s1^2 x^2) |
| `pairs` | `pairs: [[x, e], ...]` or `file` (an errstate, §10) with `tensor` | (x, e) drawn jointly from the pairs: the value comes from the pair too, and `dist` is unused |

```json
"h": {"dist": {"kind": "normal", "std": 8}, "error": {"kind": "relative", "std": 1e-6}}
"h": {"dist": {"kind": "normal"}, "error": {"kind": "pairs", "file": "mm.errstate.json"}}
```

- **Tier 3** samples e for every element and runs the kernel on round(x + e), with the exact
  reference computed on x.
- **The linearized tier** (§9.3) takes the error as the leaf's local law. That is its moments
  given x, a normal with the format's rounding variance added, or the exact rounding law for
  `round`. The law is pushed through the derivatives like every other site (J e_in + e_own).
  For `pairs` the moments come per value bin: about 256 pairs per bin, a `linear` form fitted
  in each. A single `linear` fit over all pairs was 2-3x off in the middle of a softmax
  output's range. One constant per bin missed the growth across the top bin, where the
  largest logits sit (attention 1.12x Tier 3, now 1.03x).
- **Refused where the kernel is not linear at the error's scale.** At 16 nominal inputs,
  the error at one standard deviation is applied as +e and -e (all errored leaves at once).
  The tier is refused when the second-order effect |d+ + d-| exceeds 0.1 of the first-order
  effect |d+ - d-| (summed over the outputs). Examples:
  - a softmax of N(0, 4^2) logits with an absolute error of 0.5: 0.25, refused;
  - an RMSNorm whose rows have tiny variance against the error: 0.39, refused;
  - the same kernels with small errors: 0.000.
- **The cross-check (§3.2) also compares the spread.** With input errors the outputs almost
  never round alike, so P(differ) near 1 says little. The tier's mean |error| must also be
  within 10% of Tier 3's. The std is not used: a softmax fed by an upstream error has max /
  std ~ 40, and 300 inputs leave Tier 3's std 10% apart from seed to seed. The tier uses at
  least 192 nominal inputs here; at 48 it moved +-15% between seeds.
- **The chain tiers** (exact certificates, residue chains, the DAG CLT) do not model input
  errors. With an `error` on any input, the linearized tier runs on chain programs too
  (matmuls, reductions), and a note says so.
- **Pairs** (C - R, both reading the same round(x + e)): the linearized pair tier is
  usually refused, as it is without input errors (§9.8), and Tier 3 answers. The elementwise
  law (§9.5) does not take input errors.

## 3. Report, version 1

Schema: `crates/sim/src/errdist/report.rs`. Top level:

| field | meaning |
|---|---|
| `version` | 1 |
| `kind` | `"ptxv-errdist"` |
| `mode`, `quantity` | `"single"`, `"C - exact"` or `"pair"`, `"C - R"` |
| `reference`, `candidate` | `{case, variant, front}` |
| `spec` | the input spec, echoed |
| `outputs`, `output_format`, `templates` | output elements per input, their format, chain templates |
| `estimates` | one entry per tier that ran (below) |
| `best` | index into `estimates` of the headline estimate (§3.2): Tier 0 when it applies, else a Tier 1 (opt-in), else a Tier 2 whose `applicable` is true, else Tier 3 |
| `crosscheck` | per analytic estimate against this run's Tier 3: `{estimate, tier, p, p3, p3_lo, p3_hi, tol, inputs, agrees}` (§3.2) |
| `out_of_domain` | `{mass, by_cause, method}`: probability that an input leaves the modeled domain (overflow, NaN / Inf, zero or NaN scales, subnormal accumulators), never dropped silently |
| `validation` | predicted vs observed on sampled inputs (simulator, GB300) |
| `residue` | per rounding site: largest Fourier coefficient of the residue at the ulp frequency, TV bound of the uniform model, whether it was used |
| `per_output` | with `--per-output` |
| `assumptions`, `notes` | |
| `worst_case` | the proven worst case beside the distribution (§3.1) |

An **estimate**:

| field | meaning |
|---|---|
| `tier`, `method` | 0 exact, 1 residue-exact, 2 CLT with bias, 3 Monte Carlo |
| `n`, `outputs` | inputs or states behind it; outputs per input |
| `abs` | moments of the absolute error |
| `u_scale` | moments of error / (u S(x)), u = 2^-24, S = Σ over the output's terms of \|term\| (the scale of `equiv`'s τ = k u S) |
| `ulp` | moments of error / ulp_out(y) |
| `pmf_ulp` | PMF of k = (output − reference) / ulp_out(y), `[k, p]`, on one axis for every tier: the reference is R (pair) or RN(y) (single). Tier 0: exact (`{0: 1}` under a certificate); Tier 1: the residue-exact law through the tent kernel below (16-bit outputs, so tails beyond ±1 wherever \|D\| exceeds a 16-bit ulp) or round(e) (f32 single); Tier 2: the zero-inflated normal (chain CLT) or the DAG laws through the tent kernel, the mass off 0 rescaled to the tier's P(differ) when its finer position law differs (noted); Tier 3: counted. Entries below 1e-12 are dropped. A tier without one says so in `error_bar.notes` |
| `hist_u` | histogram of the u-scale error, `[edges, probs]` |
| `p_differ` | `[p, lo, hi]`: probability that an output differs (pair: from R; single: from the correctly rounded exact value), 95% interval |
| `pair` | `{mean_r, var_r, mean_c, var_c, cov_rc, corr_rc}`: the two kernels' errors against the exact value |
| `error_bar` | `{kind, cdf_sup, mean_hw, std_rel_hw, model_mass, notes}` |
| `applicable`, `applicability` | the tier's own applicability verdict and the checks behind it (Tiers 1-2; absent for 0 and 3) |

Moments: `{mean, var, std, mean_abs, p_zero, quantiles: [[p, x]], max_abs}`. The aggregate pools
all outputs (each output element of each input equally weighted).

A **validation** entry: `{source, against, inputs, outputs, mean_pred, mean_obs, std_pred,
std_obs, p_differ_pred, p_differ_obs, ks_d, ks_p, ad, ad_p, qq: [[p, pred, obs]],
pmf_ulp, pmf_ulp_pred, pmf_tv, dag_mismatch, agree, notes}`. `pmf_ulp` is the observed PMF on
the estimates' axis, `pmf_ulp_pred` the compared estimate's, `pmf_tv` their total-variation
distance (the observed side has only inputs × outputs samples).

The tent kernel: with y's position t in its output ulp uniform, an error x (output ulps,
before the output rounding) gives k = Q(t + x) − Q(t), so P(k | x) = max(0, 1 − |x − k|);
P(k ≠ 0) = E[min(1, |x|)] is the flip formula of §5. A final round toward zero against
RN(y) shifts x by −1/2 toward |y|. For x ~ N(m, s²), P(k) = G(k−1) − 2G(k) + G(k+1) with
G(a) = E[(x − a)+], in closed form.

### 3.1 The worst case

Every report carries the proven worst case next to the distribution, however rare it is:

- pairs: the equivalence proof of `ptxv equiv` (`equiv::compare`): every output's L1 bound;
- single kernels: the kernel's own rigorous bound against the exact value (`equiv::own_bound`,
  Cor 1a; a 16-bit output adds its rounding, u16 |y|, N-ROUND).

`worst_case` holds the largest coefficients over the outputs of
|e| <= rel |y| + k_u u S(x) + abs (u = 2^-24, y the exact output, S(x) = Σ|a_k b_k|, the
scale of `equiv`'s bound; softmax pairs: rel y + k_u u + abs), in three forms:

1. `formula`, e.g. `|C - R| <= 131072u*|y| + 59.73u*sum|terms| + 9.2e-41` (c_equiv_mm_bf16
   candidate 1);
2. `human`, e.g. "|C - R| <= 2 bf16 ulps of the exact result + 3.56e-6 x sum|a_k*b_k|";
3. `dist`: the bound in output ulps of y over the Tier 3 inputs (mean, quantiles, max;
   outputs with y = 0 or subnormal left out).

`checks` compares every observed difference with the bound: the Tier 3 DAG evaluation, the
simulator and GB300 runs, with the largest |difference| / bound; `exceed` must be 0, and a
violation adds a "WORST CASE EXCEEDED" note. Without a bound (`available` false), `why` says
why (a MUFU or slice term, not L1, an untracked output, or a proof too large: errdist skips
it above 1e9 monomials over the DAG's polynomial forms, quadratic in the chain length, about
K = 2000 for a 64 x 64 bf16 GEMM pair; `ptxv equiv` has no such limit). Preconditions of the
proof (the kernels' own, N-TF32) are listed; `proven` is true without any. The bound is computed when
the model is compiled and stored in its `programs` section, so `--model` queries report it
too. Text output, `--html` and Python (`ErrDist.worst_case`) show it.

### 3.2 Default policy and the Tier 3 cross-check

The headline (`best`) is Tier 0 when it applies (a certificate, identical programs, exact
enumeration), else Tier 2 where it is applicable, else Tier 3. Tier 1 is opt-in
(`--tier1`); with it, an applicable Tier 1 comes before Tier 2.

Tier 2 is applicable when two checks pass. A confident wrong Tier 2 is the failure to
avoid, so both are conservative.

1. **Structural** (before any sampling; recorded in `applicability`):
   - the DAG CLT: y's excess kurtosis, from the product terms' moments, at most 1
     (`tier2::KURT_MAX`). The tier treats the partial sums as normal given y; heavy-tailed or
     scale-mixture terms break that (a few large terms dominate y, and the partial sums
     before them are far smaller than the normal law says). Measured: bf16 GEMM with normal
     inputs 0.09; laplace 0.5; outliers (2% N(0, 64²)) 335; lognormal 62; Student t(3)
     4e8;
   - the chain CLT on block-scaled chains (`tier2::single` / `pair`): always refused (the
     matrix has it 2-100x off);
   - the block-scaled chain tier (`tier2bs`, §5.3.1): y's excess kurtosis at most 1, from
     the terms' group-mixture cumulants (a block scale or an outlier channel is a mixture
     over the group). Measured on MXF8: normal 0.02, laplace 0.13, codes 77, outliers 84,
     ints_scaled 34, lognormal 15.5, Student t 1.6e5. Pairs of block-scaled kernels:
     refused, not run;
   - AG's non-chain tiers: their own checks (§9), and the cross-check below is required
     (refused without Tier 3).
2. **The Tier 3 cross-check**: every run samples Tier 3 (300 inputs to start). An analytic
   estimate agrees when |p − p3| ≤ max(5% of p3 − h, h), or p lies inside Tier 3's
   interval. Here h is half Tier 3's interval: Tier 3's own uncertainty comes off the 5%,
   so an agreeing estimate is within 5% of the true P(differ), not merely of this run's
   Tier 3.
   - **The check must have resolution.** With the default input count, Tier 3 grows until
     h is within 10% of P(differ), stopping at 76,800 inputs or after 30 s
     (`PTXV_XCHECK_SECONDS`). It then keeps growing toward 2% for as long as the run is
     under 10 s.
   - **A wide tolerance confirms nothing.** An estimate whose tolerance is still above 20%
     of P(differ) is not confirmed, and so not applicable. `--mc N` fixes the input count.
   - **Why:** in the full matrix (§7), the 300-input check let 7 of 162 headlines through
     wrong. At 300 inputs a P(differ) of 3e-5 has a ±30% interval: a bf16 GEMM's Tier 2 at
     0.82x passed, as did a layer norm's at 0.37-0.53x and a packed bf16 gelu's at 4x. A
     row-reduce pair's Tier 2, 5% high, passed against a Tier 3 that landed 3% high. All
     now fail, in 1-7 s of Tier 3.
   - One that disagrees is refused, the report notes `TIER 3 CROSS-CHECK FAILS`, and the
     headline falls back to Tier 3. `crosscheck` lists every comparison. With `--mc 0` (a
     model query without Tier 3), chain estimates rest on the structural checks alone.

The text report prints each analytic tier's verdict ("applicable" or "REFUSED") with its
checks, and ends with a summary: which tier answered, P(differ) (with Tier 3's interval),
the error in output ulps (|k| at the median, 99% and 99.9% of the headline's PMF, and the
largest Tier 3 saw), and the proven worst case. Two runs with the defaults:

```
$ ptxv errdist fixtures/c_equiv_mm_bf16/equiv.toml --candidate 1 \
      --inputs fixtures/errdist/normal.json
...
tier 2 applicable:
    0 of 64 sampled outputs outside the DAG tier's domain, carrying -0.000e0 of P(differ) 3.829e-5 ok
    e.g. MMA DAG (5 error terms), y's excess kurtosis 0.094 <= 1 ok
    Tier 3 cross-check (300 inputs): 3.829e-5 vs 3.499e-5 [2.533e-5, 4.714e-5], tolerance 1.09e-5 ok

summary: tier 2 (CLT on both DAGs), agrees with the tier 3 cross-check
    P(bits differ) = 3.829e-5 (tier 3: 3.499e-5 [2.533e-5, 4.714e-5])
    error in output ulps (|k|): median 0, 99% within 0, 99.9% within 0, largest seen in tier 3: 4
    worst case (proven): |C - R| <= 2 bf16 ulps of the exact result + 3.56e-6 x sum|a_k*b_k| + 9.2e-41

$ ptxv errdist fixtures/c_mm_bf16_nnt --inputs fixtures/errdist/outliers.json
...
tier 2 REFUSED:
    64 of 64 sampled outputs outside the DAG tier's domain, carrying 8.706e-5 of P(differ) 8.706e-5 FAILS
    e.g. y's excess kurtosis 335.194 > 1: heavy-tailed or scale-mixture terms, the partial sums are not normal given y FAILS
    Tier 3 cross-check (300 inputs): 8.706e-5 vs 3.174e-5 [2.257e-5, 4.339e-5], tolerance 1.04e-5 FAILS

summary: tier 3 (Monte Carlo; the analytic tiers refused, see above)
    P(bits differ) = 3.174e-5
    error in output ulps (|k|): median 0, 99% within 0, 99.9% within 0, largest seen in tier 3: 10
    worst case (proven): |C - y| <= 1 bf16 ulp of the exact result + 2.36e-6 x sum|a_k*b_k| + 4.6e-41
```

The first is a bf16 GEMM pair (cuBLAS against Triton with reversed K). The second is one
bf16 GEMM whose inputs have 2% large outliers; there Tier 2 would have said 2.7x Tier 3, and
both checks refuse it.

## 4. The error of an accumulation chain

`errdist/chain.rs` extracts the kernel's numerics DAG (num.rs) into a program `Prog`:
leaves (input codes and scales), MMA steps, roundings, adds and FMAs. Every op stores its
operand, so evaluating one input is a pass over the ops. `Prog::exact` sums an output's
products with a Shewchuk expansion, which is exact.

An output of an MMA kernel is a chain of steps S_k = step(S_{k-1}, terms_k). In all three
calibrated models (N-MMA, N-MMA-F8, N-MMA-BS), a step:
1. takes emax = the largest alignment exponent among its live terms and the accumulator;
2. truncates every term and the accumulator toward zero on the grid Q = 2^(emax − F)
   (F = 25 for f16 / f8 kinds, 35 for FP4);
3. adds them exactly and rounds once toward zero to f32.

The step's local error is

    e_k = τ_acc + Σ_j τ_j + ρ,     ρ = −sign(S')·(|S'| mod ulp(S')),

where τ is minus each value's part below Q and S' is the exact truncated sum. The kernel's
error is Σ_k e_k plus the error of any final rounding.

**Residues.** ρ depends on S' mod ulp(S'). Write r = ulp(S')/Q = 2^(b_S − emax + 2) (f32).
The truncated terms are integers on the Q grid, so S'/Q mod r is a sum of independent
residues on Z_r. Its law is the product of the terms' characteristic functions on Z_r,
inverted by FFT (`fine.rs`), which is exact. The uniform-residue model replaces it with
uniform on Z_r. Its total-variation error is at most ½·min(Σ_k |φ(k)|, √r·‖φ‖₂) over the
nonzero frequencies (`fine::tv_uniform`). The report lists max_k |φ(k)| and that bound per
step (`residue`). The bound is small for continuous inputs at large r. It is not small for
integers, for coarse code sets, or for small K: there ρ is often exactly 0, and the uniform
model is wrong (§7, target 4).

**The emax order statistic and MX block maxima** are exact signed mixtures:
P(max ≤ e) = Π_j P(a_j ≤ e), so the law of the terms given max = e is the product of the
laws restricted to ≤ e, minus the same with ≤ e − 1. An MX block's scale is a function of
the block maximum, so law(block | scale s) = F_hi^n μ_hi^⊗n − F_lo^n μ_lo^⊗n (`laws.rs`,
`TensorLaw::Blocked`). All statistics are linear in the measure, so signed weights carry
through FFTs, convolutions and moments.

**RZ bias.** ρ has the sign opposite to S', with mean −sign(S')·E[residue]·Q. Each τ has
the sign opposite to its value. Both enter the means and the error PMFs exactly.

## 5. Tiers

### 5.1 Tier 0: exact

- **Certificate** (`tier0::certify`). Let 2^L be the coarsest power of two that every term
  value (and so every partial sum) is a multiple of, on the support of the element laws. If
  every step's possible emax satisfies emax − F ≤ L, and every partial sum bound is below
  2^(L+24), then every truncation and rounding is exact and the error is identically 0 (up to
  the pruned mass).
- **Identical programs** (`tier0::identical_outputs`). Two kernels whose outputs have equal
  canonical DAGs compute the same function: C = R, up to the sign of a zero result. The
  canonical form uses the products of a step as a multiset where the model is order-free,
  and folds x + 0 (the cuBLAS β·C epilogue).
- **Enumeration** (`tier0::enumerate`). The exact signed law of (output, error), by dynamic
  programming over the accumulator's f32 value. For each emax, the step's (truncated integer
  sum, dropped part) law is the product of the groups' laws restricted to alignment ≤ emax,
  by sparse convolution, minus the same law restricted to < emax. It runs only for small
  supports (≤ 2^18 states). With a 16-bit output the misrounding probability is exact. FP4
  units are exact per unit: the 16 products of a unit are summed exactly by lattice
  convolution (`schema::unit_law`), then enter the step as one term.

### 5.2 Tier 1: residue-exact Markov chain

`tier1.rs` (the phase-1 engine, `markov.rs`, stays behind `PTXV_ERRDIST_T1_OLD`). The
state, per partial-sum cell (sign, binade) and per value of the accumulator's 2 lowest
bits at its ulp (K_BITS = 2), is the law of the error accumulated so far (`DHist`: dense
±1024 bins of 1/64 ulp, log-spaced far bins, a tail). Per step:
- **Coarse layer.** The exact law of the step's partial sums by FFT, weighting the cells
  (emax, result binade).
- **Fine layer** (`steptrans`). For each (emax, result binade, accumulator input), a 2D FFT
  over (the terms' truncated sum mod m = r·4, τ in bins of Q / tsc(r)) gives the joint law
  of the RZ residue and the dropped parts, exactly for the terms. The accumulator enters as
  zero, as a lattice (its bits known from the previous step's grid) or as a fine residue
  law whose upper bits are conditioned on the tracked low bits (`upper_given_low`). The
  output is a transition table t[l][l'] of (post, pre) local laws.
- **Propagation.** Each (cell, l) law is rescaled to the result cell and convolved with
  its transition (direct or Fourier by support size); contributions below 1e-13 go to the
  tail. The final cells' masses come from inverting y's characteristic function (`cf.rs`).

Applicability: the law's mass outside the model (tail + pruned element mass) ≤ max(5% of
P(differ), 1e-7), and the mode is one the matrix validates (not a pair of block-scaled
kernels, §6). Runtime: 5-60 s (bf16, f16, MX recipes), 100-750 s (MX random codes,
pairs); `PTXV_ERRDIST_THREADS` sets the worker count (default min(cores, 48)).

### 5.3 Tier 2: the DAG CLT

`tier2.rs` `dag` / `dag_estimate`. Any output that is a signed sum of products of inputs
without block scales: adds (reductions, split-K, atomics, scans), FMAs, MMA steps and TF32
splits (§4). Per output (up to 64 sampled outputs, averaged; scans differ by depth):
- Every node value is normal (independent leaves, exact moments) and has a **lattice law**
  P(value is a multiple of 2^L): leaves from their PMFs (f32 inputs: random low bits),
  min over the operands of an add, sum of exponents for products, raised to the ulp after
  the node's own rounding.
- A node's **rounding error** in each binade of its value is a residue uniform on the
  lattice grid below the ulp (grid moments with exact zero and ties to even; RZ for MMA
  steps). The grid is the node's lattice (a residue on 2^k points for a lattice k bits below
  the ulp), not the finest grid Q allows. MMA steps use the model's truncation at
  Q = 2^(emax − F); emax's law comes from the terms' exponent fields and the accumulator's
  binade given the result's binade (the accumulator is a sub-sum of the result).
- **Correlated rounding means.** A round toward zero has mean −sign(x) ulp(x) (1 − 2^-k) / 2:
  a function of the node value, E[e | x] ≈ β x (β by regression over x's binades). Partial
  sums share their terms, so these means are correlated along the chain: with
  B = Σ_j β_j x_j = Σ_g c_g t_g over the product terms (c_g sums β over the nodes containing
  t_g), Var(B | y) = Σ v_g c_g² − (Σ v_g c_g y_g)² / Σ v_g y_g² replaces the per-node
  Σ β_j² Var(x_j | y). Pairs use c_C − c_R (forward against reversed K: c ∝ K + 1 − 2k).
  This term grows like K³ against K for the independent part, and dominates large-K MMA
  pairs (`bias_extra`).
- **TF32 splits**: e = h_a d_b + d_a h_b − l_a l_b for a big/small split (l = x − hi(x),
  d = small's rounding: half away for CUTLASS's round_half_ulp_truncate, truncation when the
  MMA drops the bits), from exact bilinear moments of (x, l, d) per leaf law, including the
  part proportional to the product (its covariance with y: a mean β·y, 1.3 ulp for 3xTF32).
- **Given y**, each node is a sub-sum of y (x | y normal by regression); the 16 largest node
  errors are convolved as discrete laws, the rest summed as a zero-inflated normal.
  P(differ) integrates over y's binade and the exact y's position within its ulp (its own
  lattice), against the output's final rounding (RN, or RZ for an MMA root) or the 16-bit
  flip. Pairs: identical sub-DAGs (equal canonical hashes) carry the same error in both
  kernels and drop out of D.

Applicability: add / FMA / TF32 DAGs (validated, §7.1). Plain MMA chains over narrow inputs
are Tier 1's domain (the DAG CLT is 0.8-1.4x there) and report not applicable.
Block-scaled chains use the chain CLT (`tier2::single` / `pair`: per-step residue-exact
local moments from `tier1::local_moments`, zero-inflated normal error), applicable only when
y's position in its ulp is near uniform (TV ≤ 0.05), which MX recipes never are, and the
block-scaled chain tier (§5.3.1). Runtime: 0.03-7 s.

#### 5.3.1 Block-scaled chains

`tier2bs.rs` `ChainPre` / `estimate`; single MX kernels (MXF8, MXF4). Each tensor-core step
is one alignment to the step's largest exponent, a truncation of every term and of the
accumulator at Q = 2^(emax − F), an exact sum and one round toward zero to the result's
f32 grid (N-MMA-BS).
- **Lattice class.** Narrow-format terms sit on coarse grids: an output whose terms all
  have their lowest set bit at or above L is exact at every step whose ulp is at most 2^L.
  The tier conditions on y and on y's lattice class (the lowest set bit over all terms,
  against y's ulp). The class fixes every step's lattice law (the accumulator's lattice is
  the larger of the previous terms' lattice and the previous result's ulp) and y's position
  within its ulp (on the grid, or an odd multiple of 2^-k ulp).
- **Step error.** The final round toward zero is a residue uniform on the 2^k grid below
  the result's ulp (k = ulp − max(Q, lattice); none when k ≤ 0). The truncations have
  exact moments from the term atoms (block scale times element products, under the spec
  and the recipe).
- **Given y and the class**, partial sums are normal sub-sums of y; the largest step
  residues stay discrete, the rest and the truncations are a zero-inflated normal. The PMF
  over output ulps follows from y's position, the error law and the final rounding.
- **Gate:** y's excess kurtosis ≤ 1 (§3.2). The estimate is still computed and reported
  when refused.
- **Pairs: refused, not run.** Combining two chains' errors as independent given y and its
  class came out 0.77x of Tier 3 on the MXF8 reverse-K pair (normal), at 1,000 s.
- **NVFP4** with a global scale has 4e7 term atoms per step: tiers 1-2 are not run.

MXF8 single kernel (`r3_cute_mxf8`), against a 20,000-input Tier 3:

| spec | y excess kurtosis | gate | tier2bs / Tier 3 | tier2bs seconds |
|---|---|---|---|---|
| normal | 0.02 | pass | 0.966 | 13 |
| laplace | 0.13 | pass | 1.009 | 38 |
| lognormal | 15.5 | refused | 0.955 | 7 |
| Student t | 1.6e5 | refused | 1.12 | 110 |
| outliers | 84 | refused | 2.54 | 161 |
| codes | 77 | refused | 1.07 | 247 |
| ints_scaled | 34 | refused | 2.35 | 83 |

MXF4: codes 1.95x (kurtosis 36, refused); Student t: Tier 3 sees P(differ) = 0 at 20,000
inputs and tier2bs gives 4e-12 (kurtosis 4e13, refused). The other MXF4 rows are Tier 0.
Every refusal above is the gate's.

### 5.4 Tier 3: Monte Carlo through the DAG

`mc.rs`. Inputs are sampled from the spec (`spec::sample`, stream (seed, t, buffer)), and
the DAG is evaluated for both kernels. The outputs are compared with each other and with
the exact value from the expansion. This is bit-exact per sample given the models. Simulator
and GB300 runs check that: 0 DAG mismatches on every run in §7. Out-of-domain events
(overflow, NaN / Inf, zero or NaN scales) are counted per input and reported, never
dropped.

Confidence intervals treat inputs as the independent unit (N-ERRDIST-MC):
- **P(differ):** the wider of two intervals. One is Clopper-Pearson over outputs. The other
  is a batch-means interval over inputs (Student t over each input's fraction of outputs that
  differ). Outputs of one input share its elements: a GEMM row, an MX block's scale. Counted
  as independent, they gave intervals about 3x too narrow on MXF8 (seed-to-seed spread
  ±0.0002 against ±0.00007).
- **The mean:** batch means per input.

The ulp-scale mean, std and mean |.| leave out outputs whose exact value is 0 or subnormal in
the output format, where e / ulp(y) is unbounded. A note gives the count; P(differ) and the
PMF keep those outputs.

## 6. Pairs

Both kernels end in the same final rounding of y + δ_K to the f32 grid of y's binade, with
δ_K the deviation before that rounding. With U = frac(y/ulp):

    D/ulp = floor(U + c) − floor(U + r),   c = δ_C/ulp, r = δ_R/ulp.

- Tier 1 conditions on U (`combine::pair_cell_cond`): per U bin, each kernel's post-rounding
  error is (m − U)·ulp, and m_C, m_R are independent given U (N-ERRDIST-PAIR). Far bins use
  the independent floor + Bernoulli rule.
- Tier 2 integrates over the exact y's position with both kernels' error laws given y;
  shared sub-DAGs drop out.
- With 16-bit outputs, a flip has probability E[min(1, |D|/ulp16)] (N-ERRDIST-FLIP).
- Tier 3 compares the kernels' bits directly.

**Known bias.** Block-scaled kernels that use the same K partition in another order
(CuTeDSL vs Triton-rev) truncate the same terms in every step, so δ_C and δ_R are
positively correlated. Tier 1 overestimates P(D ≠ 0) by 1.2-1.3x there (§7.1) and reports
itself not applicable; a joint chain over both kernels is the open item. The bf16 pairs
(same structure, fine residues) are within 2-8%.

## 7. Validation

### 7.1 The matrix (phase 2, 2026-11-05)

`tools/errdist_matrix.py` runs precision × input family × mode through the CLI (Tier 3 with
2000 inputs as ground truth, all analytic tiers alongside) and writes `matrix.md` /
`matrix.json`. A tier passes a row when |P_tier − P_3| ≤ max(Tier 3's 99.9% CI half-width,
5% of P_3); ✗ marks a miss, (x.xx) is P_tier / P_3. "before" is tag base7 (phase 1), "after"
this version; "best" is the report's headline tier after (0 exact, 1, 2, 3 Monte Carlo).
Rows: bf16 = `c_mm_bf16_nnt` (single) and `c_equiv_mm_bf16` cand 1 (cuBLAS vs Triton-rev);
f16 = `r2_cute_gemm`; mxf8 / mxf4 / nvf4 = `r3_cute_*` and the `c_equiv_*` pairs; tf32 =
`c_tf32_mm_1x` / `_3x` and `c_equiv_tf32`; f32-sum = `r4_sum_torch` and `c_equiv_sum`
(row-reduce, split-K, atomic candidates); f32-scan = `c_cumsum`; f32-fma = `c_mm_f32_gemmsn`
(its inputs are untracked: constant outputs, Tier 3 skips them). Families: the
`fixtures/errdist/*.json` specs (`-mx` / `-nv`: the MX / NVFP4 quantization recipe).

| row | T3 | T1 before | T1 after | T2 before | T2 after | best after |
|---|---|---|---|---|---|---|
| bf16/single/normal | 2.55e-05 | 2.81e-05 (1.10) | 2.65e-05 (1.04) | 0.000114 (4.48 ✗) | 2.23e-05 (0.87) | 1 |
| bf16/single/uniform12 | 0 | 0 | 0 | - | - | 0 |
| bf16/single/student_t | 3.67e-05 | 5.19e-05 (1.41 ✗) | 3.3e-05 (0.90) | 0.000115 (3.13 ✗) | 3.62e-05 (0.98) | 1 |
| bf16/single/laplace | 4.21e-05 | 3.72e-05 (0.88 ✗) | 3.84e-05 (0.91) | 0.000115 (2.73 ✗) | 3.63e-05 (0.86 ✗) | 1 |
| bf16/single/lognormal | 1.27e-05 | 1.51e-05 (1.19) | 1.16e-05 (0.91) | 2.72e-05 (2.14 ✗) | 1.28e-05 (1.01) | 1 |
| bf16/single/outliers | 2.99e-05 | 6.35e-05 (2.12 ✗) | 3.27e-05 (1.09) | 0.000127 (4.24 ✗) | 8.51e-05 (2.85 ✗) | 1 |
| bf16/single/integers | 0 | 0 | 0 | - | - | 0 |
| bf16/single/ints_scaled | 3.63e-05 | 3.25e-05 (0.90) | 3.35e-05 (0.92) | 0.000115 (3.17 ✗) | 4.89e-05 (1.35 ✗) | 1 |
| bf16/pair-rev/normal | 3.65e-05 | 4.28e-05 (1.17 ✗) | 3.93e-05 (1.08) | 0.000125 (3.41 ✗) | 3.23e-05 (0.89) | 1 |
| bf16/pair-rev/student_t | 5.11e-05 | 8.27e-05 (1.62 ✗) | 4.73e-05 (0.93) | 0.000126 (2.46 ✗) | 5.02e-05 (0.98) | 1 |
| bf16/pair-rev/uniform12 | 0 | 0 | 0 | - | - | 0 |
| bf16/pair-rev/ints_scaled | 4.77e-05 | 4.87e-05 (1.02) | 4.84e-05 (1.02) | 0.000126 (2.64 ✗) | 6.54e-05 (1.37 ✗) | 1 |
| bf16/pair-fwd/normal | 0 | - | - | - | - | 0 |
| f16/single/normal | 0.741 | 0.727 (0.98) | 0.75 (1.01) | 0.775 (1.05) | 0.858 (1.16 ✗) | 1 |
| f16/single/student_t | 0.821 | 0.781 (0.95) | 0.825 (1.00) | 0.777 (0.95 ✗) | 0.906 (1.10 ✗) | 1 |
| f16/single/uniform12 | 0 | 0 | 0 | - | - | 0 |
| mxf8/single/normal-mx | 0.0449 | 0.0333 (0.74 ✗) | 0.0477 (1.06 ✗) | 0.774 (17.22 ✗) | 0.515 (11.47 ✗) | 1 |
| mxf8/single/student_t-mx | 0.0883 | 0.261 (2.96 ✗) | 0.0932 (1.05 ✗) | 0.774 (8.76 ✗) | 0.533 (6.03 ✗) | 3 |
| mxf8/single/laplace-mx | 0.109 | 0.0782 (0.71 ✗) | 0.116 (1.06 ✗) | 0.774 (7.07 ✗) | 0.537 (4.90 ✗) | 1 |
| mxf8/single/lognormal-mx | 0.00507 | 0.00088 (0.17 ✗) | 0.00511 (1.01) | 0.999 (197.12 ✗) | 0.503 (99.33 ✗) | 1 |
| mxf8/single/outliers-mx | 0.311 | 0.277 (0.89 ✗) | 0.316 (1.02) | 0.775 (2.49 ✗) | 0.704 (2.26 ✗) | 1 |
| mxf8/single/codes | 0.846 | 0.824 (0.97) | - | 0.807 (0.95) | - | 3 |
| mxf8/single/ints | 0 | 0 | 0 | - | - | 0 |
| mxf8/single/ints_scaled | 0.292 | 0.258 (0.88 ✗) | 0.294 (1.01) | 0.787 (2.70 ✗) | 0.699 (2.39 ✗) | 1 |
| mxf8/pair-rev/normal-mx | 0.0341 | 0.0843 (2.48 ✗) | 0.0447 (1.31 ✗) | 0.821 (24.11 ✗) | 0.095 (2.79 ✗) | 3 |
| mxf8/pair-rev/codes | 0.84 | 0.817 (0.97) | - | 0.844 (1.00) | - | 3 |
| mxf8/pair-rev/ints_scaled | 0.244 | 0.376 (1.54 ✗) | 0.287 (1.18 ✗) | 0.823 (3.38 ✗) | 0.724 (2.97 ✗) | 3 |
| mxf8/pair-rev/student_t-mx | 0.0653 | 0.473 (7.24 ✗) | 0.086 (1.32 ✗) | 0.808 (12.37 ✗) | 0.198 (3.03 ✗) | 3 |
| mxf8/pair-cublas/normal-mx | 0 | - | - | - | - | 0 |
| mxf4/single/normal-mx | 0 | 0 | 0 | - | - | 0 |
| mxf4/single/codes | 0.335 | 0.271 (0.81 ✗) | 0.342 (1.02) | 0.711 (2.13 ✗) | 0.647 (1.93 ✗) | 1 |
| mxf4/single/student_t-mx | 0 | 0 | 0 | 0.522 | 0.5 | 3 |
| mxf4/single/laplace-mx | 0 | 0 | 0 | 0.5 | - | 0 |
| mxf4/pair-rev/codes | 0.159 | 0.286 (1.79 ✗) | 0.211 (1.33 ✗) | 0.727 (4.56 ✗) | 0.636 (3.98 ✗) | 3 |
| mxf4/pair-rev/student_t-mx | 0 | 0 | 0 | 0.196 | 0 | 3 |
| nvf4/single/normal-nv | 0 | - | - | - | - | 3 |
| nvf4/single/student_t-nv | 1.53e-07 | - | - | - | - | 3 |
| nvf4/single/codes | 0.788 | - | - | - | - | 3 |
| nvf4/pair-dq/normal-nv | 0 | - | - | - | - | 3 |
| nvf4/pair-dq/codes | 0.559 | - | - | - | - | 3 |
| nvf4/pair-cublas/normal-nv | 0 | - | - | - | - | 0 |
| tf32/single/normal | 1 | error | - | error | 1 (1.00) | 2 |
| 3xtf32/single/normal | 0.97 | error | - | error | 0.977 (1.01) | 2 |
| tf32/single/student_t | 1 | error | - | error | 1 (1.00) | 2 |
| 3xtf32/single/student_t | 0.968 | error | - | error | 0.977 (1.01) | 2 |
| tf32/single/uniform12 | 1 | error | - | error | 1 (1.00) | 2 |
| 3xtf32/single/uniform12 | 1 | error | - | error | 1 (1.00) | 2 |
| 3xtf32/pair-rev/normal | 0.969 | error | - | error | 0.97 (1.00) | 2 |
| 3xtf32/pair-3x-vs-f32/normal | 0.967 | error | - | error | 0.97 (1.00) | 2 |
| f32-sum/single/normal | 0.673 | - | - | - | 0.678 (1.01) | 2 |
| f32-fma/single/normal | 0 | - | - | - | - | 3 |
| f32-sum/single/uniform12 | 0.284 | - | - | - | 0.282 (0.99) | 2 |
| f32-fma/single/uniform12 | 0 | - | - | - | - | 3 |
| f32-sum/single/student_t | 0.668 | - | - | - | 0.68 (1.02) | 2 |
| f32-fma/single/student_t | 0 | - | - | - | - | 3 |
| f32-sum/pair-rowreduce/normal | 0.705 | - | - | - | 0.744 (1.05 ✗) | 2 |
| f32-sum/pair-splitk/normal | 0.72 | - | - | - | 0.751 (1.04) | 2 |
| f32-sum/pair-atomic/normal | 0.719 | error | - | error | 0.756 (1.05 ✗) | 2 |
| f32-scan/single/normal | 0.618 | - | - | - | 0.594 (0.96) | 2 |

Before, the TF32 rows errored ("Tf32 node not modelled") and the f32 sums had no exact
value (Tier 3 skipped them); both are now Tier 3 rows (N-ERRDIST-DAG) with Tier 2 as the
analytic tier.

**Where `best` is analytic (1 or 2) it passes, except four borderline rows**: MXF8 normal
and Laplace recipes (Tier 1 1.06x; the accumulator's bits above the two tracked ones
cluster), and the f32 row-reduce / atomic pairs (Tier 2 1.05x, at the 5% line; per-kernel
variances match Tier 3 within 4-12% and Tier 3's pair correlation is 0.003-0.09).
**Where the matrix shows a bias, the tier's check fails and `best` falls back**: Tier 1 on
block-scaled pairs (1.2-1.3x) and on MX Student-t (15% of its law beyond the modeled
range); Tier 2's DAG CLT on plain MMA chains (bf16 outliers / scaled integers 1.35-2.85x,
f16 1.1-1.16x); the block-scaled chain CLT (2-11x; y's residue is far from uniform).
MXF8 random codes: Tier 1 did not finish within the 60-minute row timeout (Tier 3: 6 s).
NVFP4: the analytic tiers skip steps above 5e6 term atoms.

**GB300** (GPU 3, 300 inputs, `--gpu-trials 300`), the new Tier 3 rows: 0 DAG mismatches
and P(differ) GPU / Tier 3 = 0.96996 / 0.97008 (3xTF32), 0.99998 / 0.99999 (1xTF32),
0.6825 / 0.6735 (f32 sum, 1200 outputs), 0.6168 / 0.6181 (cumsum), 0.96837 / 0.96865
(3xTF32 pair), 3.09e-5 / 3.65e-5 (bf16 pair, 1.2M outputs). The atomic pair has 362 of 1200
outputs mismatching the DAG on the GPU (another atomic order, N-ERRDIST-DAG); P(differ)
0.742 vs Tier 3's 0.719.

**Timings** (144-core host, `PTXV_ERRDIST_THREADS=24`): Tier 3 0.1-6 s; Tier 2 0.02-7 s
(DAG CLT), 4-160 s (block-scaled chain CLT); Tier 1 2.5-55 s (bf16, f16), 24-200 s (MX
single), 60-680 s (MX pairs, MX Student-t), > 60 min (MXF8 random codes). Phase 1's Tier 1
took 30-160 s on bf16 / MX normal recipes.

**Fast subset**: `tools/errdist_matrix.py --bin … --out … --fast` (5 rows, one per analytic
domain, about 8 minutes; exit 1 when an applicable best tier misses).

### 7.2 Phase-1 runs (GB300, 2026-11-04)

The Tier 1 / Tier 2 columns of this section are phase 1's; §7.1 supersedes them. The Tier 3
and GB300 columns stand.


Every run below:
- uses the fixtures in `fixtures/errdist/`;
- uses candidate 1 (the reversed-K Triton kernel) for the `c_equiv_*_triton` and
  `c_equiv_mm_bf16` pairs;
- has 0 DAG mismatches against the simulator and against GB300 (GPU 3, 300-1000 inputs), so
  Tier 3 is the kernels' exact behavior on the sampled inputs.

P(differ) is P(C ≠ R) bitwise for a pair. For one kernel it is P(C ≠ RN(exact)).

| target | spec | T0 | T1 | T2 | T3 (DAG) | GB300 observed [95%] |
|---|---|---|---|---|---|---|
| bf16 cuBLAS vs Triton-rev | normal | — | 4.28e-5 | 1.25e-4 | 3.63e-5 | 4.00e-5 [3.41e-5, 4.67e-5] (1000 inputs, 4.1M outputs) |
| bf16 cuBLAS vs Triton (same order) | normal | 0 (identical programs) | — | — | 0 | — |
| bf16 pair | uniform[1,2), integers | 0 (certificate) | 0 | — | 0 | — |
| MXF8 CuTeDSL vs Triton-rev | random codes | — | (slow, skipped) | 0.844 | 0.840 | 0.840 ± 0.0003 (300 inputs) |
| MXF8 CuTeDSL vs Triton-rev | normal + MX recipe | — | 0.087 | 0.809 | 0.0340 | 0.0339 ± 0.0002 |
| MXF8 CuTeDSL vs Triton (same order) | normal + MX recipe | 0 (identical programs) | — | — | 0 | — |
| MXF4 CuTeDSL vs Triton-rev | random codes | — | 0.281 | 0.727 | 0.160 | 0.160 ± 0.0005 |
| MXF4 CuTeDSL vs Triton-rev | normal + MX recipe | 0 (certificate) | 0 | — | 0 | — |
| single r3_cute_mxf8 | normal + MX recipe | — | 0.0317 | 0.774 | 0.0449 | 0.0450 ± 0.0003 |
| single r3_cute_mxf8 | random codes | — | 0.823 | 0.807 | 0.846 | — (sim 2 inputs agree) |
| single r3_cute_mxf8 | integers, unit scales | 0 (certificate) | 0 | (0.774 before tier 0 supersedes it) | 0 | — |
| single r3_cute_mxf8 | integers, scales 2^-8..2^7 | — | 0.258 | 0.787 | 0.292 | — |
| NVFP4 CuTeDSL vs dequant-bf16 Triton | random codes (ue4m3 1..126) | — | (skipped) | 0.842 | 0.560 | 0.558 ± 0.0007 |
| NVFP4 CuTeDSL vs dequant-bf16 Triton | normal + NVFP4 recipe | — | (skipped) | 6.6e-12 | 0 | 0 (300 inputs) |
| single r3_cute_mxf4 | random codes | — | 0.271 | 0.711 | 0.335 | — (sim 0.331, 2 inputs) |
| single r3_cute_nvf4 | random codes | — | (too large) | (too large) | 0.789 | — (sim 0.791, 2 inputs) |

Tier 1 with the uniform-residue heuristic (`--uniform`): 0.899 on the MXF8 normal recipe
(truth 0.045), 0.833 on scaled integers (truth 0.292).

**Target 1, the bf16 flip rate.** Tier 1 predicts 4.28e-5, against GB300's 4.00e-5. That is
within the GPU interval and 1.18x Tier 3's 3.63e-5 (whose own interval is
[3.35e-5, 3.94e-5]). The flips come from the cancellation cells. There |y| ≪ the partial
sums, and D in y's ulps is large: E|D| doubles per binade below |y| ≈ 1. Before the final
characteristic-function cell masses and the log-spaced deviation bins, Tier 1 gave 1.99e-5
(tails clipped at ±64 ulps) and then 5.86e-5 (cell masses near 0 too high by 1.5x). The
remaining excess is the pair-independence bias. Tier 2 (3.1x) has the same bias, plus
uniform residues and normal partial sums.

**Target 2, why MXF8 pairs use about 5% of the bound.** On random codes, Tier 3 gives
D / (u S) with std 1.82, q0.999 7.1, and a maximum of 16.6 over 6.5M outputs. The bound is
306, so the maximum is 5.4% of it and the std is 0.6%. Tier 2 says why:
- local errors have random signs across 8 steps and about 33 truncations per step;
- each truncation loses on average half its worst case;
- Q = 2^(emax−25) is set by the step's largest value, not by S = Σ|terms|, so each step's
  error is a small fraction of u·S.

Tier 3 measures σ = 1.82 u·S. Tier 2 gives 5.2: its independence assumption and its emax
approximation both inflate it. The bound instead adds 8 steps × 33 worst cases with aligned
signs. The observed tail is somewhat heavier than normal (q0.999 = 3.9σ), and the largest
of 6.5M outputs sits at 9σ = 16.6 = 5.4% of 306. So roughly 5% is what a near-normal sum
of random-signed truncations reaches at 10^6-10^7 samples. Tier 2's P(differ) 0.844
matches GB300's 0.840.

**Target 3, FP4.**
- MXF4 with the normal recipe is certified exact by Tier 0. Every unit is an exact sum on
  the 2^-2·scale lattice, and the partial sums fit 24 bits, so C = R = exact. Tier 3 and
  Tier 1 confirm it.
- MXF4 with random codes: Tier 3 0.160 = GB300 0.160. Tier 1 gives 0.281 (pair bias).
- NVFP4 against the dequantizing bf16 Triton kernel, random codes: Tier 3 0.560, GB300 0.558.
- NVFP4 with the normal recipe: Tier 3 and GB300 give 0. Tier 2, with lattice-aware residues,
  gives 7e-12. The certificate fails only on its worst-case support bound.
- Single r3_cute_nvf4 with random ue4m3 codes has 1.7e8 term atoms per step (16k scale
  pairs × unit laws). The analytic tiers skip it above 5e6 atoms. Building the schema alone
  takes 24 GB.

**Target 4, single kernel against exact; where the uniform heuristic is wrong.**
- Normal + MX recipe: P(C ≠ RN(exact)) = 0.045. The residues are far from uniform: max |φ_k|
  ≈ 0.99 and the TV bound is 0.72-0.80 in every step (report `residue`). Block-scaled e4m3
  products are 8-bit significands on a lattice that is coarse relative to ulp(S). The
  uniform heuristic predicts 0.90 (Tier 1 `--uniform`) or 0.77 (Tier 2), 17-20x the truth.
  Tier 1 with the exact residues gives 0.032 (0.7x).
- Integers with unit scales: the certificate proves exactness. The uniform heuristic would
  predict 0.77.
- Integers with scales 2^-8..2^7: the truth is 0.292, Tier 1 gives 0.258, and the uniform
  heuristic 0.83.
- Small K: the first step of the MX chain (K = 32) has P(e ≠ 0) = 0.002 (Tier 3 `--explore`),
  where the uniform model has ≈ 1. Per-step model P(e ≠ 0) on the normal recipe is
  0.003, 0.006, 0.009, 0.012, 0.016, 0.018, 0.022, 0.023, against Tier 3's 0.002, 0.007,
  0.008, 0.009, 0.016, 0.015, 0.015, 0.017. The model is 10-40% high per step.

**Where the analytic tiers are off.**
- Pairs whose kernels share the K partition (all the rev pairs here): N-ERRDIST-PAIR's
  independence overestimates P(differ) by 1.1x (bf16 flips), 1.8x (MXF4 codes) and 2.6x
  (MXF8 normal).
- Single kernels: Tier 1 is 0.7x (MXF8 normal), 0.97x (MXF8 codes) and 0.88x (scaled
  integers) of the truth.
- Tier 2 is right only where residues are close to uniform (MXF8 codes, 1.005x). It reads
  2-20x high elsewhere: MXF4 codes 2.1x, NVFP4 dq codes 1.5x, MXF8 normal 17-24x.
- Tier 1's ulp-scale mean and std cover the central 1 − 2e-6 of its law. Cancellation cells
  put |e| / ulp(y) at 2^30 or more with tiny probability. Tier 3's ulp std is also dominated
  by its largest samples (400-5000), so compare quantiles and P(differ), not stds.
- Runtimes:
  - Tier 3: seconds;
  - Tier 2: seconds (bf16, MX), up to 100 s (NVFP4 dq);
  - Tier 1: 30-160 s (bf16, MX normal recipe), 80 s (MXF4 codes), 15-25 min (MXF8 random
    codes; `--no-tier1` skips it).

**Tests** (`cargo test --release`):
- `tier0::enumeration_matches_brute_force`: the exact signed law against brute force over
  every term tuple, including a signed mixture;
- `tier0::certificate`;
- `fine` (residue law vs enumeration, the TV bound on known laws);
- `cf::cell_masses_of_a_uniform_sum` (f(0) to 2e-4 with the Edgeworth term);
- `laws` (code PMFs and MX block laws vs sampling, cloud moments);
- `chain` (fast steps = the calibrated models, exact expansion sums);
- `stats` (KS / AD);
- `spec`;
- `tier1` (per-step P(δ ≠ 0), low-bit laws and P(differ) against simulation of e4m3 /
  e5m2 / bf16 chains: normal, Laplace, Student-t, one step to six; one step from a known
  accumulator);
- `tier2` (`reductions_match_simulation`: sequential and pairwise f32 sums of normal and
  same-sign uniform inputs; grid residue moments; the zero-inflated expectations).

The prediction-vs-simulator check is part of every CLI run (`validation`).

### 7.3 The full matrix under the default policy (2026-11-06)

Every kernel family is in this matrix: 162 rows.
- the chain rows of §7.1 (bf16, f16, MX / NV block-scaled, TF32, f32 sums, FMA chains, scans,
  single and pairs);
- the non-chain rows of §9.7 (softmax, norms, attention, elementwise, log-sum-exp, Kahan)
  under normal, laplace and Student t(3) inputs;
- inputs that carry an error (§2.1).

Each row was run three ways:
- **default:** `ptxv errdist ... --sim-trials 0`, the policy of §3.2 including the growing
  cross-check. This gives the headline tier, Tier 2's verdict, and why it refused.
- **truth:** Tier 3 at 20,000 inputs on another seed (independent of the default run's
  Tier 3 inputs).
- **GB300:** 50 inputs on GPUs 0/1/3, compared with a 2,000-input Tier 3.

P(differ) intervals take inputs as the independent unit (§5.4, N-ERRDIST-MC): the Tier 3
intervals of the cross-check, the truth's and the GB300 column are the wider of
Clopper-Pearson over outputs and batch means over inputs.

| | rows |
|---|---|
| headline right: within max(5%, truth's 95% half-width) of the truth, or Tier 3 itself | 155 of 155 |
| headline wrong | 0 |
| Tier 2 applicable and right ("holds") | 53 |
| Tier 2 refused, and it was off ("refused (correctly)") | 88 |
| Tier 2 refused, though it would have been within tolerance | 6 |
| no Tier 2 (Tier 0 answers, or none for the program) | 15 |
| GB300 P(differ) within Tier 3's interval | 161 of 161 |

The input-error rows (7) are judged on mean |error| within 10%, not on P(differ); all are
right.

Changes against the run with output-level intervals (same rows and seeds; that run drew
its inputs, truth included, with the sampler before patch 07, so on other inputs):
- **Three more Tier 2 headlines:** torch softmax and torch RMSNorm under normal inputs
  (0.955x, 0.895x of the truth; the RMSNorm truth is 1.07e-5 with a 20% half-width), and
  naive log-sum-exp (0.956x). With the wider Tier 3 interval the cross-check accepts them.
- **Tier 2 values that moved with the sample**, not with the interval: the non-chain
  tiers' probes come from the run's inputs, and at P(differ) ~1e-5 Tier 2 moves with them.
  RMSNorm under laplace (torch, triton) went 0.83x / 0.81x to 0.71x / 0.66x and triton
  layer norm under t(3) 1.10x to 0.54x; all three stay refused.
- **Verdicts at the 5% edge:** the softmax pair (1.06x) is now "refused (correctly)"; the
  split-K and atomic f32 sum pairs (1.05x) are now "would have held". All three were
  refused in both runs.

Notes:
- **Before the growing cross-check, 7 headlines were wrong** with a 300-input check:
  - bf16 GEMM, normal and laplace (Tier 2 at 0.82x and 0.85x);
  - layer norm, torch and triton (0.37-0.53x);
  - packed bf16 gelu (4x);
  - a row-reduce sum pair (1.05x, against a Tier 3 that landed 3% high).
- **The cost is a Tier 3 of up to 76,800 inputs, 1-7 s here.** The 6 "would have held"
  refusals are cross-check near-misses (the f32 sum pairs at 1.05x) or structural
  refusals where Tier 2 happened to be right (the bf16 t GEMM at 1.02x, an MXF8 pair under
  codes, an MXF4 pair with P(differ) = 0, a softmax whose input error dominates).
- **GB300.** P(differ) agrees on every row with a GPU run. Three rows flag another test:
  - the attention core, single and pair: only the 95% test of the u-scale mean (observed
    against Tier 3's mean at 2.3x and 1.4x the 95% half-width; KS p = 1.0 on both). Both
    agreed in the previous run, on other inputs;
  - the atomic split-K pair. Its GPU outputs differ bitwise from the DAG's on 52 of 200
    (another order of the atomic adds, N-ERRDIST-DAG), with KS p = 0.98 against Tier 3.
- **Non-finite errors.** Under t(3) the gelu kernels give non-finite outputs on a few
  inputs: the u-scale mean is infinite on all five gelu rows, and the two pair rows have
  one NaN error in the GB300-against-Tier 3 samples. The distribution tests now leave NaN errors out and
  the mean is not compared when it is not finite. Before, the two-sample tests' merge did
  not advance past a NaN (the gelu pair rows never finished), and the three single rows
  flagged the mean.
- **Not run on a GPU (1 row):** the cuBLAS MXF8 pair (the GPU worker could not take that
  case).
- Scripts and raw reports (the run directory, `fullmatrix/`):
  - `full_matrix.py` (default and GB300 runs);
  - `truth2.py` (the 20,000-input truth);
  - `table.py` (this table).

| row | headline tier | headline vs truth | truth: Tier 3 P(differ) [95% CI] | Tier 2 | T2/T3 | Tier 2 verdict | refusal reason | GB300 P(differ), 50 inputs | GB300 vs Tier 3 |
|---|---|---|---|---|---|---|---|---|---|
| bf16/single/normal | 3 | ok | 2.719e-05 [2.61e-05, 2.83e-05] | 2.082e-05 | 0.766 | refused (correctly) | Tier 3 cross-check (7667 inputs) | 1.953e-05 [5.17e-07, 5e-05] | agree |
| bf16/single/uniform12 | 0 | ok | 0 [0, 4.5e-08] | - | - | no tier 2 |  | 0 [0, 1.8e-05] | agree |
| bf16/single/student_t | 3 | ok | 3.47e-05 [3.34e-05, 3.6e-05] | 3.525e-05 | 1.02 | refused (would have held) | 64 of 64 sampled outputs outside the DAG tier's  | 4.395e-05 [1.7e-05, 8.34e-05] | agree |
| bf16/single/laplace | 3 | ok | 4.016e-05 [3.88e-05, 4.16e-05] | 3.588e-05 | 0.893 | refused (correctly) | Tier 3 cross-check (6314 inputs) | 4.395e-05 [1.7e-05, 8.34e-05] | agree |
| bf16/single/lognormal | 3 | ok | 1.13e-05 [1.06e-05, 1.21e-05] | 1.008e-05 | 0.892 | refused (correctly) | 64 of 64 sampled outputs outside the DAG tier's  | 1.953e-05 [5.17e-07, 5e-05] | agree |
| bf16/single/outliers | 3 | ok | 3.326e-05 [3.2e-05, 3.45e-05] | 8.706e-05 | 2.62 | refused (correctly) | 64 of 64 sampled outputs outside the DAG tier's  | 2.441e-05 [3.39e-06, 5.7e-05] | agree |
| bf16/single/integers | 0 | ok | 0 [0, 4.5e-08] | - | - | no tier 2 |  | 0 [0, 1.8e-05] | agree |
| bf16/single/ints_scaled | 3 | ok | 3.608e-05 [3.48e-05, 3.74e-05] | 5.081e-05 | 1.41 | refused (correctly) | 64 of 64 sampled outputs outside the DAG tier's  | 4.395e-05 [1.36e-05, 8.34e-05] | agree |
| bf16/pair-rev/normal | 2 | ok | 3.829e-05 [3.7e-05, 3.97e-05] | 3.829e-05 | 1 | holds |  | 3.418e-05 [9.86e-06, 7.04e-05] | agree |
| bf16/pair-rev/student_t | 3 | ok | 4.769e-05 [4.62e-05, 4.92e-05] | 6.306e-05 | 1.32 | refused (correctly) | 64 of 64 sampled outputs outside the DAG tier's  | 1.953e-05 [5.17e-07, 5e-05] | agree |
| bf16/pair-rev/uniform12 | 0 | ok | 0 [0, 4.5e-08] | - | - | no tier 2 |  | 0 [0, 1.8e-05] | agree |
| bf16/pair-rev/ints_scaled | 3 | ok | 4.764e-05 [4.61e-05, 4.92e-05] | 8.583e-05 | 1.8 | refused (correctly) | 64 of 64 sampled outputs outside the DAG tier's  | 5.859e-05 [2.55e-05, 0.000102] | agree |
| bf16/pair-fwd/normal | 0 | ok | 0 [0, 4.5e-08] | - | - | no tier 2 |  | 0 [0, 1.8e-05] | agree |
| f16/single/normal | 3 | ok | 0.741 [0.741, 0.741] | 0.8396 | 1.13 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.7395 [0.738, 0.741] | agree |
| f16/single/student_t | 3 | ok | 0.8212 [0.821, 0.821] | 0.8901 | 1.08 | refused (correctly) | 64 of 64 sampled outputs outside the DAG tier's  | 0.8208 [0.82, 0.822] | agree |
| f16/single/uniform12 | 0 | ok | 0 [0, 2.81e-09] | - | - | no tier 2 |  | 0 [0, 1.13e-06] | agree |
| mxf8/single/normal-mx | 3 | ok | 0.04498 [0.0449, 0.0451] | 0.5153 | 11.5 | refused (correctly) | y's position in its f32 ulp | 0.04512 [0.0439, 0.0464] | agree |
| mxf8/single/student_t-mx | 3 | ok | 0.08835 [0.0883, 0.0885] | 0.5331 | 6.03 | refused (correctly) | y's position in its f32 ulp | 0.0896 [0.0877, 0.0915] | agree |
| mxf8/single/laplace-mx | 3 | ok | 0.1093 [0.109, 0.109] | 0.5369 | 4.91 | refused (correctly) | y's position in its f32 ulp | 0.1103 [0.108, 0.112] | agree |
| mxf8/single/lognormal-mx | 3 | ok | 0.00501 [0.005, 0.00502] | 0.5034 | 100 | refused (correctly) | y's position in its f32 ulp | 0.004929 [0.00465, 0.0052] | agree |
| mxf8/single/outliers-mx | 3 | ok | 0.3109 [0.311, 0.311] | 0.7036 | 2.26 | refused (correctly) | y's position in its f32 ulp | 0.3119 [0.309, 0.315] | agree |
| mxf8/single/codes | 3 | ok | 0.8462 [0.846, 0.846] | 0.7975 | 0.942 | refused (correctly) | block-scaled chain CLT | 0.8463 [0.844, 0.849] | agree |
| mxf8/single/ints | 0 | ok | 0 [0, 1.13e-08] | - | - | no tier 2 |  | 0 [0, 4.5e-06] | agree |
| mxf8/single/ints_scaled | 3 | ok | 0.292 [0.292, 0.292] | 0.699 | 2.39 | refused (correctly) | y's position in its f32 ulp | 0.2928 [0.29, 0.296] | agree |
| mxf8/pair-rev/normal-mx | 3 | ok | 0.03405 [0.034, 0.0341] | 0.09497 | 2.79 | refused (correctly) | y's position in its f32 ulp | 0.03462 [0.0336, 0.0356] | agree |
| mxf8/pair-rev/codes | 3 | ok | 0.8403 [0.84, 0.84] | 0.8348 | 0.993 | refused (would have held) | block-scaled chain CLT | 0.8409 [0.84, 0.842] | agree |
| mxf8/pair-rev/ints_scaled | 3 | ok | 0.2436 [0.243, 0.244] | 0.724 | 2.97 | refused (correctly) | y's position in its f32 ulp | 0.245 [0.242, 0.248] | agree |
| mxf8/pair-rev/student_t-mx | 3 | ok | 0.06531 [0.0652, 0.0654] | 0.1982 | 3.04 | refused (correctly) | y's position in its f32 ulp | 0.06565 [0.0641, 0.0672] | agree |
| mxf8/pair-cublas/normal-mx | 0 | ok | 0 [0, 1.13e-08] | - | - | no tier 2 |  | GPU validation not run: GPU worker for c | - |
| mxf4/single/normal-mx | 0 | ok | 0 [0, 1.13e-08] | - | - | no tier 2 |  | 0 [0, 4.5e-06] | agree |
| mxf4/single/codes | 3 | ok | 0.3346 [0.334, 0.335] | 0.6467 | 1.93 | refused (correctly) | y's position in its f32 ulp | 0.3364 [0.333, 0.34] | agree |
| mxf4/single/student_t-mx | 3 | ok | 0 [0, 1.13e-08] | 0.5 | - | refused (correctly) | y's position in its f32 ulp | 0 [0, 4.5e-06] | agree |
| mxf4/single/laplace-mx | 0 | ok | 0 [0, 1.13e-08] | - | - | no tier 2 |  | 0 [0, 4.5e-06] | agree |
| mxf4/pair-rev/codes | 3 | ok | 0.1595 [0.159, 0.16] | 0.6356 | 3.98 | refused (correctly) | y's position in its f32 ulp | 0.16 [0.157, 0.163] | agree |
| mxf4/pair-rev/student_t-mx | 3 | ok | 0 [0, 1.13e-08] | 0 | - | refused (would have held) | y's position in its f32 ulp | 0 [0, 4.5e-06] | agree |
| nvf4/single/normal-nv | 3 | ok | 0 [0, 1.13e-08] | - | - | no tier 2 |  | 0 [0, 4.5e-06] | agree |
| nvf4/single/student_t-nv | 3 | ok | 7.019e-08 [3.8e-08, 1.05e-07] | - | - | no tier 2 |  | 0 [0, 4.5e-06] | agree |
| nvf4/single/codes | 3 | ok | 0.7877 [0.788, 0.788] | - | - | no tier 2 |  | 0.789 [0.786, 0.792] | agree |
| nvf4/pair-dq/normal-nv | 3 | ok | 0 [0, 1.13e-08] | - | - | no tier 2 |  | 0 [0, 4.5e-06] | agree |
| nvf4/pair-dq/codes | 3 | ok | 0.5593 [0.559, 0.559] | - | - | no tier 2 |  | 0.5586 [0.556, 0.561] | agree |
| nvf4/pair-cublas/normal-nv | 0 | ok | 0 [0, 1.13e-08] | - | - | no tier 2 |  | 0 [0, 4.5e-06] | agree |
| tf32/single/normal | 2 | ok | 1 [1, 1] | 1 | 1 | holds |  | 1 [1, 1] | agree |
| 3xtf32/single/normal | 2 | ok | 0.9701 [0.97, 0.97] | 0.9754 | 1.01 | holds |  | 0.9709 [0.969, 0.973] | agree |
| tf32/single/student_t | 2 | ok | 1 [1, 1] | 1 | 1 | holds |  | 1 [1, 1] | agree |
| 3xtf32/single/student_t | 2 | ok | 0.968 [0.968, 0.968] | 0.9754 | 1.01 | holds |  | 0.9682 [0.967, 0.97] | agree |
| tf32/single/uniform12 | 2 | ok | 1 [1, 1] | 1 | 1 | holds |  | 1 [1, 1] | agree |
| 3xtf32/single/uniform12 | 2 | ok | 1 [1, 1] | 0.9998 | 1 | holds |  | 1 [1, 1] | agree |
| 3xtf32/pair-rev/normal | 2 | ok | 0.9687 [0.969, 0.969] | 0.9661 | 0.997 | holds |  | 0.9695 [0.968, 0.971] | agree |
| 3xtf32/pair-3x-vs-f32/normal | 2 | ok | 0.9669 [0.967, 0.967] | 0.9661 | 0.999 | holds |  | 0.9674 [0.966, 0.969] | agree |
| f32-sum/single/normal | 2 | ok | 0.6736 [0.67, 0.677] | 0.6621 | 0.983 | holds |  | 0.695 [0.626, 0.758] | agree |
| f32-fma/single/normal | 2 | ok | 0.7607 [0.761, 0.761] | 0.7565 | 0.995 | holds |  | 0.7624 [0.76, 0.765] | agree |
| f32-sum/single/uniform12 | 2 | ok | 0.2903 [0.287, 0.293] | 0.2568 | 0.885 | refused (correctly) | Tier 3 cross-check (6956 inputs) | 0.29 [0.227, 0.358] | agree |
| f32-fma/single/uniform12 | 2 | ok | 0.3852 [0.385, 0.385] | 0.3765 | 0.977 | holds |  | 0.3846 [0.382, 0.388] | agree |
| f32-sum/single/student_t | 2 | ok | 0.6648 [0.661, 0.668] | 0.6626 | 0.997 | holds |  | 0.66 [0.59, 0.725] | agree |
| f32-fma/single/student_t | 2 | ok | 0.7528 [0.753, 0.753] | 0.7493 | 0.995 | holds |  | 0.7542 [0.752, 0.757] | agree |
| f32-sum/pair-rowreduce/normal | 3 | ok | 0.7025 [0.699, 0.706] | 0.7389 | 1.05 | refused (correctly) | Tier 3 cross-check (1253 inputs) | 0.665 [0.588, 0.742] | agree |
| f32-sum/pair-splitk/normal | 3 | ok | 0.7132 [0.71, 0.716] | 0.7488 | 1.05 | refused (would have held) | Tier 3 cross-check (1212 inputs) | 0.74 [0.673, 0.799] | agree |
| f32-sum/pair-atomic/normal | 3 | ok | 0.7176 [0.714, 0.721] | 0.7511 | 1.05 | refused (would have held) | Tier 3 cross-check (1123 inputs) | 0.765 [0.7, 0.822] | DISAGREE |
| f32-scan/single/normal | 2 | ok | 0.6169 [0.616, 0.618] | 0.5989 | 0.971 | holds |  | 0.6357 [0.607, 0.664] | agree |
| nonlin/single/r4_softmax_torch/normal | 2 | ok | 0.6434 [0.643, 0.644] | 0.6142 | 0.955 | holds |  | 0.6441 [0.635, 0.654] | agree |
| nonlin/single/r4_softmax_triton/normal | 2 | ok | 0.7034 [0.703, 0.704] | 0.6968 | 0.991 | holds |  | 0.703 [0.694, 0.712] | agree |
| nonlin/single/c_triton_softmax_online/normal | 2 | ok | 0.7114 [0.711, 0.712] | 0.7026 | 0.988 | holds |  | 0.7143 [0.705, 0.724] | agree |
| nonlin/single/r4_rmsnorm_torch/normal | 2 | ok | 1.074e-05 [8.69e-06, 1.29e-05] | 9.613e-06 | 0.895 | holds |  | 0 [0, 0.000144] | agree |
| nonlin/single/r4_rmsnorm_triton/normal | 2 | ok | 1.025e-05 [8.24e-06, 1.24e-05] | 9.85e-06 | 0.961 | holds |  | 0 [0, 0.000144] | agree |
| nonlin/single/r4_layernorm_torch/normal | 3 | ok | 3.271e-05 [2.92e-05, 3.64e-05] | 1.944e-05 | 0.594 | refused (correctly) | Tier 3 cross-check (76800 inputs) | 3.906e-05 [0, 0.000218] | agree |
| nonlin/single/r4_layernorm_triton/normal | 3 | ok | 3.164e-05 [2.82e-05, 3.53e-05] | 1.748e-05 | 0.552 | refused (correctly) | Tier 3 cross-check (76800 inputs) | 3.906e-05 [0, 0.000218] | agree |
| nonlin/single/c_attn_core_ref/normal | 2 | ok | 0.7503 [0.75, 0.751] | 0.7536 | 1 | holds |  | 0.753 [0.745, 0.761] | DISAGREE |
| nonlin/single/c_silu/normal | 2 | ok | 0.3283 [0.328, 0.329] | 0.3182 | 0.969 | holds |  | 0.3294 [0.325, 0.334] | agree |
| nonlin/single/c_gelu_tanh/normal | 2 | ok | 0.4288 [0.429, 0.429] | 0.4236 | 0.988 | holds |  | 0.4312 [0.427, 0.435] | agree |
| nonlin/single/c_gelu_erf/normal | 2 | ok | 0.3875 [0.387, 0.388] | 0.3793 | 0.979 | holds |  | 0.3911 [0.386, 0.396] | agree |
| nonlin/single/c_gelu_bf16/normal | 0 | ok | 5.308e-05 [4.99e-05, 5.63e-05] | 3.171e-05 | 0.597 | refused (correctly) | Tier 3 cross-check (65712 inputs) | 0.0001172 [2.61e-05, 0.000255] | agree |
| nonlin/single/c_triton_gelu_bf16/normal | 3 | ok | 2.734e-06 [2.02e-06, 3.55e-06] | 1.763e-05 | 6.45 | refused (correctly) | Tier 3 cross-check (75326 inputs) | 0 [0, 7.2e-05] | agree |
| nonlin/single/c_softcap/normal | 3 | ok | 0.3462 [0.346, 0.346] | 1 | 2.89 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.346 [0.342, 0.35] | agree |
| nonlin/single/c_cve_silumul_bf16_inductor/normal | 3 | ok | 1.282e-05 [1.24e-05, 1.32e-05] | 6.276e-06 | 0.49 | refused (correctly) | Tier 3 cross-check (2508 inputs) | 1.099e-05 [2.63e-06, 2.09e-05] | agree |
| nonlin/single/c_cve_addnorm_inductor/normal | 2 | ok | 0.474 [0.473, 0.475] | 0.4629 | 0.977 | holds |  | 0.4726 [0.458, 0.487] | agree |
| nonlin/single/c_lse_naive/normal | 2 | ok | 0.0763 [0.0745, 0.0782] | 0.07295 | 0.956 | holds |  | 0.06 [0.0205, 0.102] | agree |
| nonlin/single/c_lse_stable/normal | 3 | ok | 0.212 [0.209, 0.215] | 0.1275 | 0.602 | refused (correctly) | Tier 3 cross-check (11643 inputs) | 0.185 [0.132, 0.246] | agree |
| nonlin/single/c_sum_kahan/normal | 3 | ok | 0.4027 [0.399, 0.406] | 0.2063 | 0.512 | refused (correctly) | Tier 3 cross-check (4167 inputs) | 0.415 [0.34, 0.49] | agree |
| nonlin/pair/c_equiv_softmax-c0/normal | 3 | ok | 0.6696 [0.669, 0.67] | 0.7085 | 1.06 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.6694 [0.658, 0.68] | agree |
| nonlin/pair/c_equiv_softmax-c1/normal | 3 | ok | 0.6855 [0.685, 0.686] | 0.7646 | 1.12 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.6949 [0.68, 0.71] | agree |
| nonlin/pair/c_equiv_rmsnorm-c0/normal | 3 | ok | 0.1461 [0.144, 0.148] | 0.4882 | 3.34 | refused (correctly) | Tier 3 cross-check (8010 inputs) | 0.1442 [0.101, 0.187] | agree |
| nonlin/pair/c_equiv_layernorm-c0/normal | 3 | ok | 0.2784 [0.277, 0.28] | 0.4201 | 1.51 | refused (correctly) | Tier 3 cross-check (2010 inputs) | 0.297 [0.261, 0.333] | agree |
| nonlin/pair/c_equiv_attn-c0/normal | 2 | ok | 0.7901 [0.79, 0.791] | 0.8148 | 1.03 | holds |  | 0.7969 [0.789, 0.805] | DISAGREE |
| nonlin/pair/c_equiv_elementwise_gelu_tanh-c0/normal | 3 | ok | 0.9317 [0.932, 0.932] | 0.7744 | 0.831 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.9322 [0.93, 0.934] | agree |
| nonlin/pair/c_equiv_elementwise_silu-c0/normal | 3 | ok | 0.3471 [0.347, 0.347] | 0.4996 | 1.44 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.3473 [0.343, 0.352] | agree |
| nonlin/pair/c_equiv_elementwise_gelu_bf16-c0/normal | 0 | ok | 5.2e-05 [4.89e-05, 5.52e-05] | 1.657e-05 | 0.319 | refused (correctly) | Tier 3 cross-check (30030 inputs) | 0.0001172 [2.61e-05, 0.000255] | agree |
| nonlin/pair/c_equiv_elementwise_gelu_erf-c0/normal | 3 | ok | 0.01381 [0.0138, 0.0139] | 0.4635 | 33.6 | refused (correctly) | Tier 3 cross-check (810 inputs) | 0.01395 [0.0129, 0.015] | agree |
| nonlin/pair/errdist_cve_addnorm-c0/normal | 3 | ok | 0.06787 [0.0667, 0.069] | 0.5128 | 7.56 | refused (correctly) | Tier 3 cross-check (17402 inputs) | 0.06834 [0.0481, 0.0886] | agree |
| nonlin/pair/errdist_cve_silumul-c0/normal | 3 | ok | 0.2488 [0.249, 0.249] | 0.303 | 1.22 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.2511 [0.248, 0.254] | agree |
| nonlin/pair/errdist_cve_add_bf16-c0/normal | 3 | ok | 0.2202 [0.22, 0.22] | 0.1785 | 0.811 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.2211 [0.22, 0.222] | agree |
| nonlin/pair/errdist_cve_silumul_bf16-c0/normal | 3 | ok | 0.2747 [0.275, 0.275] | 0.3658 | 1.33 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.2748 [0.274, 0.276] | agree |
| nonlin/single/r4_softmax_torch/laplace | 3 | ok | 0.7561 [0.756, 0.757] | 0.7075 | 0.936 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.7588 [0.749, 0.768] | agree |
| nonlin/single/r4_softmax_triton/laplace | 2 | ok | 0.8061 [0.806, 0.807] | 0.7943 | 0.985 | holds |  | 0.8028 [0.795, 0.811] | agree |
| nonlin/single/c_triton_softmax_online/laplace | 2 | ok | 0.809 [0.809, 0.809] | 0.7987 | 0.987 | holds |  | 0.8061 [0.798, 0.814] | agree |
| nonlin/single/r4_rmsnorm_torch/laplace | 3 | ok | 1.025e-05 [8.28e-06, 1.24e-05] | 7.266e-06 | 0.709 | refused (correctly) | Tier 3 cross-check (76800 inputs) | 0 [0, 0.000144] | agree |
| nonlin/single/r4_rmsnorm_triton/laplace | 3 | ok | 1.094e-05 [8.88e-06, 1.32e-05] | 7.264e-06 | 0.664 | refused (correctly) | Tier 3 cross-check (76800 inputs) | 0 [0, 0.000144] | agree |
| nonlin/single/r4_layernorm_torch/laplace | 3 | ok | 2.393e-05 [2.09e-05, 2.71e-05] | 9.48e-06 | 0.396 | refused (correctly) | Tier 3 cross-check (76800 inputs) | 0 [0, 0.000144] | agree |
| nonlin/single/r4_layernorm_triton/laplace | 3 | ok | 2.539e-05 [2.23e-05, 2.87e-05] | 1.198e-05 | 0.472 | refused (correctly) | Tier 3 cross-check (76800 inputs) | 3.906e-05 [0, 0.000218] | agree |
| nonlin/single/c_attn_core_ref/laplace | 2 | ok | 0.7293 [0.729, 0.73] | 0.7355 | 1.01 | holds |  | 0.7263 [0.717, 0.736] | agree |
| nonlin/single/c_silu/laplace | 2 | ok | 0.3344 [0.334, 0.335] | 0.3223 | 0.964 | holds |  | 0.3317 [0.328, 0.336] | agree |
| nonlin/single/c_gelu_tanh/laplace | 2 | ok | 0.436 [0.436, 0.436] | 0.4342 | 0.996 | holds |  | 0.4376 [0.433, 0.442] | agree |
| nonlin/single/c_gelu_erf/laplace | 2 | ok | 0.4035 [0.403, 0.404] | 0.394 | 0.976 | holds |  | 0.4051 [0.401, 0.41] | agree |
| nonlin/single/c_gelu_bf16/laplace | 0 | ok | 0.006225 [0.00619, 0.00626] | 0.007002 | 1.12 | refused (correctly) | Tier 3 cross-check (1786 inputs) | 0.005938 [0.00525, 0.00664] | agree |
| nonlin/single/c_triton_gelu_bf16/laplace | 3 | ok | 0.005153 [0.00512, 0.00518] | 0.006405 | 1.24 | refused (correctly) | Tier 3 cross-check (2151 inputs) | 0.004844 [0.00421, 0.00548] | agree |
| nonlin/single/c_softcap/laplace | 3 | ok | 0.3615 [0.361, 0.362] | 1 | 2.77 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.3591 [0.355, 0.363] | agree |
| nonlin/single/c_cve_silumul_bf16_inductor/laplace | 3 | ok | 1.135e-05 [1.1e-05, 1.17e-05] | 7.02e-06 | 0.618 | refused (correctly) | Tier 3 cross-check (9912 inputs) | 6.104e-06 [8.47e-07, 1.42e-05] | agree |
| nonlin/single/c_cve_addnorm_inductor/laplace | 2 | ok | 0.4832 [0.482, 0.484] | 0.4709 | 0.974 | holds |  | 0.4694 [0.453, 0.486] | agree |
| nonlin/single/c_lse_naive/laplace | 2 | ok | 0.115 [0.113, 0.117] | 0.114 | 0.991 | holds |  | 0.11 [0.0666, 0.162] | agree |
| nonlin/single/c_lse_stable/laplace | 3 | ok | 0.1507 [0.148, 0.153] | 0.1023 | 0.679 | refused (correctly) | Tier 3 cross-check (18134 inputs) | 0.145 [0.0951, 0.202] | agree |
| nonlin/single/c_sum_kahan/laplace | 3 | ok | 0.4124 [0.409, 0.416] | 0.227 | 0.551 | refused (correctly) | Tier 3 cross-check (4322 inputs) | 0.43 [0.36, 0.502] | agree |
| nonlin/pair/c_equiv_softmax-c0/laplace | 2 | ok | 0.7669 [0.766, 0.767] | 0.77 | 1 | holds |  | 0.7721 [0.762, 0.782] | agree |
| nonlin/pair/c_equiv_softmax-c1/laplace | 3 | ok | 0.7729 [0.772, 0.773] | 0.8347 | 1.08 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.7695 [0.759, 0.78] | agree |
| nonlin/pair/c_equiv_rmsnorm-c0/laplace | 3 | ok | 0.1588 [0.157, 0.161] | 0.4888 | 3.08 | refused (correctly) | Tier 3 cross-check (12170 inputs) | 0.1414 [0.0942, 0.189] | agree |
| nonlin/pair/c_equiv_layernorm-c0/laplace | 3 | ok | 0.2664 [0.265, 0.268] | 0.3872 | 1.45 | refused (correctly) | Tier 3 cross-check (1666 inputs) | 0.275 [0.245, 0.305] | agree |
| nonlin/pair/c_equiv_attn-c0/laplace | 2 | ok | 0.7751 [0.775, 0.776] | 0.8018 | 1.03 | holds |  | 0.7727 [0.763, 0.782] | agree |
| nonlin/pair/c_equiv_elementwise_gelu_tanh-c0/laplace | 3 | ok | 0.9065 [0.906, 0.907] | 0.7558 | 0.834 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.9058 [0.903, 0.908] | agree |
| nonlin/pair/c_equiv_elementwise_silu-c0/laplace | 3 | ok | 0.3557 [0.355, 0.356] | 0.5088 | 1.43 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.3518 [0.348, 0.356] | agree |
| nonlin/pair/c_equiv_elementwise_gelu_bf16-c0/laplace | 0 | ok | 0.001976 [0.00196, 0.002] | 0.003266 | 1.65 | refused (correctly) | Tier 3 cross-check (6367 inputs) | 0.002148 [0.00174, 0.00259] | agree |
| nonlin/pair/c_equiv_elementwise_gelu_erf-c0/laplace | 3 | ok | 0.03018 [0.0301, 0.0303] | 0.4739 | 15.7 | refused (correctly) | Tier 3 cross-check (386 inputs) | 0.02889 [0.0275, 0.0304] | agree |
| nonlin/pair/errdist_cve_addnorm-c0/laplace | 3 | ok | 0.06523 [0.0641, 0.0663] | 0.5122 | 7.85 | refused (correctly) | Tier 3 cross-check (15624 inputs) | 0.07145 [0.0455, 0.0974] | agree |
| nonlin/pair/errdist_cve_silumul-c0/laplace | 3 | ok | 0.2474 [0.247, 0.248] | 0.299 | 1.21 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.2464 [0.244, 0.249] | agree |
| nonlin/pair/errdist_cve_add_bf16-c0/laplace | 3 | ok | 0.221 [0.221, 0.221] | 0.179 | 0.81 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.2207 [0.219, 0.222] | agree |
| nonlin/pair/errdist_cve_silumul_bf16-c0/laplace | 3 | ok | 0.2715 [0.271, 0.272] | 0.3595 | 1.32 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.2724 [0.271, 0.273] | agree |
| nonlin/single/r4_softmax_torch/student3 | 3 | ok | 0.8016 [0.801, 0.802] | 0.7544 | 0.941 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.8008 [0.789, 0.813] | agree |
| nonlin/single/r4_softmax_triton/student3 | 2 | ok | 0.8432 [0.843, 0.844] | 0.8333 | 0.988 | holds |  | 0.8489 [0.841, 0.857] | agree |
| nonlin/single/c_triton_softmax_online/student3 | 2 | ok | 0.8451 [0.845, 0.846] | 0.8361 | 0.989 | holds |  | 0.8513 [0.842, 0.86] | agree |
| nonlin/single/r4_rmsnorm_torch/student3 | 2 | ok | 1.045e-05 [8.42e-06, 1.26e-05] | 9.061e-06 | 0.867 | holds |  | 0 [0, 0.000144] | agree |
| nonlin/single/r4_rmsnorm_triton/student3 | 2 | ok | 1.025e-05 [8.24e-06, 1.24e-05] | 9.549e-06 | 0.931 | holds |  | 3.906e-05 [0, 0.000218] | agree |
| nonlin/single/r4_layernorm_torch/student3 | 3 | ok | 2.812e-05 [2.49e-05, 3.16e-05] | 1.319e-05 | 0.469 | refused (correctly) | Tier 3 cross-check (76800 inputs) | 7.813e-05 [0, 0.000282] | agree |
| nonlin/single/r4_layernorm_triton/student3 | 3 | ok | 2.529e-05 [2.22e-05, 2.86e-05] | 1.368e-05 | 0.541 | refused (correctly) | Tier 3 cross-check (76800 inputs) | 3.906e-05 [0, 0.000218] | agree |
| nonlin/single/c_attn_core_ref/student3 | 2 | ok | 0.7195 [0.719, 0.72] | 0.7284 | 1.01 | holds |  | 0.7241 [0.712, 0.736] | agree |
| nonlin/single/c_silu/student3 | 2 | ok | 0.3325 [0.332, 0.333] | 0.3225 | 0.97 | holds |  | 0.3361 [0.332, 0.34] | agree |
| nonlin/single/c_gelu_tanh/student3 | 2 | ok | 0.4436 [0.443, 0.444] | 0.4383 | 0.988 | holds |  | 0.442 [0.438, 0.446] | agree |
| nonlin/single/c_gelu_erf/student3 | 2 | ok | 0.4096 [0.409, 0.41] | 0.4097 | 1 | holds |  | 0.4102 [0.405, 0.415] | agree |
| nonlin/single/c_gelu_bf16/student3 | 0 | ok | 0.01039 [0.0103, 0.0104] | 0.0112 | 1.08 | refused (correctly) | Tier 3 cross-check (1109 inputs) | 0.01051 [0.00964, 0.0114] | agree |
| nonlin/single/c_triton_gelu_bf16/student3 | 3 | ok | 0.009317 [0.00928, 0.00936] | 0.01056 | 1.13 | refused (correctly) | Tier 3 cross-check (1199 inputs) | 0.009395 [0.00853, 0.0103] | agree |
| nonlin/single/c_softcap/student3 | 3 | ok | 0.3631 [0.363, 0.363] | 0.9999 | 2.75 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.3655 [0.361, 0.37] | agree |
| nonlin/single/c_cve_silumul_bf16_inductor/student3 | 3 | ok | 1.408e-05 [1.37e-05, 1.45e-05] | 1.305e-05 | 0.927 | refused (correctly) | Tier 3 cross-check (6967 inputs) | 1.343e-05 [5.37e-06, 2.4e-05] | agree |
| nonlin/single/c_cve_addnorm_inductor/student3 | 2 | ok | 0.4675 [0.467, 0.468] | 0.4548 | 0.973 | holds |  | 0.4671 [0.453, 0.481] | agree |
| nonlin/single/c_lse_naive/student3 | 2 | ok | 0.1452 [0.143, 0.148] | 0.1443 | 0.994 | holds |  | 0.17 [0.114, 0.229] | agree |
| nonlin/single/c_lse_stable/student3 | 3 | ok | 0.1328 [0.13, 0.135] | 0.09868 | 0.743 | refused (correctly) | Tier 3 cross-check (18024 inputs) | 0.145 [0.0951, 0.202] | agree |
| nonlin/single/c_sum_kahan/student3 | 3 | ok | 0.4118 [0.408, 0.415] | 0.253 | 0.615 | refused (correctly) | Tier 3 cross-check (4266 inputs) | 0.455 [0.385, 0.527] | agree |
| nonlin/pair/c_equiv_softmax-c0/student3 | 2 | ok | 0.8102 [0.81, 0.811] | 0.8077 | 0.997 | holds |  | 0.8192 [0.808, 0.83] | agree |
| nonlin/pair/c_equiv_softmax-c1/student3 | 3 | ok | 0.8142 [0.814, 0.815] | 0.8635 | 1.06 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.8188 [0.807, 0.83] | agree |
| nonlin/pair/c_equiv_rmsnorm-c0/student3 | 3 | ok | 0.1686 [0.166, 0.171] | 0.5032 | 2.98 | refused (correctly) | Tier 3 cross-check (12930 inputs) | 0.1472 [0.104, 0.191] | agree |
| nonlin/pair/c_equiv_layernorm-c0/student3 | 3 | ok | 0.2638 [0.262, 0.265] | 0.3879 | 1.47 | refused (correctly) | Tier 3 cross-check (1921 inputs) | 0.2623 [0.232, 0.293] | agree |
| nonlin/pair/c_equiv_attn-c0/student3 | 2 | ok | 0.7684 [0.768, 0.769] | 0.7976 | 1.04 | holds |  | 0.7747 [0.765, 0.784] | agree |
| nonlin/pair/c_equiv_elementwise_gelu_tanh-c0/student3 | 3 | ok | 0.9112 [0.911, 0.911] | 0.7654 | 0.84 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.9121 [0.91, 0.915] | agree |
| nonlin/pair/c_equiv_elementwise_silu-c0/student3 | 3 | ok | 0.3577 [0.358, 0.358] | 0.506 | 1.41 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.3557 [0.352, 0.36] | agree |
| nonlin/pair/c_equiv_elementwise_gelu_bf16-c0/student3 | 0 | ok | 0.002136 [0.00212, 0.00216] | 0.005737 | 2.69 | refused (correctly) | Tier 3 cross-check (5374 inputs) | 0.001914 [0.00155, 0.00233] | agree |
| nonlin/pair/c_equiv_elementwise_gelu_erf-c0/student3 | 3 | ok | 0.02956 [0.0295, 0.0296] | 0.4822 | 16.3 | refused (correctly) | Tier 3 cross-check (410 inputs) | 0.02973 [0.0281, 0.0314] | agree |
| nonlin/pair/errdist_cve_addnorm-c0/student3 | 3 | ok | 0.06944 [0.0684, 0.0705] | 0.5097 | 7.34 | refused (correctly) | Tier 3 cross-check (15238 inputs) | 0.06162 [0.0384, 0.0848] | agree |
| nonlin/pair/errdist_cve_silumul-c0/student3 | 3 | ok | 0.2462 [0.246, 0.246] | 0.2983 | 1.21 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.2487 [0.246, 0.251] | agree |
| nonlin/pair/errdist_cve_add_bf16-c0/student3 | 3 | ok | 0.219 [0.219, 0.219] | 0.1789 | 0.817 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.2196 [0.218, 0.221] | agree |
| nonlin/pair/errdist_cve_silumul_bf16-c0/student3 | 3 | ok | 0.2722 [0.272, 0.272] | 0.3791 | 1.39 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.2725 [0.272, 0.274] | agree |
| inerr/rmsnorm/rel1e-6 | 2 | - | 0.9659 [0.966, 0.966] | 0.9659 | 1 | holds |  | 0.9657 [0.963, 0.968] | agree |
| inerr/rmsnorm/bf16-round | 2 | - | 1 [1, 1] | 1 | 0.923 | holds |  | 1 [1, 1] | agree |
| inerr/softmax/abs0.5-on-N(0,16) | 3 | - | 1 [1, 1] | 1 | 1.02 | refused (would have held) | input errors | 1 [1, 1] | agree |
| inerr/softmax/rel1e-6 | 2 | - | 0.9441 [0.944, 0.944] | 0.9435 | 0.999 | holds |  | 0.9423 [0.938, 0.947] | agree |
| inerr/silumul/rel1e-4 | 2 | - | 0.9998 [1, 1] | 0.9998 | 1.01 | holds |  | 0.9998 [1, 1] | agree |
| inerr/silumul-pair/rel1e-4 | 3 | - | 0.2059 [0.206, 0.206] | 0.2237 | 0.714 | refused (correctly) | Tier 3 cross-check (300 inputs) | 0.206 [0.204, 0.208] | agree |
| inerr/addnorm-pair/rel1e-4 | 3 | - | 0.06637 [0.0653, 0.0674] | 0.5009 | 6.73 | refused (correctly) | Tier 3 cross-check (13010 inputs) | 0.08318 [0.0569, 0.109] | agree |

### 7.4 Block-scaled rows with the block-scaled chain tier (2026-11-06)

The 25 MX / NVFP4 rows of §7.3 under the default policy with `tier2bs` (§5.3.1), against
the same 20,000-input truth. Every headline is right (within max(5%, the truth's 95%
half-width), or Tier 3 itself).
- **Two new Tier 2 headlines:** MXF8 normal (0.966x) and laplace (1.009x); the gate
  passes and the cross-check agrees at 300 inputs.
- **Refused by the gate** (7 single rows): MXF8 Student t, lognormal, outliers,
  codes, ints_scaled; MXF4 codes and Student t. The cross-check fails on all but MXF4
  Student t (4e-12 against a Tier 3 of 0), lognormal (0.955x) included.
- **Not run:** pairs (structural refusal), NVFP4 (4e7 term atoms per step), and rows that
  Tier 0 answers.

| row | headline tier | headline / truth | truth P(differ) | tier2bs | tier2bs / truth | gate (y excess kurtosis) | cross-check (inputs) | default run, s |
|---|---|---|---|---|---|---|---|---|
| mxf8/single/normal-mx | 2 | 0.966 | 0.04498 | 0.04346 | 0.966 | pass (0.024) | agrees (300) | 42 |
| mxf8/single/student_t-mx | 3 | 1.005 | 0.08835 | 0.09894 | 1.120 | refused (1.64e+05) | FAILS (300) | 207 |
| mxf8/single/laplace-mx | 2 | 1.009 | 0.1093 | 0.1103 | 1.009 | pass (0.13) | agrees (300) | 59 |
| mxf8/single/lognormal-mx | 3 | 1.008 | 0.00501 | 0.004786 | 0.955 | refused (15.5) | FAILS (300) | 41 |
| mxf8/single/outliers-mx | 3 | 0.998 | 0.3109 | 0.7882 | 2.535 | refused (84.3) | FAILS (300) | 234 |
| mxf8/single/codes | 3 | 1.000 | 0.8462 | 0.9074 | 1.072 | refused (77.4) | FAILS (300) | 311 |
| mxf8/single/ints | 0 | 0 = 0 | 0 | - | - | not run | - | 19 |
| mxf8/single/ints_scaled | 3 | 1.003 | 0.292 | 0.686 | 2.349 | refused (34.3) | FAILS (300) | 654 |
| mxf8/pair-rev/normal-mx | 3 | 0.999 | 0.03405 | - | - | not run (pair) | - | 147 |
| mxf8/pair-rev/codes | 3 | 1.001 | 0.8403 | - | - | not run (pair) | - | 340 |
| mxf8/pair-rev/ints_scaled | 3 | 1.001 | 0.2436 | - | - | not run (pair) | - | 179 |
| mxf8/pair-rev/student_t-mx | 3 | 1.004 | 0.06531 | - | - | not run (pair) | - | 233 |
| mxf8/pair-cublas/normal-mx | 0 | 0 = 0 | 0 | - | - | not run (pair) | - | 66 |
| mxf4/single/normal-mx | 0 | 0 = 0 | 0 | - | - | not run | - | 17 |
| mxf4/single/codes | 3 | 1.002 | 0.3346 | 0.6525 | 1.950 | refused (36) | FAILS (300) | 45 |
| mxf4/single/student_t-mx | 3 | 0 = 0 | 0 | 3.929e-12 | - | refused (4.3e+13) | agrees (300) | 32 |
| mxf4/single/laplace-mx | 0 | 0 = 0 | 0 | - | - | not run | - | 18 |
| mxf4/pair-rev/codes | 3 | 1.004 | 0.1595 | - | - | not run (pair) | - | 135 |
| mxf4/pair-rev/student_t-mx | 3 | 0 = 0 | 0 | - | - | not run (pair) | - | 78 |
| nvf4/single/normal-nv | 3 | 0 = 0 | 0 | - | - | not run | - | 30 |
| nvf4/single/student_t-nv | 3 | 0.768 | 7.019e-08 | - | - | not run | - | 61 |
| nvf4/single/codes | 3 | 1.000 | 0.7877 | - | - | not run | - | 30 |
| nvf4/pair-dq/normal-nv | 3 | 0 = 0 | 0 | - | - | not run (pair) | - | 77 |
| nvf4/pair-dq/codes | 3 | 1.000 | 0.5593 | - | - | not run (pair) | - | 61 |
| nvf4/pair-cublas/normal-nv | 0 | 0 = 0 | 0 | - | - | not run (pair) | - | 69 |

## 8. Assumptions and open items

ASSUMPTIONS.md: N-ERRDIST-IID, -TEMPLATE, -RESIDUE, -MARKOV, -CF, -CLT, -BSCLT, -PAIR,
-FLIP, -MC. These are estimates, not proofs. Only Tier 0 is exact (and only on the support
of the element laws), and Tier 3 is exact per sample.

Open items:
- a joint chain for block-scaled pairs that share the K partition (Tier 1 1.2-1.3x, the
  main remaining bias);
- Tier 1's residual 6% on MX normal / Laplace recipes (bits above the tracked two);
- Tier 1 speed on MXF8 random codes (> 60 min) and NVFP4 (skipped above 5e6 atoms);
- Tier 2's DAG CLT on plain MMA chains over narrow inputs (0.86-2.85x; Tier 1 covers them);
  tier2bs on pairs (0.77x on an MXF8 reverse-K pair) and on NVFP4 with a global scale;
- per-index scale masks in the spec (true small-K runs on a K = 256 kernel; Tier 1's small-K
  coverage is the synthetic one-step tests);
- softmax, norms, attention and elementwise programs have their own analytic tiers (§9).

## 9. Non-chain programs: softmax, norms, attention, elementwise (agent AG)

**Status (2026-11-05, agent AG).** Programs whose outputs are not plain accumulation chains
have their own analytic tiers. These are softmax (two-pass and online), RMSNorm, layer norm,
attention (QK^T, softmax, PV), and elementwise functions of one to three input elements (exp
/ rcp / rsqrt / tanh MUFU slices, gelu, silu, softcap, silu * mul, add, type conversions).
The code is in `crates/sim/src/errdist/nonlin/`. These estimates are `applicable` only when
they agree with Tier 3 (the cross-check, §3.2). `best` then picks in tier order with the chain tiers' estimates (§3): the elementwise
law and the linearized Tier 1 before Tier 2. Otherwise `best` stays at Tier 3, so the
headline is never worse than Tier 3.

### 9.1 Extraction

- **Segment mode.** Single kernels and pairs retry in segment mode when the default
  extraction leaves an output untracked (softmax, norms, multi-launch regions).
- **The DAG's op kinds.** The DAG then carries:
  - instruction slices (`NlOp::Seg`: the instructions between two numerics nodes, evaluated
    bit-exactly by `seg.rs`);
  - MUFU ops (`NlOp::Mufu`);
  - maxima (`NlOp::Max`);
  - the chain ops (Add, Fma, Round, Mma).
- **Multi-launch regions.** These are extracted through global memory: x @ w1 + x @ w2 as two
  sgemms plus an add launch, against x @ (w1 + w2) (`fixtures/errdist/mm2.toml`).
- **Checks before estimating.**
  - Byte workspaces that no leaf reads keep their captured bytes.
  - A program whose DAG tracks no output is refused; the message names the instruction
    each failed slice could not evaluate.
  - So is a program whose DAG disagrees with the captured GPU output at the captured input.
    For example, inductor's `addmm` copies its input with a device-to-device memcpy, which
    the capture does not record (`fixtures/errdist/c_mm2_addmm`). Recording memcpys in
    captures is open.
- **Slice identification** (`ident.rs`, N-ERRDIST-IDENT). Each slice is fitted to a real
  function on the captured input plus 24 perturbed inputs:
  - f(c0 + Σ c_i a_i) with f in {exp2, 1/x, 1/sqrt, sqrt, tanh, identity};
  - multiplier and divisor forms;
  - a / b;
  - polynomials up to degree 3;
  - named functions (erf, gelu, tanh-gelu, silu, log, ...) with an outer affine map.
    Triton's `tl.log` is a software polynomial with an integer exponent extraction
    (`I2FP.F32.S32`), which the slice evaluator models (I2F / F2I of 8-32 bits); it is
    identified as log (`fixtures/c_lse_naive`, `c_lse_stable`).

  Constants snap to named values (log2 e, ln 2, 1/128, eps, √(2/π), 0.044715, powers of
  two) when that keeps the fit.
- **The exact value** (`real.rs`, N-ERRDIST-REAL) is the program in double-double real
  arithmetic, with slices replaced by their real functions. Tier 3 uses it as "exact" for
  single kernels.

### 9.2 Local error laws ("compile")

- **Slices and MUFU units** (`table.rs`, N-ERRDIST-NL-INDEP). Each slice function and MUFU
  variant gets a table indexed by cells of its index variable: sign, binade, and 1/8 of the
  binade. The index variable is the exp / rsqrt argument, the polynomial or named-function
  argument, or the quotient. Per cell, the table holds:
  - the law of the slice's own error (g - f) / ulp32(f), from 96 bit-exact evaluations:
    mean, second moment, P(0), a 1/8-ulp histogram;
  - the integer offset from RN(f);
  - the error before the slice's last rounding, where the slice ends in FFMA / FMUL / FADD /
    div.rn.
- **Pairs of slices.** Two slices of one function in two kernels get a test for equal
  evaluations (`same_function`) and, if they depend on the driver argument only, a joint
  table of g_C - g_R at equal arguments.
- **Round-to-nearest ops** use the exact residue law of their value's lattice, carrying the
  parity of the low bit for ties.

### 9.3 Linearized tiers (single kernels)

`lin.rs` and `agg.rs`, N-ERRDIST-NL-LIN. The steps:

1. Sample 48 nominal inputs from the spec, in real arithmetic only.
2. At each input, give every op its local law and its adjoint to the output by reverse
   mode.
3. Express each site's contribution in output ulps.
4. Add the output's own last rounding:
   - continuous position;
   - lattice residue for exact sums;
   - a final slice's offsets, or its pre-rounding error.
5. Tier 1 convolves the 6 largest discrete sites exactly on a 1/64-ulp grid, with the rest
   normal. Tier 2 is all normal.
6. For 16-bit outputs rounded from an f32 value, the probability that the computed value is
   a tie of the output grid is measured on bit-exact evaluations (RNE then picks the even
   neighbour). It is not taken as 2^-16, because the low bits of products with a 16-bit
   factor are not uniform.

Each report notes the variance budget by site class (e.g. softmax torch: f32 fma 64%, f32
add 19%, the exp2 slice before its last rounding 16%).

### 9.4 Pairs

`agg.rs` `Ctx::joint`, N-ERRDIST-NL-JOINT.

- **Shared sub-computations** are merged by canonical hash: slices compare by function, and
  max trees flatten to sets.
- **Matched roundings.** Ops of the two kernels holding the same real value (found at two
  inputs, re-checked per input) share their grid position. Their difference is
  round(w + Z) - round(w), where Z is the upstream deviation. Z is computed recursively from
  the matched pairs below, down to depth 16.
- **Two matched slices.** Their own errors cancel in any of these cases:
  - they compute the same function of the index (equal tables, or equal evaluations);
  - the candidate's slice equals, bit for bit, the reference's sub-DAG above the nodes
    holding its arguments, tested at 256 points. Example: inductor's fused
    rsqrt(1e-6 + s / 128) against eager's three ops.

  Otherwise the pair table is used when it exists, else the two errors are independent.
- **Identical programs** are Tier 0 (D ≡ 0).

### 9.5 Elementwise law

`enumerate.rs`, N-ERRDIST-NL-ELEM. It applies when every output is the same function of one
to three input elements of different buffers and the elements have a plain law (no axis
scales or outliers, round-to-nearest quantization). The output's law is then the pushforward
of the elements' law through the bit-exact output function:

- **16-bit inputs:** exhaustive over every code pair whose product of counts is <= 4e6 (Tier
  0).
- **f32:** 2^20 codes stratified over the law, allotted in proportion to the mass of each
  1/256-binade cell and placed at random points in the cell (Tier 1). The CI comes from the
  effective sample size. Codes at cell midpoints would have only ~10 significant bits, and
  rounding-sensitive results come out biased (silu: 0.321 against 0.328).
- **Larger products:** a Latin hypercube with every leaf stratified over its whole law
  (Tier 1, binomial CI). A resampled product grid sees the hash-like indicator "the two exp
  slices differ at a" at only a few thousand values (silu * mul: 4% off).
- **Packed 16-bit words.** When the other element of a packed word sits in an output's cone,
  the output must be checked independent of it.

### 9.6 Compile once, query many

- **The model.** The slice / MUFU / pair tables, keyed by the programs' canonical hash
  (`nonlin::model_key`), are the `nonlin` section of the model file of §1.1, beside the
  `programs` section.
- **Querying.** `--model FILE` (§1.1) loads the section, or compiles it with a note when it
  is absent or its key differs. If the queried distribution reaches binades the tables do
  not cover, it extends them first (seconds). Compiled or extended tables are written back
  to the file.
- **Rust API:** `nonlin::Model::{compile, load, save}`, `nonlin::ensure`, `nonlin::analytic`.

Costs on the fixtures:
- Compiling: 0.0-0.2 s. Tables are 96 evaluations per cell over ±2 binades around the
  probes, and the file is ~1.6 MB for a softmax pair.
- A query: 0.1-27 s. Most of it is the 48 nominal inputs times the outputs' cones (attention
  pair: 17 s) and the f32 quadrature (1-8 s).
- Tier 3: 0.1-3 s at 20,000 inputs on these small programs.

The analytic tiers are therefore not faster than Tier 3 here. What they add:
- a per-site explanation;
- exact laws where Tier 3 is noisy, e.g. bf16 misrounding at 5e-5, which Tier 3 resolves
  only to ±10% with 10^6 outputs;
- independence from the sampling seed.

### 9.7 Validation

This is the matrix of `tools/errdist_nonlin_matrix.py` (2026-11-05). It has 16 single kernels and 13 pairs, each
under 7 input specs:
- `normal`: N(0, 1);
- `uniform`: U[-1, 1);
- `laplace`: Laplace(0, 1);
- `student3`: t with 3 degrees of freedom;
- `shift100`: N(100, 1), large-magnitude shifted rows;
- `normal8`: N(0, 8²);
- `int8`: integers -8..8.

Every run uses Tier 3 with 20,000 inputs.

How to read a cell:
- It shows Tier 3's P(bits differ), then the analytic estimate.
- The suffix says where the analytic estimate comes from:
  - `E`: the elementwise law;
  - `L1` / `L2`: linearized Tier 1 / Tier 2;
  - `I`: identical programs.
- ✓: `best` took the analytic estimate, i.e. it is within max(5%, CI) of Tier 3.
- ✗: `best` stayed at Tier 3. The value shown is then the elementwise or Tier 1 estimate.

Output formats:
- f32: softmax, attention, silu / gelu f32, the `c_equiv_*` norms.
- bf16: `r4_rmsnorm_*`, `r4_layernorm_*`, `*gelu_bf16`, `*_bf16*`.

| case | normal | uniform | laplace | student3 | shift100 | normal8 | int8 |
|---|---|---|---|---|---|---|---|
| r4_softmax_torch | 0.644 / 0.65L1 ✓ | 0.509 / 0.52L1 ✓ | 0.756 / 0.76L1 ✓ | 0.802 / 0.801L1 ✓ | 0.496 / 0.506L1 ✓ | 0.853 / 0.862L1 ✓ | 0.499 / 0.507L1 ✓ |
| r4_softmax_triton | 0.704 / 0.706L1 ✓ | 0.573 / 0.578L1 ✓ | 0.806 / 0.811L1 ✓ | 0.844 / 0.844L1 ✓ | 0.655 / 0.658L1 ✓ | 0.942 / 0.942L1 ✓ | 0.759 / 0.814L1 ✗ |
| c_triton_softmax_online | 0.712 / 0.716L1 ✓ | 0.599 / 0.593L1 ✓ | 0.809 / 0.814L1 ✓ | 0.845 / 0.846L1 ✓ | 0.667 / 0.673L1 ✓ | 0.943 / 0.941L1 ✓ | 0.764 / 0.795L2 ✓ |
| r4_rmsnorm_torch | 9.47e-06 / 1.07e-05L1 ✓ | 1.18e-05 / 1.22e-05L1 ✓ | 1.25e-05 / 8.22e-06L1 ✗ | 1.08e-05 / 9.51e-06L1 ✓ | 9.77e-08 / 1.68e-06L1 ✗ | 9.37e-06 / 7.39e-06L1 ✗ | 0 / 4.13e-06L1 ✗ |
| r4_rmsnorm_triton | 1.01e-05 / 9.24e-06L1 ✓ | 1.14e-05 / 1.24e-05L1 ✓ | 1.2e-05 / 8.69e-06L1 ✗ | 9.96e-06 / 9.04e-06L1 ✓ | 9.77e-08 / 1.72e-06L1 ✗ | 9.86e-06 / 8.1e-06L1 ✓ | 0 / 4.13e-06L1 ✗ |
| r4_layernorm_torch | 3.12e-05 / 1.25e-05L1 ✗ | 3.53e-05 / 3.02e-05L1 ✗ | 2.4e-05 / 3.29e-05L1 ✗ | 2.93e-05 / 8.59e-06L1 ✗ | 0.000125 / 0.00299L1 ✗ | 2.95e-05 / 1.53e-05L1 ✗ | 6.6e-05 / 3.09e-05L1 ✗ |
| r4_layernorm_triton | 3.13e-05 / 1.41e-05L1 ✗ | 3.56e-05 / 1.66e-05L1 ✗ | 2.32e-05 / 2.08e-05L1 ✓ | 2.63e-05 / 2.94e-05L2 ✓ | 8.59e-05 / 5.92e-05L1 ✗ | 3.3e-05 / 2.07e-05L1 ✗ | 4.4e-05 / 5.54e-05L1 ✗ |
| c_attn_core_ref | 0.75 / 0.753L1 ✓ | 0.75 / 0.75L1 ✓ | 0.73 / 0.734L1 ✓ | 0.72 / 0.73L1 ✓ | 0.55 / 0.552L1 ✓ | 0.675 / 0.709L2 ✓ | 0.711 / 0.728L1 ✓ |
| c_silu | 0.328 / 0.328E ✓ | 0.327 / 0.327E ✓ | 0.335 / 0.334E ✓ | 0.333 / 0.333E ✓ | 0 / 0E ✓ | 0.352 / 0.353E ✓ | 0.25 / 0.25E ✓ |
| c_gelu_tanh | 0.429 / 0.429E ✓ | 0.392 / 0.392E ✓ | 0.436 / 0.436E ✓ | 0.444 / 0.444E ✓ | 0 / 0E ✓ | 0.414 / 0.414E ✓ | 0.6 / 0.6E ✓ |
| c_gelu_erf | 0.388 / 0.388E ✓ | 0.337 / 0.337E ✓ | 0.403 / 0.404E ✓ | 0.409 / 0.409E ✓ | 0 / 0E ✓ | 0.483 / 0.483E ✓ | 0.563 / 0.562E ✓ |
| c_gelu_bf16 | 5.5e-05 / 5.49e-05E ✓ | 0 / 0E ✓ | 0.00618 / 0.00621E ✓ | 0.0104 / 0.0104E ✓ | 0 / 0E ✓ | 0.242 / 0.242E ✓ | 0.25 / 0.25E ✓ |
| c_triton_gelu_bf16 | 3.27e-06 / 3.34e-06L1 ✓ | 0 / 1.95e-06L1 ✗ | 0.00512 / 0.000221L1 ✗ | 0.00934 / 0.000279L1 ✗ | 0 / 1.2e-07L1 ✓ | 0.237 / 0.248L2 ✓ | 0.25 / - |
| c_softcap | 0.346 / 0.346E ✓ | 0.345 / 0.345E ✓ | 0.361 / 0.362E ✓ | 0.363 / 0.363E ✓ | 0.235 / 0.235E ✓ | 0.53 / 0.529E ✓ | 0.625 / 0.625E ✓ |
| c_cve_silumul_bf16_inductor | 1.3e-05 / 1.37e-05E ✗ | 1.95e-05 / 1.8e-05E ✗ | 1.12e-05 / 1.2e-05E ✗ | 1.44e-05 / 1.15e-05E ✗ | 0 / 0E ✓ | 0.000211 / 0.000214E ✓ | 0 / 0E ✓ |
| c_cve_addnorm_inductor | 0.474 / 0.467L1 ✓ | 0.5 / 0.502L1 ✓ | 0.482 / 0.481L1 ✓ | 0.467 / 0.464L1 ✓ | 0.569 / 0.542L1 ✓ | 0.474 / 0.469L1 ✓ | 0.36 / 0.373L2 ✓ |
| c_equiv_softmax/equiv.toml:0 | 0.671 / 0.702L2 ✓ | 0.598 / 0.649L1 ✗ | 0.766 / 0.784L1 ✓ | 0.81 / 0.817L1 ✓ | 0.67 / 0.712L1 ✗ | 0.933 / 0.934L1 ✓ | 0.8 / 0.83L1 ✓ |
| c_equiv_softmax/equiv.toml:1 | 0.686 / 0.764L1 ✗ | 0.624 / 0.694L1 ✗ | 0.773 / 0.836L1 ✗ | 0.815 / 0.862L1 ✗ | 0.686 / 0.716L1 ✓ | 0.933 / 0.944L1 ✓ | 0.803 / 0.829L1 ✓ |
| c_equiv_rmsnorm/equiv.toml:0 | 0.148 / 0.159L1 ✗ | 0.151 / 0.172L1 ✗ | 0.157 / 0.171L1 ✗ | 0.166 / 0.176L1 ✗ | 0.141 / 0.16L1 ✗ | 0.148 / 0.159L1 ✗ | 0 / 0.135L1 ✗ |
| c_equiv_layernorm/equiv.toml:0 | 0.277 / 0.29L1 ✓ | 0.314 / 0.309L1 ✓ | 0.267 / 0.283L1 ✗ | 0.264 / 0.282L1 ✗ | 0.853 / 0.973L1 ✗ | 0.276 / 0.293L1 ✗ | 0.234 / 0.343L1 ✗ |
| c_equiv_attn/equiv.toml:0 | 0.79 / 0.81L1 ✓ | 0.795 / 0.811L1 ✓ | 0.775 / 0.795L1 ✓ | 0.769 / 0.793L1 ✓ | 0.632 / 0.638L1 ✓ | 0.743 / 0.781L1 ✗ | 0.775 / 0.784L1 ✓ |
| c_equiv_elementwise/gelu_tanh.toml:0 | 0.932 / 0.932E ✓ | 0.915 / 0.915E ✓ | 0.906 / 0.906E ✓ | 0.911 / 0.912E ✓ | 0 / 0E ✓ | 0.492 / 0.492E ✓ | 0.533 / 0.533E ✓ |
| c_equiv_elementwise/silu.toml:0 | 0.347 / 0.346E ✓ | 0.349 / 0.348E ✓ | 0.356 / 0.355E ✓ | 0.358 / 0.358E ✓ | 0 / 0E ✓ | 0.456 / 0.456E ✓ | 0.5 / 0.5E ✓ |
| c_equiv_elementwise/gelu_bf16.toml:0 | 5.37e-05 / 5.35e-05E ✓ | 0 / 0E ✓ | 0.00196 / 0.00198E ✓ | 0.00214 / 0.00213E ✓ | 0 / 0E ✓ | 0.0139 / 0.0139E ✓ | 0.0625 / 0.0625E ✓ |
| c_equiv_elementwise/gelu_erf.toml:0 | 0.0137 / 0.0136E ✓ | 0 / 0E ✓ | 0.0301 / 0.03E ✓ | 0.0296 / 0.0297E ✓ | 0 / 0E ✓ | 0.0604 / 0.0604E ✓ | 0.313 / 0.312E ✓ |
| errdist/cve/addnorm.toml:0 | 0.0693 / 0.195L1 ✗ | 0.0641 / 0.184L1 ✗ | 0.0646 / 0.189L1 ✗ | 0.0697 / 0.203L1 ✗ | 0.0734 / 0.148L1 ✗ | 0.0694 / 0.195L1 ✗ | 0 / 0.117L1 ✗ |
| errdist/cve/silumul.toml:0 | 0.249 / 0.248E ✓ | 0.261 / 0.261E ✓ | 0.247 / 0.247E ✓ | 0.246 / 0.246E ✓ | 0 / 0E ✓ | 0.206 / 0.206E ✓ | 0.352 / 0.352E ✓ |
| errdist/cve/add_bf16.toml:0 | 0.22 / 0.22E ✓ | 0.222 / 0.222E ✓ | 0.221 / 0.221E ✓ | 0.219 / 0.219E ✓ | 0.121 / 0.121E ✓ | 0.22 / 0.22E ✓ | 0 / 0E ✓ |
| errdist/cve/silumul_bf16.toml:0 | 0.275 / 0.275E ✓ | 0.265 / 0.266E ✓ | 0.271 / 0.272E ✓ | 0.272 / 0.272E ✓ | 0 / 0E ✓ | 0.224 / 0.224E ✓ | 0.0625 / 0.0625E ✓ |

analytic best: 149/203

GB300 (`--gpu-trials 32`, GPU 0, normal spec):
- The DAG reproduced the GPU bit for bit on every input, with 0 mismatches, on:
  - `r4_softmax_torch`, `c_attn_core_ref`, `c_silu`, `r4_rmsnorm_torch`;
  - the softmax and RMSNorm pairs;
  - `cve/silumul_bf16`, `mm2`, `gelu_bf16`.
- GPU P(differ) against the analytic estimate:

| case | GPU | analytic |
|---|---|---|
| softmax torch | 0.647 | 0.650 (L1) |
| attention | 0.752 | 0.753 |
| silu | 0.323 | 0.328 (E) |
| silu*mul bf16 pair | 0.2750 | 0.2748 (E) |
| mm2 pair | 0.864 | 0.863 |
| softmax pair | 0.660 | 0.712 (L1 rejected, best = Tier 3) |
| rmsnorm pair | 0.136 | 0.156 (rejected) |


### 9.8 What falls back to Tier 3, and why

- **Pairs of reductions in different orders** (RMSNorm, layer norm, softmax vs online
  softmax, eager vs inductor add + RMSNorm). Tier 1 is 5-15% high, and addnorm is 2.8x high
  (0.195 against 0.069). The matched partial sums' joint laws overestimate P(K != 0): 0.38
  predicted against 0.21 observed per matched add on addnorm (N-ERRDIST-NL-JOINT). Joint
  laws of reductions are the main open item.
- **bf16 layer norm (single).** Tier 1 is up to 2.5x low. The 16-bit misrounding is
  dominated by rare near-zero outputs, where (x - mean) * rstd * w cancels the bias. The 48+
  nominal inputs do not sample them, and Tier 3 does.
- **bf16 RMSNorm under laplace, shift100 and int8.** Rare events again, and Tier 3 sees 0
  on integers.
- **`c_triton_gelu_bf16` under heavy tails.** A packed two-element bf16 slice: the
  elementwise law does not apply (each output depends on both halves through the packing).
  The linearized tier misses large-|x| behaviour (0.0002 against 0.005).
- **`c_cve_silumul_bf16_inductor` single.** P(differ) is ~1e-5. The Latin hypercube's
  binomial CI (4e6 draws) is wider than 5%, so it is not selected.
- **Not modeled at all:**
  - captures missing a device-to-device memcpy (refused, §9.1);
  - bf16 cuBLAS epilogues with `UTMAREDG.3D.ADD` (TMA reduce-add: unsupported SASS
    opcode);
  - specs with axis scales or outliers (no elementwise law; the linearized tiers still run).

## 10. Composition: errstate files and chains of kernels

A multi-launch region (matmul then RMSNorm, attention as QK^T / softmax / PV, an MLP)
composes stage by stage. Each kernel maps (input values, input errors) to (output value,
output error). The map is the kernel's own rounding plus the input errors pushed through it.
Its output state is the next kernel's input error (§2.1).

### 10.1 The errstate file

`--errstate FILE` (Python: `errstate=True`) writes the output's state from Tier 3's (value,
error) pairs. The error is against the exact result of the whole chain so far on error-free
inputs.

```json
{"kind": "ptxv-errstate", "version": 1, "model": "iid",
 "source": {"reference": {...}, "candidate": null, "tier3_inputs": 300},
 "tensors": {"out": {"shape": [32, 64], "format": "f32",
   "value_mean": ..., "value_var": ..., "error_mean": ..., "error_var": ..., "cov": ...,
   "linear": [alpha, beta, s0, s1], "n": 72147,
   "pairs": [[value, error], ...],
   "corr": {"error_same_row": ..., "error_other_row": ..., "value_same_row": ...,
            "value_other_row": ..., "value_error_same_row": ..., "value_sq_same_row": ...,
            "error_sq_same_row": ..., "n_same_row": ..., "n_other_row": ...}}}}
```

- `pairs` holds at most 32768 pairs (an even subsample in (input, output) order, so the file
  is reproducible). `linear` is the one-fit moment form, a summary only: the next kernel's
  linearized tier reads the pairs per value bin (§2.1).
- `model: "iid"` is the only state this ptxv writes or reads: elements independent, the
  joint law of (value, error) per element (N-ERRDIST-INERR). `corr` measures what the iid
  model drops (§10.3) and is not used. A richer state (a factor model, a low-rank
  covariance) would be a new `model` value, read through the same `error` field, with no
  change to the spec or the Python API.

### 10.2 CLI and Python

```
ptxv errdist chain/mm --inputs normal.json --errstate mm.es.json
ptxv errdist chain/rms --inputs rms.json     # "h": {"error": {"kind": "pairs", "file": "mm.es.json"}}
```

Python (docs/python.md): `m(inputs, errstate=True).as_input()` is the output as the next
input. `m1.then(m2, "h", **other_inputs)` chains compiled models. The intermediate errstates
are written automatically, and the result is the last kernel's `ErrDist`.

### 10.3 Correlations across outputs (measured)

These were measured on the end-to-end outputs (300 inputs, normal inputs, f32). Same row
means two outputs in one row of the last dimension.

| output | error, same row | error, other rows | value-error, same row | value^2, same row | error^2, same row |
|---|---|---|---|---|---|
| matmul (32x64) | -0.0012 | 0.0001 | -0.0007 | 0.013 | 0.008 |
| matmul -> RMSNorm | 0.0008 | -0.0000 | -0.0020 | -0.017 | -0.0001 |
| matmul -> softmax | -0.016 | -0.0004 | -0.0006 | -0.013 | 0.006 |
| attention | 0.0016 | -0.0002 | -0.0019 | 0.077 | 0.18 |
| MLP | 0.0025 | -0.0005 | 0.0002 | 0.097 | 0.070 |

- **The errors themselves are uncorrelated across outputs.** Softmax's -0.016 is its
  sum-to-one constraint (-1/(n-1) at n = 64). RMSNorm mixes a whole row, but the shared
  1/rms term is a common relative factor. Its errors stay uncorrelated (0.0008), as do
  value and error (-0.002), and even the squared errors (-0.0001): the outputs' own
  rounding dominates the shared rstd error.
- **The error magnitudes are correlated within a row**:
  - attention 0.18 and the MLP 0.07: rows with larger values have larger errors throughout;
  - the iid model draws each element's magnitude independently.

  At these sizes it does not matter. Under independence, the composed estimates below are
  within 5% of end to end, attention and the MLP included. A downstream row statistic of
  squared errors (a second norm on an attention output) is where it would show first.

### 10.4 Validation

Each region was run end to end (one multi-launch capture, Tier 3 with 1000 inputs, and
GB300 with 100 inputs on GPU 1). It was then run composed: each stage fed the previous
stage's errstate, assuming independence. The table gives the abs error std, with ratios to
the end-to-end Tier 3. Captures and scripts are in `chain/` of the run directory
(`validate.py`, `validate_cve.py`, `gpu_e2e.sh`).

| region | end-to-end Tier 3 | GB300 P(differ) vs Tier 3 | composed Tier 3 | composed Tier 2 | Tier 2 |
|---|---|---|---|---|---|
| matmul -> RMSNorm | 1.502e-7 | 0.8089 vs 0.8086 | 1.009 | 1.018 | applicable |
| matmul -> softmax | 7.057e-8 | 0.9812 vs 0.9816 | 1.046 | 1.034 | applicable |
| MLP (matmul -> silu * mul -> matmul) | 1.503e-4 | 0.9094 vs 0.9091 | 0.994 | 1.003 | applicable |
| attention (QK^T -> softmax -> PV) | 5.545e-8 | 0.8928 vs 0.8926 | 0.977 | 0.979 | applicable |
| silu(x w1) * (x w3), eager vs exact | 9.601e-6 | 0.9155 vs 0.9151 | 0.998 | 1.007 | applicable |
| same, torch.compile vs exact | 9.616e-6 | 0.9157 vs 0.9153 | 0.998 | 1.007 | applicable |
| same, compile - eager (pair) | 1.075e-6 | 0.2065 vs 0.2061 | 0.984 | 0.619 | refused (cross-check) |

Hand-written upstream errors, with no upstream run:

| region | input | reference | composed Tier 3 | Tier 2 |
|---|---|---|---|---|
| matmul -> RMSNorm | h ~ N(0, sd_mm) + absolute(sd of the matmul's error) | end to end 1.502e-7 | 1.033 | 1.036 |
| MLP | g ~ N(0, sd_g) + absolute(sd of g's error) | end to end 1.503e-4 | 0.998 | 0.996 |
| RMSNorm | h rounded to bf16 | its Tier 3 1.621e-3 | - | 0.999 |
| last matmul of the MLP | g with relative error 1e-3 | its Tier 3 0.5047 | - | 1.003 |

- With every stage composed under independence, Tier 3 matches the end-to-end run within 5%
  (the compile-vs-eager pair included). This tests the iid errstate, not a tier.
- Tier 2 is within 4% wherever it is applicable. Before the per-bin moments (§2.1), attention
  was 1.12. The compile-vs-eager pair is refused by the cross-check, as the linearized pair
  tier is without input errors (§9.8), and Tier 3 answers.
- The pair's composed Tier 3 sees the input errors cancel: both kernels read the same noisy
  value, and C - R keeps only the different roundings.
