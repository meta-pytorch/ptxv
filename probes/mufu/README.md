# MUFU / PTX `.approx` sweep (N-MUFU)

Exhaustive GB300 characterization of the multi-function unit and of the PTX approx ops built on
it, and the reconstruction the simulator uses (`crates/sim/src/mufu.rs`). 2026-11-03, GB300
(sm_103a), CUDA 13.0, driver 580, GPU 1.

## Files

| file | what |
|---|---|
| `kernels.cu` | one kernel per op: inline PTX on every input bit pattern (`base + i`) |
| `probe.cpp` | driver-API host: streams one op's outputs for all 2^32 inputs to stdout |
| `patch.py` | `mufu_raw.cubin`: the sin/cos kernels with the FMUL.RZ 1/(2 pi) constant patched to 1.0 (bare `MUFU.SIN/COS`) |
| `fits/*.json` | fitted interpolator tables (C0, C1, C2 per segment, rounding offset) |
| `gen_tables.py` | `fits/` -> `crates/sim/src/mufu_tables.rs` |
| `approx_ptx/` | every PTX approx form through the PTX front-end vs `gb300_out.bin` (cargo test) |
| `f64_ptx/` | `rcp/rsqrt.approx.ftz.f64` (MUFU.RCP64H / RSQ64H) through both front-ends vs `gb300_out.bin` |

## Reproduce

```bash
source env.sh
cd probes/mufu
nvcc -cubin -arch=sm_103a -O3 -o mufu.cubin kernels.cu && python patch.py mufu.cubin mufu_raw.cubin
g++ -O2 -o probe probe.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -lcuda
# one op, all 2^32 inputs, compared with the model on the fly (no dump needed):
CUDA_VISIBLE_DEVICES=1 ./probe . k_ex2_ftz | ptxv mufu-sweep ex2.ftz -
# with a dump (16 GiB) the sample for cargo test can be (re)written:
CUDA_VISIBLE_DEVICES=1 ./probe . k_ex2_ftz > ex2_ftz.bin
ptxv mufu-sweep ex2.ftz ex2_ftz.bin --sample ../../fixtures/mufu/samples.json
```

Kernel names: `k_ex2 k_ex2_ftz k_rcp k_rcp_ftz k_rsq k_rsq_ftz k_lg2 k_lg2_ftz k_sqrt
k_sqrt_ftz k_sin k_cos raw:k_sin raw:k_cos k_tanh` (f32, 2^32 inputs) and `h_ex2_f16
h_ex2_bf16 h_tanh_f16 h_tanh_bf16` (pass `0 65536`), `h_*_x2` (packed, 2^32 pairs), `d_rcp64h
d_rsq64h` (2^32 high words; `ptxv mufu-sweep rcp64h / rsq64h`).

## How ptxas (CUDA 13.0, sm_103a) expands the PTX ops

| PTX | SASS |
|---|---|
| `ex2.approx.ftz.f32` | `MUFU.EX2` |
| `ex2.approx.f32` | `FSETP.GEU P0, x, -126`; `@!P0 FMUL x, x, 0.5`; `MUFU.EX2`; `@!P0 FMUL y, y, y` |
| `rcp.approx.ftz.f32` | `MUFU.RCP` |
| `rcp.approx.f32` | s = (|x| > 2^126) ? 0.25 : (|x| < 2^-126) ? 2^24 : 1 (`FSETP`x2, `FSEL`x2); `MUFU.RCP(x*s) * s` |
| `rsqrt.approx.ftz.f32` | `MUFU.RSQ` |
| `rsqrt.approx.f32` | |x| < 2^-126: `MUFU.RSQ(x * 2^24) * 4096` |
| `sqrt.approx.ftz.f32` | `MUFU.SQRT` |
| `sqrt.approx.f32` | |x| < 2^-126: `MUFU.SQRT(x * 2^24) * 2^-12` |
| `lg2.approx.ftz.f32` | `MUFU.LG2` |
| `lg2.approx.f32` | |x| < 2^-126: `MUFU.LG2(x * 2^24) + (-24)` |
| `sin/cos.approx[.ftz].f32` | `FMUL.RZ x, 0.15915493667125701904` (1/(2 pi)), `MUFU.SIN/COS` (same code with and without .ftz) |
| `tanh.approx.f32` | `MUFU.TANH` |
| `ex2.approx.f16`, `ex2.approx.ftz.bf16` | `MUFU.EX2.F16` / `.BF16` (high half of the result 0) |
| `ex2.approx.f16x2`, `ex2.approx.ftz.bf16x2` | `MUFU.EX2.F16x2` / `.BF16x2` |
| `tanh.approx.f16/bf16` | `MUFU.TANH.F16/.BF16`; the x2 forms are two of them (`R.H1` for the high lane) + `PRMT` |
| `rcp.approx.ftz.f64` | `MUFU.RCP64H` (not modeled) |
| `div.full.f32` | `MUFU.RCP` of the (scaled) divisor times the dividend (not lowered from PTX by ptxv) |

The simulator models each PTX op directly as that expansion (`mufu::ptx_approx`), so the PTX
front-end computes what the ptxas binary computes; the SASS front-end executes the MUFU and the
surrounding instructions themselves.

## Results: every reconstructed op matches GB300 on every input

All ops below were swept over all 2^32 inputs (16-bit forms: all 2^16; packed x2 forms: all
2^32 pairs, lane-wise equal to the scalar form) and `ptxv mufu-sweep` found **0 mismatches**
against the model. Error against the true function (f64 libm reference), maxima over the sweep:

| op | algorithm | max error |
|---|---|---|
| `MUFU.EX2` | x -> fixed point q = trunc(\|x\| 2^23) (negative: one's complement unless the fraction is 0); 2^frac by a 64-segment quadratic interpolator (n = 17); exponent = floor part; results < 2^-126 flush to +0 | 2.055 ulp (normal results) |
| `MUFU.RCP` | 128 segments of the mantissa (n = 16); exact power-of-two results; ftz in and out | 0.975 ulp |
| `MUFU.RSQ` | 128 segments over [1, 4) (exponent parity + 6 mantissa bits, n = 17); x = 4^k exact | 1.515 ulp |
| `MUFU.SQRT` | same indexing as RSQ, C2 at half-ulp resolution | 0.944 ulp (normal inputs) |
| `MUFU.LG2` | log2(m) as a 36-fraction-bit fixed point (64 segments), plus the exponent; negative sums negated in one's complement; truncated to 24 bits; log2(1) = 0 | 1 ulp + 2.62 * 2^-24 |
| `MUFU.SIN/COS` | x in revolutions truncated to 25 fraction bits; quarter-wave table (64 segments), quadrants 1 and 3 mirrored in one's complement; 37-fraction-bit fixed point truncated to 24 bits | 1 ulp + 5.46 * 2^-24 (5.71 * 2^-24 absolute) |
| `sin/cos.approx.f32` | the bare unit after `FMUL.RZ x, 1/(2 pi)` | (8 + 2.7 \|x\|) * 2^-24 for \|x\| <= 1024 (5.83 / 6.78 * 2^-24 for \|x\| <= pi) |
| `.F16/.BF16/x2` | the f32 unit, then truncation (f16: \|y\| >= 2^16 -> inf, subnormals kept; bf16: drop the low half); NaN -> 0x7fff | |
| `MUFU.TANH` | \|x\| < 2^-8: x (denormals included); \|x\| >= 8: +-1; NaN: canonical NaN. For \|x\| in [2^e, 2^(e+1)), e = -8..2, only the top K = 17,17,17,18,18,19,19,19,20,21,20 mantissa bits are read; the top K-17 of them pick one of 51 segments, the low 17 bits k enter S = C0 + C1·k·2^17 + C2·sq19(k) (units 2^(e-66)); result = S truncated to 24 bits, sign mirrored. Fit: `fits/fit_tanh.json`. 0 differences on all 2^32 inputs, no exception table | 134 ulp |
| `MUFU.RCP64H` | on the high word of an f64 (the low word is not an input): mantissa = the f32 `MUFU.RCP` of 1.m (20 bits, three zeros below) truncated to 20 bits, exponent rebuilt for the f64 range; zero / denormal -> +-inf, inf -> +-0, results < 2^-1022 -> +-0, NaN -> 0x7fffffff (low word 0) | 16.6 f32-ulps of 1/x (x the whole f64) |
| `MUFU.RSQ64H` | likewise from the f32 `MUFU.RSQ` of 1.m * 2^p (p = exponent parity); +-0 / denormal -> +-inf, negative -> NaN, +inf -> +0 | 9.1 f32-ulps |

Every interpolator is S = C0 2^a + R + C1 k 2^b + C2 sq(k) 2^c, result floor(S / 2^sh), with
sq(k) the square of the low `n` bits from a truncated squarer that drops every partial-product
bit of weight below 2^(2n-15). Tables: C0 26 bits, C1 16 bits, C2 10-14 bits (Oberman & Siu,
ARITH 2005). The fits are exact: each segment's (C1, C2) and C0 window were solved from all
2^17 (2^16) outputs of the segment, then the whole op was replayed over 2^32 inputs.

## TANH reconstruction

The input is truncated per binade (K bits, table above), so each binade is one interpolator
of the usual form on 2^(K-17) segments. A least-squares fit of (C1, C2) per segment is biased
by the truncated squarer and misses the exact coefficients by more than a few units; solving
an LP on the basis [1, k, k^2, sq19(k) - k^2] for each segment's feasible (C0, C1, C2) and then
searching integers around the LP point (C1 on a per-binade grid: 2^9 .. 2^32; C2 often a round
number such as -2^17) gives coefficients for all 51 segments, including the 2^-8 binade and the
segments next to saturation. C0 is the multiple of the largest power of two in its feasible
window. No exception table is needed.

## f64: MUFU.RCP64H / RSQ64H

`rcp.approx.ftz.f64` and `rsqrt.approx.ftz.f64` are the bare units on the input's high word
(`MUFU.RCP64H R3, hi`; low result word `R2 = 0`), so a sweep of all 2^32 high words covers every
f64 input; `probes/mufu/kernels.cu` `d_*` also feeds a hashed low word, which does not reach the
unit. Both match the model above on all 2^32 high words: the 64H units reuse the f32 tables
(RCP64H = the f32 RCP mantissa of 1.m truncated to 20 bits; RSQ64H likewise from RSQ).

ptxas's `div.rn.f64`, `sqrt.rn.f64` and `rsqrt.approx.f64` (non-ftz) seed Newton iterations with
the same units (`DFMA`/`DMUL`/`DADD`/`DSETP`, plus a slow-path subroutine); the units are modeled,
but those sequences are not yet executable: the SASS front-end has no f64 arithmetic (`DFMA`
etc.), and the PTX front-end has no `div/sqrt.rn.f64`.

## What the error bounds give an online-softmax proof (for equivalence work)

Facts from the exhaustive sweeps (f32, GB300; "ulp" = ulp of the true result,
2^(floor(log2 |f|) - 23)):

- **EX2** (`ex2.approx[.ftz].f32`, bare `MUFU.EX2`): |hw - 2^x| <= 2.056 ulp(2^x) for normal
  results, i.e. relative error eps_e <= 2.056 * 2^-23 = 2.45e-7. `.ftz` / the bare unit flush
  results below 2^-126 to +0 (absolute error < 2^-126); `ex2.approx.f32` keeps denormal results.
  **Exact points:** ex2(+-0) = 1 exactly; ex2(k) = 2^k exactly for every integer k in
  [-126, 127] (non-ftz also [-149, -128]; -127 is not exact). **Monotone:** non-decreasing over
  all inputs (checked on all 2^32), so x <= 0 gives ex2(x) <= 1 and the row maximum maps to
  exactly 1.
- **RCP** (`rcp.approx[.ftz].f32`, `MUFU.RCP`): relative error eps_r <= 0.975 * 2^-23 on
  normal inputs; powers of two exact; monotone (non-increasing on each sign). `div.full.f32`
  is a * rcp(b) (with the same power-of-two scaling for |b| > 2^126 or < 2^-126): relative
  error <= eps_r + 2^-24 (one FMUL rounding).

Online softmax over x_1..x_N, scale c = log2(e) folded into the argument:
m_j = max(m_{j-1}, x_j), l_j = l_{j-1} * ex2((m_{j-1} - m_j) c) + ex2((x_j - m_j) c),
y_i = ex2((x_i - m_N) c) * rcp(l_N) (or divided by l_N).

1. **Max is exact.** max is exact and order-independent; m_N equals the two-pass max bit for
   bit, so every final exponent argument (x_i - m_N) c is formed by the same f32 ops in both
   schedules when the kernel computes it the same way (FFMA x*c - m*c vs FADD then FMUL: the
   argument roundings must match; if they differ, bound them as below).
2. **Argument rounding.** t = fl(fl(x - m) * c) (or FFMA(x, c, -m c)) has |t - t_exact| <= 2 u |t|
   (+ |m c| u for the FFMA form, from rounding m c), u = 2^-24. Since 2^(t + d) = 2^t (1 + d ln 2 + ...),
   the relative error from argument rounding is <= ln2 * |d|; for t <= 0 the absolute effect on
   p = 2^t is <= ln2 * 2u * |t| 2^t <= 2u / e (maximum of |t| 2^t ln2 is 1/e), i.e. at most
   0.74 * 2^-24 absolute per term, independent of N.
3. **Each exponential term:** p_i = 2^(t_i) (1 + th_i), |th_i| <= eps_e + ln2 |d_i| (+ the flush
   to 0 below 2^-126 for `.ftz`: absolute 2^-126 per term).
4. **Rescaling.** In the online form a term added at step j is multiplied by the rescale factors
   a_k = ex2((m_{k-1} - m_k) c) of every later max update k (and one FMUL/FFMA rounding each).
   MUFU is not multiplicative: ex2(a) * ex2(b) != ex2(a + b) bitwise, but each factor is within
   (1 + eps_e)(1 + u) of its exact value, and a_k = 1 exactly when the max does not change
   (ex2(0) = 1, and FMUL by 1 is exact). So with R_i rescales after term i,
   l_N(online) = sum_i 2^(x_i - m_N) c * (1 + e_i), |e_i| <= (R_i + 1)(eps_e + u) + ln2 |d| terms
   + gamma_{N-1} (the recursive f32 sum; pairwise/tree sums in the two-pass form give
   gamma_{ceil(log2 N)}). The two-pass form has R_i = 0. Since all terms are positive, the relative
   error of l is at most the maximum e_i: |l_online / l_exact - 1| <= (R_max + 1)(eps_e + u) +
   gamma_{N-1} + 2 ln2 u max|t|, and the same with R_max = 0 for two-pass.
5. **Normalization.** y_i = p_i * rcp(l): relative error <= th_i + eps_l + eps_r + u (with
   `div.full`: + the same). Online vs two-pass outputs then differ by at most
   |y_online - y_twopass| <= y_exact * (eps_l(online) + eps_l(twopass) + 2 (eps_r + u)) when both
   use the same p_i bits (else add 2 th_i), with y_exact <= 1.
6. **Numbers.** eps_e = 2.056 * 2^-23, eps_r = 0.975 * 2^-23, u = 2^-24. For N = 1024 and up
   to R rescales: eps_l ~ (R + 1) * 2.56 * 2^-23 + 1023 * 2^-24 (recursive sum dominates).

These are exactly the constants in `mufu::error_bound` (`ASSUME(N-MUFU)`), which equiv.rs
already attaches to each MUFU pseudo-leaf; the extra facts an online-softmax proof needs beyond
them are the exact points and monotonicity above (ex2(0) = 1 makes non-updating rescales exact,
so R counts only true max changes).
