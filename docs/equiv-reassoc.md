# Equivalence of non-identical implementations (`ptxv equiv`, after rung 2e)

**Status (2026-11-03).** `ptxv equiv` now proves equivalence, with a worst-case bound per output,
for implementations that compute the same operation in different orders or with different
algorithms. Proven pairs (the norms and the attention core added later the same day, §4–5):

| reference | candidate | level | bound per output, every input | random trials: largest fraction of the bound used |
|---|---|---|---|---|
| torch.cat | Triton cat | **L0** (bitwise) | 0 | 0 (all bitwise equal) |
| torch.sum(x, -1) | Triton butterfly row sum | **L1** | 15u · Σ\|x\| | 0.19 (sim), 0.18 (GB300) |
| torch.sum(x, -1) | Triton split-K row sum (2 launches) | **L1** | 15u · Σ\|x\| | 0.19 (sim), 0.19 (GB300) |
| torch.sum(x, -1) | Triton fp32-atomic row sum (every order) | **L1** | 16u · Σ\|x\| + 1.4e-37 | 0.18 (sim), 0.17 (GB300) |
| torch.cumsum(x, -1) | Triton tl.cumsum | **L1** | ≤ 15u · Σ\|x\| (per output) | 0.70 (sim), 0.63 (GB300) |
| torch.softmax(x, -1) | Triton two-pass softmax | **L1** | 44.95u · y + 1.24u (≤ 46.19u) | 0.068 (sim), 0.076 (GB300) |
| torch.softmax(x, -1) | Triton online softmax | **L1** | 53.85u · y + 1.24u (≤ 55.08u) | 0.061 (sim), 0.072 (GB300) |
| F.rms_norm (f32) | Triton rmsnorm | **L1** | 17.69u · \|y\| | 0.30 (sim), 0.33 (GB300) |
| F.layer_norm (f32, Welford) | Triton two-pass layernorm | **L1~** | (4.68 + 26.55 κ)u · \|w z\| + 27.01u · T, κ ≤ 498; exact per-input form beyond | 0.29 (sim), 0.40 (GB300) |
| attention core, scores given, normalize-first (Triton) | flash form (Triton, rescaled accumulator) | **L1** | 83.21u · V1 + 18.60u · Vmax | 0.051 (sim), 0.063 (GB300) |
| attention f32, softmax(q kᵀ) v, normalize-first (Triton) | flash form (Triton) | **L1** | (87.21 + 24.02 Q)u · V1 + 18.60u · Vmax | 0.023 (sim), 0.026 (GB300) |
| torch.mm bf16 (cuBLAS nvjet, tcgen05) | Triton bf16 matmul, K reversed | **L1** | 2^-7 \|y\| + 59.73u · Σ\|ab\| | 0.51 (sim), 0.96 (GB300) |
| torch SDPA flash bf16 (FA2, sm80 HMMA) | Triton flash attention bf16 (tcgen05) | **L1** | (262913.25 + 111.88 Q)u · V1 + 9.93u · Vmax | 0.34 (sim), 0.36 (GB300) |

Elementwise kernels (gelu, silu, tanh-GELU, ...), for which no normal form exists, are proven
in elementwise mode instead: every output is one input element's function, swept over every
f32 or bf16 value (docs/equiv-elementwise.md). bf16 / f16 inputs are numerics leaves too
(N-HALF-LEAF), so bf16 sums get L1 bounds as well.

u = 2^-24; Σ|x| is the sum of the magnitudes of the terms an output adds; y is the exact softmax
output (≤ 1). For the norms y, w z (the normalized value times its weight), κ and T are in §4;
for attention V1 = Σ_j p_j |v_j| ≤ Vmax = max_j |v_j| and Q = max_j Σ_d |q_d k_jd| (§5). All
softmax / sum shapes are 4x128 f32 (the cat: two 8x32); the norms 4x128 f32, the attention core
s 16x64, v 64x16 f32, the f32 attention q 16x16, k and v 64x16. L1~:
the same function up to rounded constants (§4). Fixtures: `fixtures/c_equiv_cat`,
`c_equiv_sum`, `c_equiv_cumsum`, `c_equiv_softmax`, `c_equiv_rmsnorm`, `c_equiv_layernorm`,
`c_equiv_attn` (equiv.toml and full.toml); `cargo test` runs them all
(`crates/cli/tests/equiv.rs`). The bf16 tensor-core
pairs (the last two rows: MMA nodes, bf16 provenance and roundings) are in docs/equiv-mma.md;
§5 lists what blocked them and how each blocker was removed.

The levels L0–L3 are rung 2e's (docs/rung2e.md). New here: the bound each output achieves (not
only a tolerance to check), its form, and a machine-readable summary.

## 1. Leaves through plain moves (N-LEAF)

Before, the numerics shadow gave a value an identity only when an FP operation touched it, so a
pure data-movement kernel had "untracked" outputs and equivalence stopped at "L2 (unproven)".

- A 4-byte load of an f32 input word that nothing has written and that still holds its initial
  bits is an `Input` leaf `(buffer, element)`; vector loads give each register its word.
- Moves keep the leaf: registers, shared and global memory, `SHFL` (all modes; it used to drop
  the value's history), `MOV`, `IMAD.MOV` (ptxas's `IMAD d, RZ, RZ, c`), and `SEL`/`FSEL` with a
  predicate that does not depend on input data.
- Two latent bugs fixed on the way: vector loads, `SEL` and 64-bit stores left stale nodes in
  registers or memory words they overwrote; now every destination an instruction writes without a
  numerics hook loses its node.

Result: torch's `CatArrayBatchedCopy_vectorized` and the Triton cat compute every output as the
same input leaf: L0 for every input, PROVEN.

## 2. Reassociated sums (N-REASSOC, N-ATOMIC)

### The bound

Each output's canonical form is its exact polynomial over input leaves (rung 2e). Two
reductions of the same multiset of terms have the same polynomial (L1), whatever their trees.
For the bound, each monomial m also carries, summed over its occurrences o,

  g_m = Σ_o |c_o| (Π_{v on o's path} (1 + ε_v) − 1),

with ε_v = u for a round-to-nearest add, 2u toward zero, the MMA contract's ε for a tensor-core
step. This is fp_noise Cor 1a without the first-order approximation, so

  |K(x) − E(x)| ≤ Σ_m |x_m| g_m   (every input x, no overflow)

for each kernel K against the exact value E. Thm 4 combines the two kernels:

  |C(x) − R(x)| ≤ Σ_m |x_m| (g_R,m + g_C,m) ≤ k u Σ_m a_m |x_m|,

with a_m = Σ_o |c_o| (1 for a plain sum) and k = max_m (g_R,m + g_C,m) / (a_m u), rounded up by
2^-40 for its f64 evaluation. k is the per-output bound `ptxv equiv` reports. Nothing compares
the trees: linear, pairwise, warp-shuffle butterflies, vectorized partial sums and split-K all
fall out of the per-monomial depths. Subnormal sums of nearest adds are exact, so there is no
absolute term (FTZ atomics have one, below).

### Split-K

`c_triton_rowsum_splitk` (tools/capture_examples/triton_rowsum_splitk.py): one launch writes 4
partial sums per row into a workspace, a second launch sums them. The workspace words carry the
partials' nodes from the first launch to the second, so the output is one tree: depth 5 in the
first launch, 2 in the second.

### fp32 atomics

`c_triton_rowsum_atomic` (triton_rowsum_atomic.py): 4 programs per row each add their partial
sum with `tl.atomic_add`, i.e. `REDG.E.ADD.F32.FTZ.RN.STRONG.GPU`.

- **Model.** `atom/red.add.f32` is one indivisible round-to-nearest add with FTZ (new `AtomOp::FAdd`
  in the IR, both front-ends, exec). The numerics chain the adds on a location (`AtomAdd`).
- **Every order at once.** The chain's order is schedule-dependent. In any order of S adds onto a
  location that starts at +0, the first add is exact, so each value sits under at most S − 1
  roundings (S with a nonzero start): `equiv` gives every value that worst case (fp_noise Cor 4b),
  plus 3·2^-126 absolute per add for FTZ. The bound therefore holds for every order, not only the
  ones the explorer visited. L0 is refused (the bits depend on the order).
- **Exploration.** CTAs share nothing but the atomic adds, but a flat exploration of 16 CTAs
  interleaving their atomics did not finish in 20M states. `--slices` now files bytes that only
  fp32 atomic adds without a returned value touch as reductions (`GFoot::red`): two clusters'
  reductions into a byte do not conflict (any other access by another cluster still does), and the
  verdict says the byte's final value depends on the order and reports the launch as
  nondeterministic (16 slices, 96 states). `equiv` accepts "nondeterministic" only when every
  atomic in the run is such an add.

### Results

`ptxv equiv fixtures/c_equiv_sum/equiv.toml`, `.../c_equiv_cumsum/equiv.toml` (reference torch):

| candidate | per-kernel bound (reference, candidate) | bound k (u · Σ\|x\|) | trials: max used | observed k |
|---|---|---|---|---|
| Triton butterfly row sum (`r4_rowreduce_triton`) | 8u, 7u | 15 | 0.19 | 2.71u |
| Triton split-K (`c_triton_rowsum_splitk`) | 8u, 7u | 15 | 0.19 | 2.72u |
| Triton fp32 atomics (`c_triton_rowsum_atomic`) | 8u, 8u | 16 (+1.4e-37) | 0.18 | 2.73u |
| Triton tl.cumsum (`c_triton_cumsum`; vs torch.cumsum) | up to 7u, 8u | 0 to 15 by output | 0.70 | 3.93u |

(200 trials each.) The trials (`--trials N`) draw N inputs (even trials normal, odd uniform in [1, 2), where
rounding errors share a sign and accumulate), run both kernels in the simulator on a random
schedule (so the atomic orders vary), and check every output's |C − R| against its bound
evaluated at that input (Σ_m |x_m| (g_R,m + g_C,m)). "Max used" is the largest |C − R| / bound,
"observed k" the largest |C − R| / (u Σ|x|). Each kernel is also checked against the exact sum.
Random inputs use little of a worst-case bound (errors of a sum of n terms grow like √n, the
bound like the depth); the cumsum's first outputs, short sums, come closest (each kernel alone
reaches 0.98 of its own bound there: x0 + x1 rounded once).

**On GB300.** `tools/equiv_gpu_check.py` runs the same torch and Triton kernels on the GPU
(2000 inputs per pair, bounds from `equiv.json`): no output exceeds its bound; the largest
fractions used are 0.18 (butterfly), 0.19 (split-K), 0.17 (atomic), 0.63 (cumsum). One input
run 500 times through the atomic kernel gave 4 distinct outputs (GB300's atomic orders), all
within the bound; `ptxv hw` saw 2 distinct outputs in 2000 runs of the captured input.

## 3. Softmax (N-SEG, N-SOFTMAX)

### Segment mode

`segments = true` in an equiv.toml turns on a numerics mode for kernels beyond sums.

- **More nodes.** Products (`FFMA`/`FMUL`, nearest, `-x` modifiers allowed), negated adds, and exact
  maxima (`FMNMX`, or `FSETP` + `SEL` choosing between the compare's operands: torch's max) get
  nodes. They do so only when they combine values that depend on different tracked nodes: x − max
  is a node, while an exp idiom's `FFMA` steps on its own argument are not.
- **Slices.** Every other input-derived value is materialized when a tracked operation consumes
  it or it is stored, as an instruction slice: a backward slice along the thread's executed trace
  from tracked nodes (its arguments) and constants (`seg.rs`). Control flow does not depend on
  input data, so the trace is every input's.
- **Data-dependent guards.** A guard that depends on input data (Triton's `@!P FMUL t, t, 0.5`
  for the subnormal range of ex2, div.full's range scaling) is a `Guard` op in the slice: both
  outcomes. Its taint violation is excused because the slice covers it ("relies on: N-SEG").
- **div.rn.** ptxas's `div.rn.f32` (fast path, `FCHK`, a call of the slow path) is one opaque
  correctly rounded quotient. Registers its slow path writes must be dead afterwards.

A slice is evaluated at any argument by re-executing its instructions with exec.rs's semantics.

### Normal form

`softmax.rs` puts an output into the form exp(L_i) / Σ_j exp(L_j):

- **Exp arguments.** L is an exact linear form over input leaves and maxima (a max is identified by
  its set of inputs). It must be provably ≤ 0: x − max(X) with x ∈ X, max(A) − max(B) with A ⊆ B,
  or −inf.
- **Telescoping.** Products of exponentials add their arguments, so online softmax's
  l · exp(m_old − m_new) + s telescopes to Σ_j exp(x_j − max(X)): the running maxima cancel.
- **Slice contracts.** Each one-argument slice is identified and bounded by evaluating it on
  every f32 argument of its domain:
  - exp: all 2^31 − 2^23 + 2 values d ≤ 0, against `f64::exp`;
  - 1/x: every f32 in the divisor's proven range.
  The contract is |g(d) − e^d| ≤ (r0 + r1|d|) e^d where e^d ≥ 2^-126, ≤ a below.
- **Range simplification.** Before identification, predicates comparing the divisor S with a
  constant are decided from S's proven range [S_lo, S_hi]. This removes div.full's scaling paths
  and torch's `FSEL q, NaN, S != 0`.
- **Same function.** Two kernels compute the same function when their normal forms are equal:
  same numerator argument, same denominator terms.

| kernel | max | exp (slice contract) | sum | division |
|---|---|---|---|---|
| torch `softmax_warp_forward` | FSETP.GT/GEU + FSEL, SHFL butterflies | expf: FFMA.SAT/FFMA.RM range reduction, MUFU.EX2, SHF exponent: r0 = 2.36u, r1 = 0.110u/\|d\| | per-thread 4 terms + 5 butterflies | div.rn (u) |
| Triton two-pass | FMNMX trees | x·log2e, ex2 with the subnormal-range guard: r0 = 2.63u, r1 = 1.20u/\|d\| (log2e rounded to f32) | butterflies + smem | div.full: MUFU.RCP slice r0 = 1.65u, then FMUL (u) |
| Triton online | running FMNMX3 | same ex2 idiom, also for exp(m_old − m_new) | 4 blocks: l = l·exp(m_old − m_new) + s (FFMA) | div.full |

The contracts come from the bit-exact MUFU model (N-MUFU, agent B's exhaustive sweeps) composed
with the instructions around the unit. They agree with B's per-unit bounds:
- EX2 ≤ 2.056·2^-23: the Triton ex2 idiom's r0 = 2.63u = 1.32·2^-23 on |d| ≤ 1; its r1 is the
  f32 rounding of log2 e times |d|;
- RCP ≤ 0.975·2^-23: div.full's 1/x slice r0 = 1.65u = 0.82·2^-23 on the divisor's range [≈1, ≈130];
- ex2(0) = 1 exactly, so the computed term of the maximum is exactly 1. The theorem itself uses
  only the exact S ≥ 1.

### The theorem

Per kernel and output i, with p_j = exp(L_j)/S and n terms:

- **Terms.** Each computed term is exp(L_j)·φ_j + α_j, with
  φ_j ∈ [lo_j e^{−s|L_j|}, up_j e^{s|L_j|}] and |α_j| ≤ A_j:
  - up_j / lo_j multiply (1 ± r0) per exp slice and (1 ± u) per rounding on the term's path;
  - s collects the slices' r1 and the arguments' rounding (|fl(L) − L| ≤ u|L|);
  - the exponents add along a product because every argument is ≤ 0, so |L_a| + |L_b| = |L_a + L_b|.
- **Facts used.**
  - S ≥ 1: the term of the maximum is exp(0).
  - Σ_j p_j |L_j| ≤ H(p) ≤ ln n.
  - e^{−t}(e^{st} − 1) ≤ s/(e(1 − s)).
- **Denominator.** Hence Ŝ/S ∈ [S_lo, S_hi], with
  - S_lo = min_j lo_j (1 − s ln n) − A_S;
  - S_hi = max_j up_j (1 + s ln n e^{1024 s}) + n e^{−1000} + A_S.
- **Bound.** With the division's relative error in [ρ_lo, ρ_up]:

  |ŷ_i − y_i| ≤ K_rel y_i + K_arg + K_abs, where
  - K_rel = max(up_i ρ_up / S_lo − 1, 1 − lo_i ρ_lo / S_hi);
  - K_arg = max(up_i ρ_up / S_lo, 1) · s / (e(1 − s));
  - K_abs = A_i ρ_up / S_lo + (the division's absolute term).

Thm 4 adds the two kernels' bounds. The argument roundings cost an additive K_arg rather than a
relative term: an element far below the maximum has a large relative error but a tiny value
(y |L| ≤ 1/e). Inputs must be finite.

### Results

`ptxv equiv fixtures/c_equiv_softmax/equiv.toml` (reference torch):

| candidate | reference K_rel + K_arg | candidate K_rel + K_arg | bound | trials: max used | each kernel vs exact |
|---|---|---|---|---|---|
| Triton two-pass (`r4_softmax_triton`) | 19.61u | 26.59u | 44.95u · y + 1.24u (≤ 46.19u) | 0.068 (\|C − R\| ≤ 0.25u) | 0.12 / 0.09 |
| Triton online (`c_triton_softmax_online`) | 19.61u | 35.48u | 53.85u · y + 1.24u (≤ 55.08u) | 0.061 (\|C − R\| ≤ 0.75u) | 0.12 / 0.08 |

(200 simulator trials; "max used" is |C − R| over its bound evaluated at the input, about
(45 y + 1.24)u with y ≈ 1/128 here.) On GB300 (`tools/equiv_gpu_check.py`, 2000 inputs) the
largest fractions are 0.076 (two-pass) and 0.072 (online), none above 1.

The sweeps take ~26 CPU-minutes (7 distinct slices; ~30 s on 144 threads). Their results are
cached in `fixtures/c_equiv_softmax/sweeps.json`, keyed by each slice's instructions;
`--no-sweep-cache` recomputes them.

### What is not covered

- **Tighter online bound.** The online bound charges every rescale with an exp contract and a
  rounding. A rescale where the maximum does not change is exact: ex2(0) = 1, and the products
  and sums by 1 are exact (agent B's MUFU notes). But which rescales those are depends on the
  input: for every block after the first there are inputs whose maximum grows there, so no
  rescale is provably free for every input and the worst case stands. (The first is: m_old =
  -inf makes the rescaled sum an exact zero, which the normal form already uses.) A sound
  refinement exists only per input, in the evaluated bound (count the rescales where m grows
  at that x); it does not change the proven k and needs per-path records the term records
  merge, so it was not done.
- **L2 and L3** (no less accurate, a fixed tolerance) are not computed for the softmax form; the
  per-output bound replaces them.
- **Other transcendental shapes:** layer_norm, rms_norm (§4) and attention (§5).
- **The div.rn slow path** is never simulated. Its correctness is PTX's definition of
  `div.rn.f32` (N-SEG).

## 4. Layer norm and RMS norm (N-NORM)

### The kernels (recaptured in f32)

The rung-4 norm fixtures are bf16 in and out; equivalence leaves are f32 words (N-LEAF), and a
bf16 output rounding (2^-9) would swamp every other error. So all four kernels are recaptured
at 4x128 f32 (`tools/capture.py`; Triton sources `tools/capture_examples/triton_{layernorm,
rmsnorm}_f32.py`, the rung-4 kernels without the bf16 casts). All four match GB300 bitwise.

| kernel | mean | variance / mean square | rho | output |
|---|---|---|---|---|
| torch `vectorized_layer_norm_kernel<float>` (layer) | per-thread Welford (FFMA with the f32 constants 1/2, 1/3, 1/4), Welford combines over shuffles and shared memory | Welford M2, div.rn by n | + eps, rsqrt.approx.f32 (guarded MUFU.RSQ) | FFMA(w, rstd · (x - mean), b) |
| Triton `layernorm_kernel` | butterfly sum; d = FFMA(-S, 1/n, x) (the mean never rounded on its own) | sum of d², FFMA(S2, 1/n, eps) | MUFU.SQRT, then div.full 1/x | FFMA(d · rstd, w, b) |
| torch (rms) | — | FFMA chain + shuffles + shared memory, div.rn by n | + eps, rsqrt.approx.f32 | w · (rstd · x) |
| Triton `rmsnorm_kernel` | — | butterfly sum, FFMA(S, 1/n, eps) | bare MUFU.RSQ | (x · r) · w |

### Atomization

`norm.rs` turns each output into a polynomial P(x, rho) = Σ_m c_m x_m rho^e_m in the input
leaves and one nonlinear scalar rho per row:
- **Polynomial prefix.** The slice computing rho starts with round-to-nearest FFMA / FADD /
  FMUL and div.rn by a constant (torch's division of M2 by 128); those fold into A's polynomial
  with their roundings (a power-of-two divisor is exact). What remains reads one polynomial
  value A^ (range guards read scaled copies of it: the root is the value all others are
  computed from).
- **The power.** g(4) / g(1) picks p in {-1/2, -1, 1/2}; g is then evaluated on every positive
  normal f32 (2^31 - 2^24 values) against a^p: torch's rsqrt.approx.f32 r0 = 2.092u, Triton's
  bare MUFU.RSQ 2.092u, Triton's sqrt + div.full 2.585u (`fixtures/c_equiv_*norm/sweeps.json`).
- **Cor 1a with rho^ as a leaf.** Each kernel's output polynomial carries the rigorous
  multipliers g_m (N-REASSOC), rho^ entering as an exact leaf; A's polynomial carries its own.
- **The pivot.** The reference's exact polynomials (A_R, P_R) are the pivot. Welford's exact
  algorithm is not the layer norm: with the f32 constant 1/3 its mean's coefficients differ
  from 1/128 by up to 2^-19 relative. Every coefficient difference dc is charged as an error
  (|dc_m| |x_m|), and the level is reported as **L1~** (more than 2^-10 is refused as "not the
  same function").

Numerics changes on the way: a square x·x of one tracked node is a product node (sums of
squares; segment mode only anchored products of different nodes); a vector store (`STS.64`)
materializes each word (torch's broadcast of (mean, rstd) through shared memory);
structurally equal slice operations are merged (the slicer repeats an instruction read at two
trace positions).

### The theorem

Per output, with rho* = A_R(x)^p, P1_R the rho-linear part of P_R, and for kernel K
- e_K(x) = Σ_m |x_m| (g^A_K,m + |dc^A_m|) / A_R(x): A^_K's relative error against A_R (dc^A
  only for the candidate);
- eta_K = max over the corners of |(1 ± r0_K)(1 ± e_K)^p - 1|: rho^_K = rho* (1 + theta_K) with
  |theta_K| ≤ eta_K (contract and A's error);
- H = 1 + max_K eta_K ≥ |rho^_K| / rho*;

expand P_C(x, rho^_C) - P_R(x, rho^_R) = P1_R(x) (rho^_C - rho^_R) + Σ_m dc_m x_m rho^_C^e_m
and add Cor 1a of both kernels:

  |C - R| ≤ |P1_R(x)| rho* (eta_R + eta_C) + Σ_m |x_m| rho*^e_m H^e_m (g_R,m + g_C,m + |dc_m|),

valid when e_K ≤ 1/2 and A_R(x) ≥ 2^-125 (so A^ is a positive normal f32, inside the swept
domain). `ptxv equiv` evaluates this exactly at each trial input. The **compact form**
linearizes eta ≤ r0 + (1 + r0) c1 e with c1 the chord of |(1 ∓ e)^p - 1| on e ≤ 2^-10 (convex
or concave, so the chord or the tangent bounds it), and e_K ≤ k_A,K u κ with

- κ(x) = S_A(x) / A_R(x), S_A = Σ_m a_m |x_m| (A's conditioning; a_m = the larger sum of
  |coefficient| over the monomial's occurrences in the two DAGs, any positive weights being
  valid);
- T(x) = Σ_m a_m |x_m| rho*^e_m and |Y1| = |P1_R(x)| rho* ≤ T(x):

  |C - R| ≤ (rel + kappa_u κ) u |Y1| + add u T(x)   for κ ≤ κ_max = 2^-10 / (k_A u).

For the RMS norm A's monomials are squares with positive coefficients, so S_A = A and κ ≤ 1 for
every input; and P has the one monomial x_i w_i rho, so |Y1| = T = |y_i|.

### Results

`ptxv equiv fixtures/c_equiv_rmsnorm/equiv.toml`, `.../c_equiv_layernorm/equiv.toml` (reference
torch):

| pair | rel + kappa_u κ (u · \|Y1\|) | add (u · T) | per kernel (reference, candidate) | sim (200 trials) | GB300 (2000 inputs) | each kernel vs exact (GB300) |
|---|---|---|---|---|---|---|
| RMS norm | 4.18 + 9.51 κ, κ ≤ 1 proven | 4.00 | 9.10u, 8.60u (of 17.69u · \|y\|) | 0.30 (5.00u · \|y\|) | 0.33 (5.50u · \|y\|) | 0.48 / 0.48 |
| layer norm | 4.68 + 26.55 κ, κ ≤ 498 | 27.01 | 8220u, 5038u at κ = 498 | 0.29 | 0.40 | 0.86 / 0.96 |

("sim" and "GB300": the largest |C - R| over its bound evaluated at the input; the trials draw
normal inputs and, alternately, uniform in [1, 2) for x, w and b alike.) Every trial input is
covered by the exact form: κ reached 3704 on the uniform inputs (mean 1.5, σ 0.29), beyond the
compact form's 498, with A's relative error still far below 1/2.

- **RMS norm: tight to a factor ~3.** Both kernels are within half of their own bounds.
- **Layer norm: the exact form is within 2.5x of the worst observed; the compact form is loose.**
  Cor 1a's monomial form charges each pair x_j x_k of A with its occurrences' weight, and
  Welford's partial M2 enter with weight up to ~1/12 per pair of one thread's elements where
  the variance's coefficient is -2/n² (cancellation up to 700x). The multipliers are relative
  to those weights (k_A = 33u for torch, 20u for Triton including the coefficient
  differences), so the weight lands in κ = S_A / A: up to 3704 on the uniform
  ones, where the true conditioning Σx² / (n (V + eps)) is ~28. A bound relative to the partial
  M2 values themselves (a running error bound) would remove it; the monomial form cannot see
  that M2_a ≤ M2.
- **The output rounding** of y ≈ b (small w) is where both kernels come closest to their own
  bounds.

GB300 trials: `tools/equiv_gpu_check.py --dump DIR` runs both kernels of each pair on 2000
inputs and writes inputs and outputs; `PTXV_EQUIV_HW_TRIALS=DIR ptxv equiv FILE` checks them
against the same per-input bounds (no other change to the command).

## 5. Attention (N-ATTN)

### What is proven (f32, CUDA cores): the attention core

The part of attention that softmax does not already cover is the **rescaled accumulation of
P·V**: o = Σ_j softmax(s)_j v_j computed as a flash kernel does (running max m and sum l, the
accumulator multiplied by exp(m_old - m_new) whenever m grows, o = acc / l), against a
reference that normalizes P first. Pair (`fixtures/c_equiv_attn`, scores given as an input,
s 16x64, v 64x16 f32, both bitwise = GB300):

- reference `c_attn_core_ref` (tools/capture_examples/triton_attn_core_ref.py): p = exp(s - max)
  / Σ (div.full per element), o = Σ_j p_j v_j (FFMA and FADD2 trees);
- candidate `c_attn_core_flash` (triton_attn_core_flash.py): 4 blocks of 16 keys, l = l·α + Σ e,
  acc = acc·α + Σ e v (FFMA2), o = rcp(l) · acc.

**Weighted normal form** (`softmax.rs`, `WSum`, `wquot`): o = Σ_j exp(L_j) v_j / Σ_j exp(L_j),
L_j = x_j - max(X) as in §3, v_j input leaves:
- a weighted sum's terms are keyed by (L, v) and carry the §3 records (a signed sum's rounding
  multiplies all its terms by one (1 + δ); two occurrences of a term with opposite signs are
  refused); absolute errors are tracked as a0 + av max|v|;
- the flash form telescopes like online softmax: acc's terms are exp(x_j - m_k) v_j times the
  later rescalings exp(m_k - m_k+1) ..., i.e. exp(x_j - max(X)) v_j;
- the normalize-first form is a sum of v_j times exp(L_j) / S^ where every term divides by the
  same S^: per-thread copies of the row sum are recognized as one value by a structural hash
  (identical rounding DAGs round alike for every input), and each term carries its own
  division error.

**Theorem.** With the §3 notation (Ŝ/S ∈ [S_lo, S_hi], the division error in [rho_lo, rho_up],
r = (1 + rho) S / Ŝ ∈ [r_lo, r_hi] = [rho_lo / S_hi, rho_up / S_lo]),
o^ - o = Σ_j p_j v_j (phi_j r - 1) + (absolute part), and |phi_j r - 1| ≤ K_rel + up r_hi
(e^(s|L_j|) - 1), so

  |o^ - o| ≤ K_rel V1 + up r_hi b_up Vmax + K_abs,   V1 = Σ_j p_j |v_j|, Vmax = max_j |v_j|,

with K_rel = max(up r_hi - 1, 1 - lo r_lo) and b_up = s ln n e^(1024 s) + n e^-1000 ≥ Σ_j p_j
(e^(s|L_j|) - 1) (the entropy bound of §3). Thm 4 adds the two kernels.

| | bound | per kernel (reference, candidate) | sim (200 trials) | GB300 (2000 inputs) | each kernel vs exact |
|---|---|---|---|---|---|
| attention core | 83.21u · V1 + 18.60u · Vmax (+ 9.3e-44) | 43.51u, 58.30u | 0.051 (4.10u · Vmax) | 0.063 (5.12u · Vmax) | 0.09 / 0.09 |

The bound is loose by ~16x: it charges every term the worst rescale count, the row's worst
exp contract, and Vmax where the errors of a 64-term weighted sum mostly cancel.

### With the scores computed: softmax(q kᵀ) v in f32

`fixtures/c_equiv_attn/full.toml`: the same two forms with the scores computed in the kernels
(`c_attn_f32_ref`, `c_attn_f32_flash`: s_j = Σ_d q_d k_jd as FFMA trees, scale 1; q 16x16,
k and v 64x16 f32; both bitwise = GB300).

- **Score atoms.** A node that is a rounded polynomial of the inputs, used as an exp argument
  or a member of the max, is an atom keyed by its exact polynomial (Σ_d q_d k_jd): both kernels
  get the same atoms, so the normal forms compare as before. Every node of one kernel with that
  key must round alike (identical DAGs, structural hash): the atom is then one value, the
  kernel's computed score ŝ_K,j.
- **Composition.** Each kernel's theorem holds with its computed scores as the inputs: |o^_K -
  o(ŝ_K)| ≤ K_rel V1(ŝ_K) + .... With Δ_K = max_j |ŝ_K,j - s_j| ≤ max_j Σ_m |x_m| g_K,m (Cor 1a
  of the score), p_j(ŝ) / p_j(s) = e^(δ_j) / Σ_k p_k e^(δ_k) ∈ [e^-2Δ, e^2Δ], so V1(ŝ_K) ≤
  e^(2Δ_K) V1 and |o(ŝ_K) - o(s)| ≤ (e^(2Δ_K) - 1) V1, V1 at the exact scores:

  |C - R| ≤ Σ_K [K_rel,K e^(2Δ_K) V1 + K_arg,K Vmax + K_abs,K + (e^(2Δ_K) - 1) V1].

  `ptxv equiv` evaluates this exactly per input (the scores' exact values and Δ_K from their
  polynomials). The compact form, with Δ_K ≤ k_s,K u Q (Q = max_j Σ_d |q_d k_jd|) and
  e^(2Δ) - 1 ≤ 2.002 Δ for Δ ≤ 2^-10 (Q ≤ 2731 here), is (rel + kq Q) u V1 + add u Vmax.

| | bound | sim (200 trials) | GB300 (2000 inputs) | each kernel vs exact |
|---|---|---|---|---|
| f32 attention, scores computed | (87.21 + 24.02 Q)u · V1 + 18.60u · Vmax | 0.023 | 0.026 | 0.10 / 0.10 |

The score term dominates (Q ≈ 13 on normal inputs, ≈ 36 on uniform [1, 2) ones) and is the
loosest part: it charges every score its worst-case Cor 1a error, in the worst direction for
every weight at once.

### The bf16 tensor-core pair (torch SDPA flash vs Triton flash attention): now proven

**Update (2026-11-03): all five blockers below are removed, and the pair is L1 for all 8192
outputs** (docs/equiv-mma.md):
1. HMMA and tcgen05 steps are MMA nodes (`num_hmma`, `num_mma`) with the model's contract
   (≈ 10u per full step, `MmaModel::eps_terms`).
2. bf16 provenance is tracked per byte through registers, shared, global, local memory and
   TMEM (`halfin.rs`), including cp.async, LDSM(.T), STSM, SHFL, LDTM / STTM.
3. Computed operands: a product's factors are any tracked nodes. F2FP.BF16 of a computed value
   is a Round node (2^-8 relative, N-ROUND).
4. Scores from MMA nodes are score atoms. The exp2 scale stays outside them (base-2 atoms; the
   shifted form for the FFMA s c' - m c').
5. bf16 outputs are read per element, and the final rounding is peeled (N-ROUND).

The original analysis follows.

`ptxv equiv` on `r4_sdpa_flash_torch` (cuDNN-style sm80 flash: cp.async, LDSM, HMMA.16816) vs
`r4_attn_triton` (tcgen05: Q via TMEM, K/V by cp.async, UTCHMMA, LDTM) gets no numerics at all.
The blockers, in the order a proof would hit them:

1. **HMMA has no numerics hook.** `exec.rs` `Op::Hmma` computes with the calibrated model
   (N-HMMA) but writes every result with `num_clobber`: no MMA node. tcgen05 MMAs have nodes,
   but only with operand origins from TMA-loaded shared memory (`orig_at`).
2. **bf16 provenance.** Q, K, V are bf16: two elements per 32-bit register or word. The
   numerics track 32-bit words; cp.async (now word-tracked for f32), LDSM (non-trans words only;
   `.trans` reshuffles 16-bit halves) and the Triton kernel's LDG → STTM path for Q would need
   per-half origins for the MMA's product terms to be input monomials.
3. **Computed MMA operands.** P·V multiplies the computed probabilities by V. The MMA node's
   product terms are pairs of input origins; P needs terms (node × leaf), and F2FP.BF16.PACK (P
   rounded to bf16: relative 2^-9 per term) is not modeled in numerics or slices.
4. **Scores from MMA nodes.** Score atoms (above) take rounded polynomials of the inputs; with
   1–3 done, an MMA node's Cor 1a multipliers (N-MMA / N-HMMA ε) would give Δ the same way,
   and the scale (sm_scale · log2 e folded into the exp's FFMA) would have to stay out of the
   atom (a power-of-two scale is exact).
5. **bf16 outputs.** Both kernels store bf16: `ptxv equiv` reads outputs as f32 words (pairs of
   bf16 here), and the final rounding (2^-9 relative) would dominate the bound.

Two cheaper detours were tried and dropped:
- torch's f32 SDPA backends: efficient attention is `fmha_cutlassF_f32_aligned_64x64_rf_sm80`,
  which uses TF32 HMMA (192 HMMA), and lifting stopped at `LDL.U8`.
- `torch.softmax(s, -1) @ v` (softmax_warp_forward + `cutlass3x_sm100_simt_sgemm`): the 4-byte
  LDGSTS form is now lowered, but the sgemm's epilogue still faults on an unresolved address in
  its parameter block.

## 6. Report format

`ptxv equiv FILE [--trials N] [--per-output] [--no-sweep-cache]` prints, per candidate:

- the levels;
- per output (grouped into runs, or every output with `--per-output`) the level and the bound
  achieved: `k u * sum|terms|` for sums, `rel u * y + add u` for softmax, `(rel + kappa_u
  kappa) u * |Y1| + add u * T` for norms, `rel u * V1 + add u * Vmax` for attention;
- the overall bound;
- with `--trials` (or `trials = N`), the empirical check.

The JSON summary goes to `<out>/equiv.json` (`--out`, default `$TMPDIR/ptxv-equiv-<fixture dir>-<file stem>/`). It
is a list of candidates, each with:
- `reference`, `candidate`, `strongest`, `proven`, `preconditions`, `relies_on`, `max_bound_u`;
- `report`: `levels`, `per_output`, `untracked`, `unordered`, `softmax`. Each `per_output` entry
  has `index`, `level`, `bound_u`, `ref_u`, `cand_u`, `abs_extra`, `form`, and for softmax also
  `rel_u`, `add_u`, `sm_ref`, `sm_cand`; for norms `norm` (`rel_u`, `kappa_u`, `add_u`,
  `kappa_max`, `kappa_le_1`, `ref_part`, `cand_part`, `coef_diff`, `p`, `contracts`), and
  `bound_u` is then the coefficient of u T(x) valid for every covered input; for attention
  (`form` = "attn") `rel_u` (of V1) and `add_u` (of Vmax), `bound_u` = their sum;
- `trials`.

equiv.toml additions: `trials`, `segments`, `sweep_cache`, and per candidate `expect_bound_u`.
The largest per-output bound must not exceed `expect_bound_u`. `expect = "L1~"`: every output
has a bound, the exact polynomials differing by rounded constants (norm form).
`PTXV_EQUIV_HW_TRIALS=DIR` replaces the simulator trials by GPU inputs and outputs
(`tools/equiv_gpu_check.py --dump DIR`).

## Reproduce

```sh
ptxv equiv fixtures/c_equiv_cat/equiv.toml --no-hw
ptxv equiv fixtures/c_equiv_sum/equiv.toml --no-hw --trials 200
ptxv equiv fixtures/c_equiv_cumsum/equiv.toml --no-hw --trials 200
ptxv equiv fixtures/c_equiv_softmax/equiv.toml --no-hw --trials 200 [--no-sweep-cache]
ptxv equiv fixtures/c_equiv_rmsnorm/equiv.toml --no-hw --trials 200
ptxv equiv fixtures/c_equiv_layernorm/equiv.toml --no-hw --trials 200
ptxv equiv fixtures/c_equiv_attn/equiv.toml --no-hw --trials 200
ptxv equiv fixtures/c_equiv_attn/full.toml --no-hw --trials 200
TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/equiv_gpu_check.py --trials 2000 --dump /tmp/e-hw
PTXV_EQUIV_HW_TRIALS=/tmp/e-hw ptxv equiv fixtures/c_equiv_layernorm/equiv.toml --no-hw   # (and rmsnorm, attn, attn/full)
ptxv check fixtures/c_triton_rowsum_atomic --slices
# captures (GPU 3):
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py --triton tools/capture_examples/triton_cumsum.py:cumsum_kernel --binary --out fixtures/c_triton_cumsum
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py --triton tools/capture_examples/triton_rowsum_atomic.py:rowsum_atomic_kernel --binary --out fixtures/c_triton_rowsum_atomic
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py --file tools/capture_examples/triton_rowsum_splitk.py --binary --out fixtures/c_triton_rowsum_splitk
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py --triton tools/capture_examples/triton_softmax_online.py:softmax_online_kernel --binary --out fixtures/c_triton_softmax_online
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py --op 'F.layer_norm(x, (128,), w, b, 1e-5)' --inputs 'x=randn(4,128,f32),w=randn(128,f32),b=randn(128,f32)' --out fixtures/c_layernorm_torch_f32
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py --op 'F.rms_norm(x, (128,), w, 1e-6)' --inputs 'x=randn(4,128,f32),w=randn(128,f32)' --out fixtures/c_rmsnorm_torch_f32
# Triton captures: TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas; NAME in layernorm_f32,
# rmsnorm_f32 (kernel layernorm_kernel / rmsnorm_kernel), attn_core_ref, attn_core_flash
# (attn_ref_kernel / attn_flash_kernel), attn_f32_ref, attn_f32_flash (attn_f32_ref_kernel /
# attn_f32_flash_kernel):
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py --triton tools/capture_examples/triton_NAME.py:KERNEL --binary --out fixtures/c_...
```
