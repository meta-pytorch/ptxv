# Equivalence on tensor cores: bf16 GEMM and bf16 flash attention (`ptxv equiv`)

**Update (2026-11-04, agent L).**
- 16-bit outputs (bf16 and f16) have an exact per-input bound and a one-ulp statement (§2).
  Eager's bf16 roundings between kernels are bounded as errors.
- FP8 / FP4 tensor-core steps have a contract derived from the calibrated models: 17.5u to
  130u per e4m3 / e5m2 step, about 2u per FP4 step, against eps >= 2 before (§5).
- An MXF8 GEMM gets a bound against the exact GEMM for every input: 255.5u x sum |terms|.
- (agent N) MXF8 against cuBLASLt and Triton, MXF4 against Triton: L0 for every same-order
  pair (against cuBLASLt L0~0, L0 up to the sign of zero, since MISS 109); reversed K is L1 with
  306u (MXF8) / 10.04u (MXF4) x sum |terms| (§5).
- (agent S) NVFP4: CuTeDSL against cuBLASLt's 2-CTA kernel is L0~0, L0 up to the sign of zero; against
  a dequantize-then-bf16 GEMM it is L1 with 169.38u x sum |terms|, because the dequantization
  is exact (§5).

**Status (2026-11-03).** The two bf16 tensor-core pairs that docs/equiv-reassoc.md §5 left open
are proven, each with a bound per output and every input of the captured shape:

| reference | candidate | level | bound per output, every input | sim trials: largest fraction of the bound | GB300 (2000 inputs) | each kernel vs exact |
|---|---|---|---|---|---|---|
| torch.mm (cuBLAS nvjet, 64x64x64 bf16) | Triton matmul, same K order | **L2** (bitwise but for the sign of zero) | 2^-7 \|y\| + 79.31u · Σ_k \|a_ik b_kj\| | 0 (bitwise, 20 trials) | 0 (bitwise) | 0.993 (sim), 0.995 (GB300) |
| torch.mm (cuBLAS nvjet) | Triton matmul, K blocks reversed | **L1** | 2^-7 \|y\| + 59.73u · Σ_k \|a_ik b_kj\| | 0.507 (20 trials, 81920 outputs) | **0.9605** | 0.993 (sim), 0.995 (GB300) |
| torch SDPA flash (FA2, sm80 HMMA; q, k, v 128x64 bf16) | Triton flash attention (tcgen05) | **L1** | (262913.25 + 111.88 Q)u · V1 + 9.93u · Vmax + 9.2e-41 | 0.343 (8 trials, 65536 outputs) | 0.359 (16.4M outputs, 0 over) | 0.40 / 0.39 (sim), 0.43 / 0.43 (GB300) |

u = 2^-24. y = Σ_k a_ik b_kj is the exact product. For attention, V1 = Σ_j p_j |v_j| ≤
Vmax = max_j |v_j|, with p the exact softmax of the exact scores, and Q = max_j Σ_d |q_d k_jd|
(the unscaled scores; the scale 1/8 and log2 e are in the exp's FFMA). The compact form holds
for Q ≤ 584. Beyond that, the exact per-input bound
applies (§3).

Fixtures:
- `fixtures/c_equiv_mm_bf16/equiv.toml` (cases `c_mm_bf16_nnt`, `c_triton_mm_bf16`,
  `c_triton_mm_bf16_rev`);
- `fixtures/c_equiv_sdpa/equiv.toml` (cases `r4_sdpa_flash_torch`, `r4_attn_triton`).

`cargo test` runs both (`crates/cli/tests/equiv.rs`: `bf16_gemm_is_bounded`,
`bf16_attention_is_bounded`).

Both bounds are dominated by bf16 roundings, not by the tensor cores:
- **GEMM.** Each output rounds to bf16 once in each kernel, which costs 2^-8 |y| per kernel
  (2^-7 for the two together). The bound is tight: 0.96 of it is reached on GB300, because the
  two bf16 results really do differ by one bf16 ulp where the f32 sums straddle a rounding
  boundary.
- **Attention.** Each kernel rounds P to bf16 before P·V (FA2: F2FP.BF16.PACK in registers;
  Triton: `p.to(bf16)` into TMEM) and rounds the output. Each of these roundings costs 2^-8
  (65536u) relative. They account for 262144u of the 262913u coefficient. The other 769u is
  the exp2, rcp, MMA and sum roundings of the two kernels together.

## 1. Numerics nodes for tensor-core steps (N-MMA, N-HMMA)

**MMA nodes.** `num.rs` `NodeKind::Mma { acc, terms, ops, eps }` is one model step:
- for tcgen05: one K=16 slice of a `tcgen05.mma` (`complete_mma`);
- for sm80: one `HMMA.16816.F32.BF16` (`num_hmma`).

The node has:
- the accumulator node (none, or constant +0);
- the 16 products, each operand a tracked value (`HalfOp`). An operand is either an input
  element or a computed node; a zero operand has no product;
- the step's error contract, `eps = MmaModel::eps_terms(n)`, where n is the number of terms
  that can be nonzero.

**The contract** (fp.rs `eps_terms`, derived from the calibrated model
`g16-f25-rz-accin-tzero-einputs`, rung 1a / N-HMMA). Write n for the number of terms (the
products and the accumulator). Then:
- every term's alignment exponent e satisfies 2^e ≤ |term|: e_a + e_b for a product, the
  accumulator's own exponent for the accumulator. Hence 2^emax ≤ max |term| ≤ Σ |terms|;
- every term is truncated toward zero to a multiple of 2^(emax − 25). The term that sets emax
  loses nothing: a bf16 × bf16 significand product has 16 bits and the f32 accumulator 24,
  both ≤ 26. Each of the other n − 1 terms loses less than 2^(emax − 25);
- the exact sum of the truncated terms is rounded toward zero to f32, which adds a relative
  2^-23.

So |computed − Σ terms| ≤ ((n − 1) 2^-25 + 2^-23 (1 + (n − 1) 2^-25)) Σ |terms|. For a full
step (n = 17) this is ≈ 10u. The previous contract charged 2^(1−f) per term with a factor 2
of slack, ≈ 36u. Consequences:
- Cor 1a multipliers and rung 2e's L2/L3 comparisons of MMA kernels are ~3.6x tighter;
- r2e_gemm's L3 tolerance τ = 576u is still proven, but the bounds now give 159u (79.5u per kernel,
  was 288u; docs/rung2e.md).

**Canonical forms** (`equiv.rs`). An MMA node canonicalizes as (accumulator, multiset of
product hashes) for L0. For L1/L2 its polynomial is Σ products + accumulator, and each child's
first-order multiplier grows by eps. The constant-zero accumulator is the same as none.

**16-bit provenance** (`halfin.rs`, N-HALF-LEAF extended). Products need operand origins at
the granularity of bf16 halves. Each byte of a register, of shared, global, local and TMEM
memory carries a `Src`:
- `E{alloc, elem, b}`: byte b of an input element;
- `N{node, b}`: byte b of the bf16 rounding of numerics node `node`;
- `Zero`.

The bytes move through:
- loads and stores of every width;
- cp.async, TMA load and store, LDSM / LDSM.T, STSM, SHFL, PRMT, MOV / IMAD.MOV, byte shifts;
- tcgen05.cp / LDTM / STTM.

An MMA operand register whose two halves are each a whole element (E or N) becomes a product
of tracked factors. `half_op` creates one memoized `Input` node per bf16 input element.

## 2. 16-bit roundings and 16-bit outputs (N-ROUND)

`NodeKind::Round { arg, f16 }` is a round-to-nearest-even of an f32 node to bf16 (f16 with
`f16`). Its error is relative u16 = 2^-8 (f16: 2^-11), plus a16 = 2^-134 (f16: 2^-25) absolute
where the result is subnormal (`equiv::round_unit`; f16 overflow is outside the bound). It comes
from:
- `F2FP.BF16.F32.PACK_AB` / `F2FP.F16.F32.PACK_AB` and `F2F.F16.F32` of a computed value
  (`num_f2fp16`); PTX `cvt.rn.bf16.f32` / `cvt.rn.f16.f32` lower to the packs. Each half's
  bytes become `N{round node}`, so P packed in registers (FA2) or stored to TMEM (Triton)
  arrives at the P·V MMA as a tracked operand;
- the 16-bit output stores. An output half whose two bytes come from one rounding node of the
  output's format is that node (`half_out_node`); one whose bytes are an input element's is
  that element's leaf (a copy kernel). `ptxv equiv` reads bf16 and f16 outputs element by
  element.

f16 rounding nodes and f16 input leaves are marked (`HalfState::f16_nodes`): their `N` bytes
are an f16 encoding, so they are never read as a bf16 MMA operand or widened as a bf16 value;
`HADD2.F32` widens them.

In the canonical forms a Round node grows its argument's multipliers by u16 and adds a16 to
the absolute part. Every rounding node also carries the absolute part of its operands through,
times (1 + eps): an add (1 + u), an FMA (1 + u, times |k| when the other factor is an exact
constant k: eager's `x * 1 + y`), an MMA (1 + eps, plus 2^-149 for an f32 result below the
normal range, which rounds toward zero). A rounded value times a non-constant factor has no
polynomial bound (its absolute error is scaled), as before.

**The 16-bit output form** (`equiv.rs` `compare`, `round16_gap`). When both outputs are
roundings to the same 16-bit format of L1-equal values a_R, a_C (exact value y), the
comparison peels the final rounding. Cor 1a bounds each kernel before it:
|a_K - y| <= G_K(x) = sum_m |x_m| g_K,m + A_K. Rounding to nearest is monotone, so
rnd(a_K) lies in [rnd(y - G_K), rnd(y + G_K)], and

  |C - R| <= max(rnd(y + G_C) - rnd(y - G_R), rnd(y + G_R) - rnd(y - G_C)).

This is the exact per-input bound (`OutBound::bound_at`): the tightest that the two kernels'
bounds imply. It is a whole number of 16-bit steps, and it is 0 where both intervals round to
one value. y and G_K are evaluated in f64 and widened by 2^-49 relative (`slack16`). The compact
form, for every input, is

  |C - R| <= 2 u16 |y| + (1 + u16) sum_m |x_m| (g_R,m + g_C,m) + (1 + u16)(A_R + A_C) + 2 a16.

**One ulp.** The spacing of the format at v exceeds u16 |v| (bf16 2^-8 |v|, f16 2^-11 |v|), so
two consecutive rounding boundaries (midpoints) around values of magnitude >= |y| - D are more
than h (|y| - D) apart, h = u16 / 2. If D = G_R + G_C satisfies D (1 + h) < h |y|, the
interval [y - G_R, y + G_C] holds at most one boundary, and C and R are equal or adjacent
16-bit values. With D <= k u S + A (S = sum_m a_m |x_m|), this holds for every input with
S <= kappa1 |y| and |y| >= y1, kappa1 = (h / 2) / ((1 + h) k u), y1 = 2 (1 + h) A / h (each half
of the budget). The report states it when kappa1 >= 1. `round16_gap_covers_and_adjacency_holds`
checks both claims on 400,000 random intervals; `rnd16_matches_the_conversions` checks the
f64 rounding against the hardware conversions (N-FALU) on 2^20 values.

**Different functions.** Eager torch rounds intermediates to bf16 between kernels (`x + a`,
`type_as(x)`, `silu(x)`), where Inductor keeps f32 inside one fused kernel. The two outputs are
then not the same function. The comparison still pivots on the exact polynomial y (the bf16
rounding inside eager's value is one more rounding node, charged 2^-8 relative), so the bound
above applies unchanged. The report says which kernel rounds before the output
(`round_inner`). Such a bound cannot promise one ulp: an intermediate value |x + a| much larger
than |y| contributes 2^-8 |x + a|. The norm form (layer norm, RMS norm) peels the output
rounding the same way (`NormBound::rounded`, `NormEval::own`).

**16-bit product outputs** (elementwise mode, `elementwise::product_sweep`). An output
bf16(rn(f(x) * y)) of two 16-bit input elements (fused silu * mul, against eager's silu kernel
then its mul kernel) is evaluated in both kernels at all 2^32 (x, y) pairs: f from the
exhaustive sweep of each kernel's slice, then the f32 product and the 16-bit rounding as the
hardware computes them. Every bound reported is a maximum over all inputs, not an estimate.

## 3. bf16 flash attention

**The kernels.**

Reference: `pytorch_flash::flash_fwd_kernel`, sm80 code.
- cp.async moves Q, K and V into shared memory; LDSM / LDSM.T load the fragments;
- S = Q Kᵀ by HMMA, with a row max over the 128 keys;
- P = exp2(S · c log2 e − m · c log2 e) by FFMA + MUFU.EX2, then rounded to bf16 by F2FP;
- O = P V by HMMA. The row sum is accumulated in f32, and the final 1/l is an rcp.rn
  (`MUFU.RCP` + Newton, with a CALL to the slow path).

Candidate: the rung 4 Triton kernel.
- tcgen05 S = Q Kᵀ into TMEM, LDTM;
- online softmax over two key blocks of 64 (BLOCK_N = 64) with exp2;
- P converted to bf16 and stored to TMEM (tcgen05.st); tcgen05 O += P V, rescaled by α between blocks;
- o = acc / l by div.full, per element.

**The normal form** (`softmax.rs`). The weighted normal form of docs/equiv-reassoc.md §5,
extended as follows:
- **MMA and Round nodes in weighted sums.** An MMA step adds its accumulator and its products,
  with every term record widened by (1 ± eps), as Cor 1a does. A Round node widens every
  record by (1 ± 2^-8) and adds 2^-134 to the absolute part.
- **Scores from MMA nodes.** A score atom is keyed by its exact polynomial Σ_d q_d k_jd.
  Both kernels' scores are rounded MMA polynomials, so both get the same atoms. Each kernel's
  Δ_K comes from its MMA nodes' multipliers.
- **Base-2 exponentials.** The exp2 slices get their contract from an exhaustive sweep of
  MUFU.EX2 over its domain (`exp2_contract`). The atoms are in log2 units: o = Σ_j 2^(x_j)
  v_j / Σ_j 2^(x_j), with x_j = c log2(e) s_j. A score error Δ in log2 units is ln 2 · Δ in
  the natural exponent, which scales kq.
- **The shifted form.** Both kernels compute 2^(fl(c' s_j) − fl(c' m)) as one FFMA
  (s_j c' − m c'), so the term's argument has the unrounded product c' s_j while the max is
  over the rounded fl(c' s_k). The normal form keeps the exact argument c' s_j − max_k fl(c' s_k).
  Rounding to nearest is monotone, so fl(c' max) = max fl(c' s_k). Its difference from
  c'(s_j − max s_k) is θ ≤ 2^-24 |c' max| ≤ 2^-8 while |c' s| ≤ 2^16 (SHIFT_X). Each term
  record widens by e^(32θ s) and each absolute part by 2^θ (`shift_factor`). The per-input
  bound checks |score| + Δ ≤ SHIFT_X (`AttnEval::covered`). The compact form limits Q to
  SHIFT_X.
- **rcp.rn.** FA2's 1/l is ptxas's rcp.rn sequence: an exponent filter, MUFU.RCP with one
  Newton step (FFMA, FADD.FTZ, FFMA), and a CALL of the slow path for the filtered inputs. The slicer treats it
  as one DivRn(1, l) (`seg.rs` `rcprn_at`), like the div.rn idiom (N-SEG). The fast path is
  checked to be correctly rounded on every one of the 2^32 inputs that pass its filter
  (`rcp_fast_ok`).
- **The flash form with a slice of the inverse.** FA2's output is fma(Seg(1/l), acc).
  Triton's is div.full(acc, l), the flash form docs/equiv-reassoc.md §5 already handles. `rcp_slice` reduces FA2's to (the sum S^, the rcp contract) by
  folding the slice's comparison with the row sum (`fold_sum_compare`). It then simplifies,
  prunes and substitutes the sum.
- **Output rounding.** Both outputs are bf16 roundings of the form, so each kernel's bound
  becomes (1 + 2^-8)(bound) + 2^-8 V1 + 2^-134 (`compare`, attention path).

**Result** (`ptxv equiv fixtures/c_equiv_sdpa/equiv.toml`). All 8192 outputs are L1 with

  |C − R| ≤ (262913.25 + 111.88 Q)u · V1 + 9.93u · Vmax + 9.2e-41,   Q ≤ 584.

Per kernel (Q = 0): reference 131458.20u, candidate 131464.98u. The exact per-input bound
evaluates the scores and Δ_K at the input (`AttnEval::at`). Trials use the per-input bound:

| | sim (8 trials, 65536 output-trials) | GB300 (2000 inputs, tools/equiv_gpu_check.py --dump-mma) | each kernel vs exact (sim) |
|---|---|---|---|
| fraction of the per-input bound used, max | 0.343 (2569.61u observed) | 0.359 (2946.43u observed), 0 of 16,384,000 output-trials over | 0.40 / 0.39 (GB300: 0.43 / 0.43) |

**What remains loose.** The bound charges each P rounding at its worst (2^-8 per term in the
same direction). Typical differences are much smaller, because most P values round the same
way in both kernels: the two kernels compute P differently (a different max and exp
schedule), so they round differently only near ties.

## 4. Precondition fixes found on the way

- **Taint** (`num.rs`):
  - local memory (STL / LDL) and cp.async destinations now carry their source's taint;
  - HMMA operands and STL values are value registers, not control;
  - half-warp LDTM (`.16x32bx2`) uses the lanes that execute (`eff_active`).
- **E-DESC-VERSION.** Triton 3.5's tcgen05 shared-memory descriptors leave bits 46–48
  (version) at 0. `check` reports this as a `contract` finding (2026-11-06; a violation
  before), and `ptxv equiv` lists it as relied on: GB300's MMA ignores those bits
  (probes/desc_version), and the captured outputs match the simulator bit for bit.
- **Trials** run with one schedule (`run_once`) unless the kernel has fp32 atomic adds: the
  kernels are race-free (precondition), so every schedule gives the same outputs. Random
  schedules of `c_mm_bf16_nnt` once deadlocked here: a cluster's `barrier.cluster` completing
  woke the other cluster's waiters of the same generation (fixed in `exec.rs`
  `cluster_try_complete`; test `cluster_barrier_wakes_only_its_cluster`).
- **The Triton matmul** uses num_stages=1. With the default pipelining, Triton 3.5 omits the
  proxy fence between its cp.async writes and the MMA. ptxv flags this as a race.

## 5. Narrow-format tensor cores: the error contract (N-MMA-F8, N-MMA-BS)

Before: every `kind::f8f6f4` / block-scaled step carried `fp.rs` `NARROW_MMA_CONTRACT`
(eps >= 2 times sum |terms|): sound, but 8 chained steps of r3_cute_mxf8 gave a bound ~10^22
times the error. N-MMA's derivation does not apply, because these models align on exponent
fields, not magnitudes.

**f8f6f4 and mxf8f6f4** (`mma_lp::lp_trunc`, `lp_eps`; model `g32-f25-rz-accin-tzero-efields`).
A step adds the terms t_j (32 products and the accumulator). Let e_j be the alignment
exponent: ef_a + ef_b from the fields (a subnormal code at 1 - bias), and the accumulator's own
exponent. Every term is truncated toward zero to a multiple of Q = 2^(emax - 25), then the sum
is rounded toward zero to f32.
1. A term that is a multiple of Q loses nothing; any other loses less than Q (and at most
   itself).
2. The term k that sets emax is a multiple of Q. A product is M_a M_b 2^(e_k - mb_a - mb_b)
   with integers M (normal or subnormal codes alike), and mb_a + mb_b <= 6 <= 25. The
   accumulator is a normal f32 (subnormal accumulators are outside the model), a multiple of
   2^(e - 23).
3. Normal setter (both operands normal, or the accumulator): 2^emax <= |t_k| <= sum |t|. The
   other n - 1 terms lose less than (n - 1) 2^-25 sum |t|. This is N-MMA's argument.
4. Setter with a subnormal operand: a subnormal code is at least 2^(ef - mb), so
   2^emax <= 2^s_k |t_k|, s_k = the subnormal operands' mantissa bits (e4m3 3 each, e5m2 2).
   Each lossy term then loses less than 2^(s_k - 25) sum |t|. A product j is lossy only if
   e_j <= emax - 26 + mb_a + mb_b. `lp_trunc` enumerates every code pair of the two formats.
   Where no product can lie that far below a subnormal setter (e4m3 x e4m3: the setter's
   e_k <= 2, products >= 2^-12), only the accumulator can lose. Otherwise the other 31
   products can too.
5. tau = the largest of these; eps = tau + 2^-23 (1 + tau). An f32 result below the normal
   range adds less than 2^-149 absolute (`F32_RZ_SUB`, in the canonical form's absolute part).
   An mxf8f6f4 step (scale_vec 1X: one ue8m0 pair per K = 32 step) scales every product by
   one power of two, which changes neither ratio.

| A x B | eps, accumulator 0 | eps, accumulator nonzero | what sets it |
|---|---|---|---|
| e4m3 x e4m3 | 17.5u | 34u | subnormal x subnormal setter, lossy accumulator: 2^(6-25) |
| e5m2 x e5m2 | 64u | 66u | one-subnormal setter, 31 lossy products: 31 x 2^(2-25) |
| e4m3 x e5m2 | 126u | 130u | e4m3-subnormal setter, 31 lossy products: 31 x 2^(3-25) |
| e2m3 x e2m3 | 17.5u | 34u | as e4m3 x e4m3 |
| e3m2 x e3m2, e2m1 x e2m1, e4m3 x e2m1 | 17.5u | 18u | normal setters only: (n - 1) 2^-25 |

u = 2^-24. The bf16 contract (N-MMA) is 10u for 17 terms.

**mxf4 / mxf4nvf4** (`mma_lp::fp4_eps`; model `u16-f35-live-sfields`). The terms are the 4
unit sums P_u s_u and the accumulator, truncated at Q = 2^(emax - 35), emax over the live units'
scale-exponent sums and the accumulator.
- A unit sum is a multiple of 1/4 (e2m1 products), so P_u s_u is a multiple of 2^(se_u - 2) for
  ue8m0 (s_u = 2^se_u), of 2^(se_u - 8) for ue4m3 (s_u = M_a M_b 2^(se_u - 6)). The setter loses
  nothing.
- A live unit has a product |a b| >= 1/4, so 2^se_u <= C sum (its |a b s_u|), with C = 4 for
  ue8m0 and 2^8 for ue4m3 (both scales nonzero: M >= 1).
- Up to 3 units and the accumulator lose less than Q each: tau = 4 x 2^(c - 35) (c = 2 or 8),
  eps = 2.008u (ue8m0) / 2.5u (ue4m3) with an accumulator.
- **Requirement**: no ue4m3 scale of a live unit is 0. A zero scale still sets emax, while its
  unit adds 0 to sum |terms|, so no relative bound exists. Steps with one get the loose
  contract (exec.rs `complete_mma_lp`), and for inputs with one the contract does not hold
  (ASSUMPTIONS N-MMA-BS).

**Tested against the models** (`mma_lp::contract_tests`, `cargo test`):
- `lp_contract_covers_the_model`: 10 format pairs x 6,000 adversarial steps (random codes,
  subnormal-heavy operands, a subnormal setter next to a far-below accumulator, wide spreads,
  cancelling pairs, zeros, ue8m0 scales; errors in exact arithmetic). For each step: model error
  <= the data-dependent bound `lp_step_bound` (sum over non-multiples of min(Q, |t|), plus the
  rounding) <= the contract. Largest error / contract: 0.04 to 0.19 per pair.
- `lp_contract_is_reached`: e4m3 x e5m2 with a subnormal setter (2^-9 x 2^15: aligned at 2^9,
  value 2^6) and 31 products just below Q: the error is over 0.9 of the contract.
- `subnormal_setter_breaks_the_bf16_derivation`: e4m3 2^-9 x 1 with an accumulator 1.75 x 2^-31.
  The error (3u relative) exceeds N-MMA's eps_terms(2) (2.5u). The derivation above is needed.
- `fp4_contract_covers_the_model`: 20,000 steps per scale format, largest 0.997 (ue8m0) and
  0.831 (ue4m3) of the contract; and a zero ue4m3 scale that truncates an accumulator to 0.

**An MXF8 GEMM, every input** (`fixtures/c_equiv_mxf8/equiv.toml`, test `mxf8_gemm_is_bounded`).
FP8 operand bytes and ue8m0 scale bytes are now leaves: `Half::Byte` inputs (u8 / u16 buffers);
`byte_leaf` decodes them in the format the step reads, with the format in the leaf index. Their
provenance follows TMA into shared memory and `tcgen05.cp` into TMEM. A block-scaled MMA node
carries each product's two scale factors (`NodeKind::Mma::sc`), so its polynomial term is
a b sa sb. The CuTeDSL MXF8 GEMM (r3_cute_mxf8, K = 256, 8 steps) against itself is L0, with
each kernel's own bound against the exact MXF8 GEMM:

  |C - exact| <= 255.5u x sum_k |a_mk b_nk sa sb|   for every input (was >= 2^8 x sum, useless).

That is 17.5u for the first step plus 7 x 34u. 20 random-input trials (random e4m3 codes,
subnormals included, random ue8m0 scales 2^-8 .. 2^7) use at most 0.078 of it. At the captured
input, `ptxv num` now gives bound / error 2068 on average (was 1.3 x 10^22). The B30 mutant
(K block 0 uses K block 1's scales) is not L1: 128 monomials a b sa sb differ. Its witness is
not replayed, because setting a byte leaf to "1" does not give the value 1.

**MXF8 across implementations** (agent N, 2026-11-04). Three other implementations of the
same GEMM, on r3_cute_mxf8's input bytes and buffer names (A, B, SFA, SFB, C):

| candidate | kernel | level | bound per output, every input | sim trials (20): used | GB300 (1000 inputs, 16.4M outputs): used |
|---|---|---|---|---|---|
| `c_smm_mxf8`: torch._scaled_mm (cuBLASLt) | nvjet_sm103_qqsss_128x128_128x6 | **L0~0** (L0 up to the sign of zero) | \|C - R\| = 0 | 0 | 0 (bitwise) |
| `c_triton_mxf8`: Triton tl.dot_scaled, K blocks of 128 | tcgen05 mxf8f6f4 block_scale 1X | **L0** | \|C - R\| = 0 | 0 | 0 (bitwise) |
| `c_triton_mxf8_rev`: the same, K blocks of 32 reversed | same | **L1** | 306u x sum_k \|a b sa sb\| | 0.067 | 0.052, 0 over (84% of outputs differ) |

Fixtures: `fixtures/c_equiv_mxf8_cublas/equiv.toml`, `fixtures/c_equiv_mxf8_triton/equiv.toml`
(tests `mxf8_cublas_is_bitwise_up_to_zero_signs`, `mxf8_triton_is_bounded`). GB300 trials:
`tools/mx_gpu_check.py` (random e4m3 codes, subnormals and -0 included; ue8m0 scales 2^-8 ..
2^7), checked by `PTXV_EQUIV_HW_TRIALS`.

- **cuBLASLt** runs the same 8 steps (two K tiles of 128, 4 UTCQMMA each) in the same order. Its
  epilogue is `FFMA2 acc * alpha + 0` with alpha = 1, so its output is acc + (+0). The level is
  decided with that final +0 peeled (`Canon::l0_z`): the outputs are equal bits except that a -0
  accumulator becomes +0, level L0~0, not L0 (N-EQUIV; until MISS 109 it was reported as L0). Lifting it needed two SASS fixes: the 32-bit immediate of
  `UIADD3.64` is sign-extended (it stepped a tensor-map pointer back by -0x800 and left the
  parameter block), and the bulk read wait is a control-word wait on `UTMACMDFLUSH`'s read
  scoreboard (on a `BAR.SYNC`, no `DEPBAR`), modeled as `wait_group.read 0`
  (`Kernel::bulk_read_waits`). Without the second, ptxv reported a proxy race on the reused
  staging buffer. With both, `check` gives only E-BULK-EXIT, as for the bf16 nvjet.
- **Triton** moves the codes by LDG / STS and the scales through LDSM / STSM and `tcgen05.st`
  into TMEM (byte provenance already followed them). In K order it is L0. K-reversed, the 8 steps
  run from K = 224 down to 0. That is the same exact function with another rounding order: L1,
  not L2 (on some monomials its multiplier is 170u against the reference's 136u). The bound,
  306u x sum |terms|, is Thm 4's max over monomials of (g_R + g_C) / a_m. It is below the sum of
  the two own bounds (511u), because a term early in one kernel's chain is late in the other's.
- Each kernel against the exact GEMM uses at most 0.078 (sim) / 0.060 (GB300) of its own 255.5u
  bound.
- Byte buffers pair by size (u8 [1024] in Triton, u16 [2, 256] in CuTeDSL's TMA view of SFA):
  their leaves are byte offsets.

**MXF4 across implementations.** `fixtures/r3_cute_mxf4` is the same CuTeDSL example for MXF4
(e2m1 codes two per byte, low nibble first; ue8m0 scales per 32; `kind::mxf4nvf4.block32`;
A and B through `16U4_ALIGN8B` tensor maps). Its simulator output equals GB300 on both
front-ends. The candidates are Triton tl.dot_scaled "e2m1" (`kind::mxf4 scale_vec::2X`), in
K order and with K blocks of 64 reversed (`fixtures/c_equiv_mxf4_triton/equiv.toml`, test
`mxf4_triton_is_bounded`):

| candidate | level | bound per output, every input | sim trials (20): used | GB300 (1000 inputs): used |
|---|---|---|---|---|
| `c_triton_mxf4` | **L0** | \|C - R\| = 0 | 0 | 0 (bitwise) |
| `c_triton_mxf4_rev` | **L1** | 10.04u x sum_k \|a b sa sb\| | 0.546 | 0.471, 0 over (16% of outputs differ) |

Each kernel's own bound is 8.03u x sum |terms| (2.0u for the first step, 2.008u for each of the
other 3: `fp4_eps`). Each kernel uses at most 0.71 (sim) / 0.67 (GB300) of it, so the FP4 contract
is close to tight. What was needed:
- FP4 leaves: element j of a K = 64 slice is nibble j % 2 of byte 16 (j / 32) + (j % 32) / 2. Its
  leaf is input nibble 2 byte + nib (`code_op`, `equiv::leaf_code`; trials decode the same way).
- The L0 form of an FP4 step is a multiset of 16-element units (`Mma::units`). The model sums
  each unit exactly before truncating, so it is not invariant under moving terms across units.
- The step's contract counts the units that can hold a nonzero product, not ceil(n / 16).
- 4-bit tensor maps (case `elem4`): byte maps whose coordinate 0 counts elements (S-TMA).

**NVFP4 across implementations** (agent S, 2026-11-04). `fixtures/r3_cute_nvf4` is the CuTeDSL
example for NVFP4 (e2m1 codes, ue4m3 scales per 16 K elements, none 0; `kind::mxf4nvf4.block16`,
one CTA, 4 steps of K = 64). Its simulator output equals GB300 on both front-ends. The
candidates read the same bytes and names (A, B, SFA, SFB; C f32; no global scale):

| candidate | kernel | level | bound per output, every input | sim trials (20): used | GB300 (1000 inputs, 16.4M outputs): used |
|---|---|---|---|---|---|
| `c_smm_nvf4`: torch._scaled_mm (cuBLASLt) | nvjet_sm103_oosss_128x128_256x7_1x2_2cta (a CTA pair) | **L0~0** (L0 up to the sign of zero) | \|C - R\| = 0 | 0 | 0 (bitwise) |
| `c_triton_nvf4_dq`: dequantize, then a bf16 GEMM (Triton) | cvt e2m1 / e4m3 -> f16, f32 product, bf16 rounding, tcgen05 kind::f16 (A in TMEM) | **L1** | 169.38u x sum_k \|a b sa sb\| | 0.130 | 0.153, 0 over (56% of outputs differ) |

Fixtures: `fixtures/c_equiv_nvf4_cublas/equiv.toml`, `fixtures/c_equiv_nvf4_dq/equiv.toml`
(tests `nvf4_cublas_is_bitwise_up_to_zero_signs`, `nvf4_dequant_bf16_is_bounded`). GB300 trials:
`tools/mx_gpu_check.py --nvf4` (random e2m1 bytes; ue4m3 scale codes 1..126, subnormals
included, none 0 and none NaN).

- **Each kernel against the exact GEMM:** 9.88u x sum |terms| for the block-scaled kernels (4
  FP4 steps, `fp4_eps` with ue4m3 scales). They use at most 0.524 of it on the simulator's
  trials and 0.650 on GB300's.
- **Input precondition: every ue4m3 scale byte of a unit with a nonzero product is nonzero**
  (and none is 0x7f, NaN). A zero scale still sets the step's alignment while adding nothing to
  sum |terms|, so the FP4 contract (N-MMA-BS) gives no relative bound there. Steps with a
  possibly zero scale get the loose contract, and the per-kernel bound above is claimed only for
  inputs that meet the precondition. The pair's L0 is a statement about identical rounding DAGs
  and holds for every input; GB300 agreed bitwise on all 1000 trials.
- **cuBLASLt** runs a 2-CTA pair: `UTCOMMA.2CTA.BLOCK16` with M = 256 per pair tile (the model
  takes only M = 128 per CTA of a pair, and matches GB300; the problem has M = 128), B split 64 / 64, each CTA's scale factors copied into its own TMEM by
  `UTCCP.T.S.2CTA.4x32dp128bit` (SFB complete in both), the accumulator read back with
  `LDTM.16dp128bit.x4`, and the warp roles dispatched with `LOP3.LUT.PAND` (docs/lowp.md,
  E-PAIR, N-MMA-BS, S-TMEM, L-LOP3-P). The 4 steps are the CuTeDSL kernel's, in the same order;
  the epilogue is `FFMA2 alpha * acc + 0` with alpha = 1, so the outputs are equal but for the
  sign of a zero accumulator. `check` gives only E-BULK-EXIT, as for the other nvjet kernels.
- **Global scale.** torch.nn.functional.scaled_mm's two-level NVFP4 (BlockWise1x16 + TensorWise)
  does not change the MMA steps: torch computes alpha = fl(GA GB) in an elementwise kernel and
  cuBLASLt's epilogue computes fl(alpha acc), acc being the output above. On GB300 this matches
  all 3,276,800 outputs of 200 random inputs, while fl(fl(acc GA) GB) matches 64%
  (`tools/nvf4_global_scale_check.py`). The `_bx`
  kernel reads alpha through a device pointer and branches on alpha == 0 (data-dependent control),
  so it is not a case here.
- **Dequantize-then-bf16.** The Triton kernel (`tools/capture_examples/triton_nvf4_dequant.py`)
  converts each e2m1 code with `cvt.rn.f16x2.e2m1x2` and each scale with `cvt.rn.f16x2.e4m3x2` of
  code & 0x7f (e4m3(c & 0x7f) = ue4m3(c) for every byte), multiplies in f32, rounds to bf16 and
  runs a bf16 tl.dot. Converted codes are byte leaves (N-HALF-LEAF, `half_unpack`), so its
  products are the monomials a b sa sb of the block-scaled kernels. Not a different function:
  e2m1 x ue4m3 values are N 2^L with |N| < 2^6, L >= -10, below 2^12, which f32 and bf16 hold
  exactly (N-EXACT). The f32 product and the bf16 rounding are therefore exact for every input,
  and only the MMA contracts remain: 16 bf16 steps of K = 16 give 159.50u, the FP4 kernel
  9.88u, 169.38u together (Thm 4). Without N-EXACT the two bf16 roundings alone would cost
  2 x 2^-8 = 131,072u, plus an unbounded subnormal term. The same argument does not cover MXF8
  (e4m3 x ue8m0 reaches 2^-136, below bf16's spacing): there the dequantization is a different
  function, not claimed.

Not covered: an f16 D (the numerics forget it), FP6 operands of `kind::f8f6f4`, operands that a
kernel converts to FP8 itself (`N` bytes: unknown), dequantize-then-bf16 references for MXF8 /
MXF4 (not exact; their bf16 roundings and subnormal terms would need the general product
bound).

## 6. TF32 tensor cores, the 3xTF32 split, and benign data-dependent control (N-TF32, N-FINITE, N-FMAX)

**Update (2026-11-04, agent O).** `HMMA.1688.F32.TF32` steps are numerics nodes, and CUTLASS's
big + small split of f32 operands is followed and bounded against the exact product. Three
taint-precondition relaxations, each with a written argument (ASSUMPTIONS.md) and negative
tests, cover finiteness tests, float-max atomics and flag words.

| reference | candidate | level | bound per output, every admissible input | sim trials (20): largest fraction |
|---|---|---|---|---|
| f32 SIMT GEMM (FFMA, k in order) | 3xTF32 (CUTLASS split) | **L1** | 239.52u · Σ_k \|a_ik b_kj\| | 0.196 |
| | 3xTF32, K steps reversed | **L1** | 183.52u · Σ_k \|a_ik b_kj\| | 0.193 |
| | 1xTF32 (raw f32 words) | **L1** | 32895.59u · Σ_k \|a_ik b_kj\| | 0.397 |
| | 2xTF32 (small*big dropped) | not proven (a correction without its partner is untracked) | | |
| sequential fmaxf row max | atomicMaxFloat into shared memory | **L0** | | |
| | the same, choice by v > 0.5 | not proven (data-dependent branch) | | |

D = A[32x64] B[64x32] (fixtures/c_equiv_tf32, kernels in `mm_tf32.cu`); row maxima of X[32x128]
(fixtures/c_equiv_fmax, `fmax.cu`). Every kernel equals GB300 bitwise (`ptxv hw`, 200 runs on
the fixture input, 20 runs on each of 8 more inputs, normal and uniform). Each 3xTF32 kernel
against the exact product: 175.52u · Σ |a b| (theta 32u, the rest the 24 MMA steps' contracts);
the f32 reference: 64u. Tests: `tf32_gemms_are_bounded`, `tf32_data_dependent_guards_block`,
`float_atomic_max_is_the_maximum` (crates/cli/tests/equiv.rs); `tf32_split_theta_holds`,
`tf32_triple_matches_only_same_nodes`, `cancelled_monomial_keeps_its_error` (equiv.rs);
`atomic_max_float_is_the_float_max`, `flag_bits_keep_their_own_taint` (num.rs).

**TF32 parts** (`NodeKind::Tf32`). For a tracked f32 value x:
- hi(x), the low 13 mantissa bits cleared: what the TF32 MMA reads of any register, and
  `LOP3 d, x, 0xffffe000, RZ, 0xc0` (the LUT is checked bit by bit). Relative error < 2^-10.
- lo(x) = x - hi(x): `FADD d, x, -hi`, exact.
- lo'(x): `IMAD.IADD.U32 d, lo, 0x1, k` or `LEA d, lo, k, 0x0` with an input-independent
  k = 0x1000, then the MMA's truncation: lo rounded half away from zero to 11 bits
  (CUTLASS's `round_half_ulp_truncate`), relative error <= 2^-11. lo read unrounded: 2^-10.

A step's polynomial is that of the generic MMA node (N-MMA's contract, n = 9 for TF32). On its
own, hi(a) hi(b) is a b with each factor rounded by 2^-10 (1xTF32: about 2^-9 per product).

**3xTF32** (`equiv.rs` `tf32_chain`). CUTLASS issues, per K step, lo'(a) hi(b), hi(a) lo'(b) and
hi(a) hi(b) as three MMAs into one accumulator. The canonical form of an MMA node walks its
accumulator chain (each step a term of the next, coefficient 1) and matches each hi(a) hi(b)
with one of each correction whose a and b have the same canonical hashes (one value loaded
twice is two nodes with one hash). Then

  hi_a hi_b + lo'_a hi_b + hi_a lo'_b = a b (1 + theta),
  theta = (lo'_a - lo_a) hi_b / (a b) + hi_a (lo'_b - lo_b) / (a b) - lo_a lo_b / (a b),
  |theta| <= 2^-10 d_a + 2^-10 d_b + 2^-20 = 2^-19 for d = 2^-11.

Below the last of the three steps the products are rounded separately, |t1| + |t2| + |t3| <=
(1 + w_a + w_b) |a b| with w = 2^-10 (1 + d); the term enters at the last step with multiplier
theta + (1 + w_a + w_b)(h - 1), h those steps' growth, and is a plain term from there on.
An unmatched correction is untracked, so dropping one (2xTF32) is not proven.

Requirement (`condition: N-TF32` in the report): every value read as a TF32 operand or split is
0 or at least 2^-103 in magnitude, so hi and lo are normal or zero (lo is a multiple of x's
f32 ulp) and the MMA's operands are in N-HMMA's calibrated domain.

**N-FINITE** (taint). CUTLASS rounds the small part only if it is finite: `FSETP.NE.AND P, PT,
|lo|, +INF, PT; @P IMAD.IADD.U32 lo, lo, 0x1, k`. The guard is input data, but for finite x it
is always true. `num_fin_setp` clears the taint of an `FSETP` whose one tainted operand is a node
`fin` proves finite (inputs, finite constants, sums, FMAs, MMA steps, roundings, EX2 / TANH /
SIN / COS, TF32 parts, maxima with a non-+inf constant, slices made of such operations) and
whose other operand is an input-independent infinity or NaN. By induction along the trace, an
admissible input (finite elements, no operation overflows) never leaves the captured trace.
The report says "relies on: N-FINITE". Negative tests (fixtures/c_equiv_tf32/neg.toml): rounding
only when `small > 0`, or when the float of x's bits shifted left is finite, stays a
data-dependent guard.

**N-FMAX** (taint). CUTLASS's `atomicMaxFloat`: `ISETP.GE.AND P, PT, v, RZ, PT; @!P BRA L;
ATOMS.MAX.S32 RZ, [a], v; BRA J; L: ATOMS.MIN RZ, [a], v; J:`. Either side writes the float
maximum (FMNMX's order) of non-NaN floats, to the same word, with the same footprint, and the
maximum is commutative. `fmax_diamond` recognizes the shape (predicate on the same v, the
sides the right way round); each atomic makes the word's node `Max(old, v)` after checking both
are non-NaN for admissible inputs. The variant that branches on `v > 0.5` still blocks.

**Bitwise LOP3 taint.** A LOP3 result bit is input data only where it can take both values as
the tainted operand bits vary (`lop3_taint`). CUTLASS's f32 attention keeps index masks and the
ex2 range predicates as bits of one word (R108); with whole-register taint every mask test
became input data.

**A cancelled monomial keeps its error.** The canonical forms dropped monomials whose
coefficient is 0. For (x + y) - x against y that left the bound 2u |y| for an error of up to
u |x| (x = 1e8, y = 1: bound 1.2e-7, error 1). Such monomials are now kept with their g_m (on
either side alone), L1 ignores their zero coefficient, and `rigorous` takes both sides'
monomials.

**f32 SDPA (fixtures/c_cve_sdpa_f32, `fmha_cutlassF_f32_aligned_64x64_rf_sm80`).** Proven L0 against
its compile-vs-eager partner since agent Q's update (§7). As of agent O: not proven.
Its 229 data-dependent guards and branches per kernel were:

| # | what | status |
|---|---|---|
| 160 | `@P IMAD.IADD.U32` / `@P LEA` rounding the TF32 small part, P = isfinite(small) | 158 no longer reported (N-FINITE decides 120 comparisons); 2 `LEA` of values not proven finite remain |
| 56 | ex2.approx.f32 (non-ftz) range idiom: `FSETP.GEU x, -126; @!P FMUL x, 0.5; MUFU.EX2; @!P FMUL y, y`, the predicate kept in R108 bits | 19 inside slices (N-SEG); 34 `FMUL` + 3 `LOP3` remain (their slices fail: `P2R` is not a slice operation, slices over 1024 operations) |
| 4 | atomicMaxFloat of each warp's row maximum | branches cleared by N-FMAX; the row maximum has no node or slice, so the 8 atomics are reported |
| 1 | `changed = m_prime < mi` (0x86a0) | m_prime is the -inf start, so N-FINITE would decide it once mi (the shared maximum) has a `fin` node |
| 8 | rcp.rn.f32 fast / slow path of 1 / s_prime | 2 recognized; 6 use ptxas's other filter (`VIADD R, x, 0x1800000; LOP3 0x7f800000; ISETP.GT.U32 0x1ffffff`), which `rcprn_at` does not match |

With this work 61 per kernel remain (segment mode), and the outputs are untracked: P reaches
the P·V MMAs through shared memory, and its stored slices fail as above. Beyond the
preconditions, the attention normal form (N-ATTN) would need TF32-aware weighted sums (MMA
chains with `Tf32` factors and matched corrections in `softmax.rs`) and CUTLASS's bookkeeping
through shared memory (mi, m_prime, s_prime, out_rescale, the per-warp row sums).

## 7. f32 SDPA: CUTLASS's online softmax through slices (N-SEG, N-FINITE, N-FMAX, N-EQUIV)

**Update (2026-11-04, agent Q).** `fmha_cutlassF_f32_aligned_64x64_rf_sm80` (fixtures/c_cve_sdpa_f32)
has no data-dependent guard left and every output has a numerics DAG. torch.compile and eager
launch the same kernel on the same shape, so the compile-vs-eager pair is **L0 for every
admissible input** (finite elements, no overflow; relies on N-SEG, N-FMAX, N-FINITE):
`ptxv compare` on the recorded pair, 8192/8192 outputs, 84 guards inside slices, 161 comparisons
decided by finiteness, 4 float-max diamonds. 32 s, 1.7 GB.

What blocked, and the rule that cleared it (each in ASSUMPTIONS.md with tests):

| blocker (§6) | cause | fix |
|---|---|---|
| row-max slices failed | a slice crossing the atomicMaxFloat diamond evaluated its branch predicate (`ISETP` was not a slice op) | slices cross a recognised diamond: its sides write no register (N-SEG) |
| row max "not a non-NaN float" | the causal mask `FSEL R, R, -INF, P` tainted the -inf; the HMMA chains' RZ accumulator was an unmarked leaf, so no MMA node was `fin` | SEL / FSEL by an input-independent predicate carries the chosen operand's taint (N-EQUIV); RZ accumulator a constant (N-FINITE) |
| shared row max had no node | the first lane's atomic saw "tainted" bytes: the read flag was the whole warp instruction's | per-thread read taint (N-FMAX) |
| `m_prime < mi` (0x86a0) | decided by N-FINITE once mi has a node | (no new rule) |
| ex2 range guards (34 `FMUL`, 3 `LOP3`), slices over 1024 ops | the predicates live in bits of R108, saved with `P2R`, read back with `LOP3 P, RZ, R108, bit` / `ISETP.NE`; a slice of one bit pulled in every other bit's history | slices in demanded bits: flag-word writes to other bits skipped, LOP3 sources narrowed, untainted bits constant; `P2R`, `ISETP`, `VIADD` slice ops (N-SEG) |
| guarded flag writes tainted every lane | a guard tainted in one lane tainted all members' destinations | per-thread guards for register operations (N-EQUIV) |
| rcp.rn x6 | `VIADD` filter; quotient written into b's register | both recognised; the slow-path callee's registers now checked dead (they were not) |
| P to P·V through shared memory, row sums through `SHFL` | a masked P (constant 0) became "tainted without a node": loads and shuffles took the whole warp's taint, `STS.64` the pair's | per-register loads, per-word stores, per-lane shuffles (N-EQUIV) |
| 2 TF32 split guards (`LEA`), "TF32 low part outside a matched product" | the masked P = exp2(-inf - mi) was not `fin` | abstract slice values: -inf - finite = -inf, exp2(-inf) = 0 (N-FINITE) |

Only one key block exists at seq 64, so the rescaling `exp2(m_prime - mi)` is exp2(-inf - mi) and
`s_prime` is 0 times it; the slices carry it as computed, and L0 needs no normal form.

**Negative tests** (fixtures/c_equiv_online, online.cu, GB300 1000/1000 runs = simulator per
case): the online-softmax kernel is L0 against itself; with m_prime starting at -1e30 the branch
stays data-dependent; a mask chosen by `x > 0`, a shuffle from lanes holding data, and a float2's
data word each keep their guard. The same kernel did not prove at `base` (row max, guards).

**What is still open for a bound against a different attention kernel** (N-ATTN with TF32): the
weighted normal form (`softmax.rs` `WSum`) must read 3xTF32 MMA chains (the matched triple as one
term a b (1 + theta), the chain's growth per step), and CUTLASS's shared-memory bookkeeping must
reduce to exp(s - max) / sum exp · V: here mi = Max(warp maxima) (a `Max` node), the row sum
s_prime + per-warp partial sums through `SHFL.BFLY` (Add nodes), out_rescale and s_prime as
slices of exp2(-inf - mi) (constant 0 for admissible inputs), and the output fma(Seg(1 / s), acc)
with `FSEL(s != 0, rcp, 1)` inside the slice. Neither is implemented; L0 does not need them.

## Reproduce

```
source env.sh; cargo build --release
ptxv equiv fixtures/c_equiv_mm_bf16/equiv.toml --trials 200
ptxv equiv fixtures/c_equiv_sdpa/equiv.toml --trials 8        # ~10 GB, a few minutes
ptxv equiv fixtures/c_equiv_tf32/equiv.toml --trials 200      # TF32 GEMMs (§6)
ptxv equiv fixtures/c_equiv_tf32/neg.toml; ptxv equiv fixtures/c_equiv_fmax/equiv.toml
ptxv equiv fixtures/c_cve_sdpa_f32/equiv.toml; ptxv equiv fixtures/c_equiv_online/equiv.toml   # §7
# GB300 trials (GPU 1): dump inputs and both kernels' outputs, then check them
CUDA_VISIBLE_DEVICES=1 $PTXV_PYTHON tools/equiv_gpu_check.py --trials 2000 --dump-mma /tmp/e-mma
PTXV_EQUIV_HW_TRIALS=/tmp/e-mma ptxv equiv fixtures/c_equiv_sdpa/equiv.toml
```
