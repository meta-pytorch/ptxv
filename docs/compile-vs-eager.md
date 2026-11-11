# torch.compile vs eager, kernel by kernel

**Status (2026-11-03, agent G).** ptxv's first end-to-end use case: every kernel that
`torch.compile(mode="default")` (Inductor) generates for a Llama-style transformer block is
paired with the eager kernel sequence computing the same tensor, both are captured on the same
input bytes, and `ptxv compare` reports the execution verdicts, the equivalence level and bound
(or why not), and the GPU difference.

**f32: every Inductor (Triton) kernel output is proven equivalent to its eager counterpart, 6 of
6**, each a fusion of 2 to 7 eager kernels (RMSNorm + weight; residual add + RMSNorm; RoPE,
one kernel writing q and k; silu * mul; residual add + add). The 6 extern pairs (cuBLAS f32 mm x5,
f32 SDPA) are the same library call on both sides and are blocked. **bf16: nothing is proven**
(the outputs are bf16, which the normal forms do not cover yet, plus two lift gaps and an nvjet
model gap). Totals over both dtypes: 25 pairs, 6 proven, 0 refuted, 19 blocked.

| | pairs | proven | blocked |
|---|---|---|---|
| f32, Inductor Triton kernels | 6 | **6** (L1 x2, L2 x3, elementwise product x1) | 0 |
| f32, extern (mm, SDPA) | 6 | 0 | 6 |
| bf16, Inductor Triton kernels | 7 | 0 | 7 |
| bf16, extern (mm, SDPA) | 6 | 0 | 6 |

Every kernel on both sides lifts except two (the f32 SDPA kernel; the bf16 RoPE Triton
kernel), and every lifted kernel is certified clean and equal to GB300 bitwise, except the bf16
nvjet GEMMs (below).

**Update (2026-11-04, agent K).** Every kernel of both dtypes now lifts, is certified clean
(E-WELLSYNC) and equals GB300 bitwise. Re-running `compare --recompare` on the same captures:
**25 pairs, 16 proven, 0 refuted, 9 blocked.** The 10 extern mm pairs are the same library call
on both sides and are proven bitwise identical (L0) for every input:

| | pairs | proven | blocked |
|---|---|---|---|
| f32, Inductor Triton kernels | 6 | 6 | 0 |
| f32, extern (mm, SDPA) | 6 | **5** (mm x5, L0) | 1 (SDPA) |
| bf16, Inductor Triton kernels | 7 | 0 | 7 |
| bf16, extern (mm, SDPA) | 6 | **5** (mm x5, L0) | 1 (cuDNN SDPA) |

- **bf16 mm x5** (nvjet, cluster of 8, multicast TMA): the "simulator model gap" below was
  two bugs. First, the S-TMA alignment rule was too strict: the simulator required a swizzled
  TMA destination to be aligned to 8 rows x span (1024 B for 128B), and a multicast slice at
  stage base + 0x200 broke it. GB300 requires 128-byte alignment and takes the swizzle phase
  from the shared-memory address bits (probes/tma_mcast_swizzle: 48/48 configs, multicast load
  and store; unaligned starts fault). The data placement (`swz` of the absolute address) was
  already right: the case run alone matched GB300 bitwise. The false `violation` defeated the
  certificate, so the exploration ran past 200k states. Second, the "half the bytes differ"
  report came from `compare`. It compared the output of the one explored cluster
  (`explore_symmetric` keeps a slice's partial output) with the whole launch's output. Now
  certified (2514 to 7542 steps), and all five pairs are proven L0. Fixture
  `c_mm_bf16_mcast`.
- **f32 mm x5** (cutlass3x SIMT sgemm, seq 64): L0, already with main at 67a8ef22. At seq 8
  (`--seq 8`, cuBLAS `gemmSN_TN_kernel`) the five mm pairs are also proven L0. That needed
  `VIMNMX3`, `STS.128 [R], RZ`, and per-register S-INIT for a partly stale `LDS.128` (padding
  words of the split-K reduction slots; fixtures `c_mm_f32_gemmsn`, `c_mm_f32_gemmsn_tn`).
- **f32 SDPA** (`fmha_cutlassF_f32`) lifts: `HMMA.1688.F32.TF32` is calibrated on GB300
  (N-HMMA: `g8-f25-rz-accin-tzero-einputs`, operands' low 13 mantissa bits truncated,
  probes/tmma). Also lifted: `FSET.BF`, `FCHK` with an immediate, `ATOMS.MIN` / `.MAX.S32`,
  `LDL` / `STL .U8`, and predicated-off `LDSM`. The float-max atomics (`MAX.S32` / `MIN.U32`
  mix) commute under C-ORDER. The kernel is certified clean and = GB300 (fixture
  `c_cve_sdpa_f32`). It stays blocked: 229 data-dependent branches (the taint check), and its
  outputs are untracked (TF32 MMA numerics; the kernel splits f32 operands into big + small
  TF32 parts with `LOP3` / `FADD`, which the numerics do not follow).
- **bf16 RoPE** (Triton, `FFMA2 ... -R16.F32`) lifts (negation flag on `Op::Ffma2`), is
  certified clean and = GB300 (fixture `c_cve_rope_bf16_inductor`). It stays blocked on
  16-bit outputs; eager and Inductor also differ by one bf16 ulp there (7.8e-3).
- Every new lowering is checked against GB300 by fixtures/s_lift_k: all pairs of the s_alu2
  operands, `hw` 1000/1000 bitwise.

**Update (2026-11-04, agent L): bf16.** Every bf16 Inductor (Triton) kernel is now proven
against its eager counterpart, 7 of 7 (`compare --recompare` on the same captures). The six extern
pairs (mm x5: 01, 06, 08, 09, 11; cuDNN SDPA: 04) run the same library kernel on both
sides and are proven L0 for every input (GPU bitwise equal on all outputs). All 13 bf16 pairs
are proven.

| # | compiled kernel | eager counterpart | level, bound (every input) | GPU: bitwise =, max \|C-R\| (fraction of the per-input bound) |
|---|---|---|---|---|
| 00 | RMSNorm + weight (`..._to_copy_add_mean_mul_pow_rsqrt_0`) | _to_copy, pow, mean, add, rsqrt, mul, _to_copy, mul | **L1**, norm form, output rounding peeled: \|C-R\| <= 196916u T(x); per input the distance between the bf16 roundings; the captured input: <= 1 ulp at all 8192 outputs | 6122/8192, 1.56e-2 (1.000) |
| 02 / 03 | RoPE, q / k | mul, mul, sub, mul, mul, add, stack | **L2**, 16-bit output form: \|C-R\| <= 2^-7 \|y\| + 65796u sum\|terms\| | 6415 / 6381 of 8192, 7.8e-3 (not compared: workspace W0 differs) |
| 05 | transpose copy (`clone`) | clone | **L0** (input elements moved byte for byte) | 8192/8192, 0 |
| 07 | residual add + RMSNorm | add, _to_copy, pow, mean, add, rsqrt, mul, _to_copy, mul | **L1**, norm form, peeled: \|C-R\| <= 271218u T(x) | 5426/8192, 1.56e-2 (0.667) |
| 10 | silu * mul | silu, mul | **elementwise, exhaustive**: every output is bf16(rn(f(x) y)); both kernels at all 2^32 (x, y) pairs: at most 1 bf16 ulp apart where f and the outputs are normal (15487 ulp at worst, subnormal silu(x) times y ~ 2^128) | 12031/16384, 7.8e-3 |
| 12 | residual add + add | add, add | **L2**, 16-bit output form: \|C-R\| <= 2^-7 \|y\| + 65796u sum\|terms\|; the captured input: 138 outputs proven bitwise equal, 6953 within 1 ulp | 6046/8192, 3.12e-2 (1.000) |

In every one of these pairs eager rounds to bf16 between its kernels (`type_as`, the silu
output, `x + a`, each mul of RoPE), so the two kernels compute different functions. The proofs
pivot on the exact polynomial (or normalization) y. Eager's inner roundings are charged as
errors, and the output rounding is peeled (docs/equiv-mma.md §2). The per-input bound is the
distance between the bf16 roundings of [y - G_R, y + G_C]. It is the tightest the two kernels'
Cor 1a bounds imply, and the GPU's differences reach it (fraction 1.000: one bf16 ulp where the
bound is one ulp) without exceeding it. One ulp is not implied in general: an intermediate
much larger than y (cancellation) carries 2^-8 of itself.

What it took (all in the numerics and equivalence files):

| blocker | fix |
|---|---|
| bf16 outputs had no node at 1 of 8 elements (eager add: `PRMT R12, R10, 0x7632, R12` with an unknown R12) | `PRMT` with an unknown other source also moves rounding-node bytes (`halfin.rs`) |
| Inductor RoPE: `FFMA2 R10, R11.F32, R6.F32, R10.F32` (lane 1 reads what lane 0 writes) clobbered both lanes | both lanes' nodes are made before either is written (`num.rs` `num_ffma2_seg`) |
| eager `sub` is `x * 1 + y` of bf16-rounded values: "a bf16-rounded value inside a product" | an exact constant factor k scales the absolute part by \|k\|; absolute parts pass through every rounding node (`equiv.rs`) |
| the copy kernel's outputs: input bytes, no node | `half_out_leaves`: such an output element is the input element's leaf |
| torch's bfloat16 copy kernel: `LDG.E.ENL2.256` lost its nodes | two 16-byte vector loads in the numerics (`exec.rs` `Op::Ld256`) |
| Inductor RMSNorm: `STSM.16.M88` (f32 words through shared memory) lost its nodes | non-transposed `stmatrix` moves whole words (`num_st`) |
| silu * mul: product of a rounded slice | 16-bit product outputs, swept at every (x, y) pair (`elementwise::product_sweep`) |
| RoPE was reported **NOT EQUIVALENT** with a witness whose two outputs were equal (1.0 vs 1.0) | a witness refutes only when its outputs differ (`compare.rs`) |

Fixtures (seq 64, library cubins slimmed): `c_cve_add_bf16_{eager,inductor}`,
`c_cve_clone_bf16_{eager,inductor}`, `c_cve_silumul_bf16_{eager,inductor}`; tests
`compare_bf16_residual_add_is_bounded`, `compare_bf16_clone_is_l0`,
`compare_bf16_silu_mul_is_exhaustive` (`crates/cli/tests/compare.rs`). The f32 Triton pairs
are unchanged (6 of 6 proven, same bounds).

**Update (2026-11-04, agent O): f32 SDPA.** Still the one blocked pair (24 of 25 proven). Its
TF32 numerics and most of its data-dependent control now have rules (docs/equiv-mma.md §6):
- `HMMA.1688.F32.TF32` steps are MMA nodes; the big + small split (`LOP3 0xffffe000`, exact
  `FADD`, `+0x1000`) is followed, and the three products per (a, b) are one term
  a b (1 + theta), |theta| <= 2^-19 (N-TF32). Shown on a TF32 GEMM pair (fixtures/c_equiv_tf32):
  3xTF32 vs an f32 SIMT GEMM is L1 with 239.52u · Σ|a b|, 1xTF32 32895.59u.
- The 229 data-dependent guards per kernel: 160 isfinite tests of the small part (N-FINITE
  decides them for finite inputs without overflow; 2 remain), 56 ex2.approx.f32 range-idiom
  guards (19 inside slices, 37 remain), 4 atomicMaxFloat branches (N-FMAX), 1 online-softmax
  `m_prime < mi` branch, 8 rcp.rn.f32 fast / slow paths (6 remain). CUTLASS's flag word
  (index masks and ex2 predicates as bits of R108) needed bitwise LOP3 taint.
- Remaining (61 per kernel, outputs untracked): the shared row maximum has no node (the warp
  maxima's slices fail), so `m_prime < mi` stays open; ex2 range-idiom slices fail (`P2R`, more
  than 1024 operations); ptxas's second rcp.rn filter (`VIADD x, 0x1800000; LOP3 0x7f800000;
  ISETP.GT.U32 0x1ffffff`) is not recognized; P reaches P·V through shared memory as failed
  slices. After those, N-ATTN needs TF32-aware weighted sums and CUTLASS's shared-memory
  bookkeeping (mi, m_prime, s_prime, out_rescale, per-warp row sums).
- Found on the way: the canonical forms dropped monomials whose coefficients cancel, with their
  error (a false bound for (x + y) - x; fixed, docs/equiv-mma.md §6).

**Update (2026-11-04, agent Q): f32 SDPA proven; 25 of 25 pairs.** The last blocked pair
(`fmha_cutlassF_f32_aligned_64x64_rf_sm80`, the same kernel and shape on both sides) is
**PROVEN bitwise identical (L0) for every admissible input** (finite elements, no overflow):
`compare --recompare` on the f32 captures, 8192/8192 outputs, relying on N-SEG (84 guards
inside slices), N-FMAX (4 diamonds) and N-FINITE (161 comparisons). All 12 f32 pairs are
proven (the 11 others unchanged: L1 x2, L2 x3, elementwise x1, L0 x5). What it took
(docs/equiv-mma.md §7):
- slices cross atomicMaxFloat diamonds, and are taken only in the bits of CUTLASS's flag word
  (R108) their readers use; `P2R`, `ISETP`, `VIADD` are slice operations;
- exact per-thread taint: SEL / FSEL by an input-independent predicate (the causal mask's
  -inf), per-thread guards, per-register loads, per-word stores, per-lane shuffles, per-thread
  atomic reads;
- N-FINITE on slice values: -inf - finite = -inf, so the masked exp2(-inf - mi) is the constant
  0; the RZ HMMA accumulator is a constant;
- ptxas's other rcp.rn forms (`VIADD` filter, quotient into b's register); the slow-path
  callee's registers are now checked dead too.
A bound against a different attention kernel (N-ATTN for 3xTF32 chains and CUTLASS's
shared-memory bookkeeping) is still open; this pair needs none.

## The block

`tools/compile_vs_eager.py` (`llama_block`): hidden 128, 2 heads (head dim 64), SwiGLU MLP
(256), seq 64, batch 1, causal SDPA, RMSNorm written the Llama way
(`x.float() * rsqrt(mean(x^2) + eps)`, `.type_as(x) * w`), RoPE by even/odd halves
(`stack([x1 cos - x2 sin, x1 sin + x2 cos]).flatten`), residual adds. f32 and bf16 (weights,
activations, cos/sin all in the dtype). torch 2.12, Triton 3.5, GB300 (GPU 3).

Inductor's f32 graph is 12 kernels: 6 Triton (2 RMSNorms, a RoPE kernel writing q and k, silu
* mul, the final residual add) and 6 extern (5 `mm`, efficient-attention SDPA). bf16 adds one
Triton kernel (the attention output transpose, `clone`) and uses cuDNN SDPA.

## Usage

```sh
source env.sh
# 1. eager + compiled forward passes, pairing, one ref/cand capture file pair per output
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/compile_vs_eager.py record --dtype f32 --out /tmp/cve_f32
# 2. capture both sides of every pair on GPU 3 and run ptxv compare (32 GB memory cap each)
$PTXV_PYTHON tools/compile_vs_eager.py run /tmp/cve_f32 [--only REGEX] [--recompare]
# 3. the table
$PTXV_PYTHON tools/compile_vs_eager.py report /tmp/cve_f32 /tmp/cve_bf16 --md table.md
```

`--seq N` shrinks the sequence (the committed fixtures use 8). `--recompare` reuses the
captures and only reruns `ptxv compare` (no GPU).

## How the pairing works

All automatic; nothing is hand-mapped.

1. **Eager.** The model runs under a `TorchDispatchMode` that records every ATen op as eager
   executes it (after CompositeImplicit decompositions: `F.linear` is `t` + `mm`, SDPA is
   `_scaled_dot_product_efficient_attention`), with tensor identity (data pointer, size,
   stride: views of one storage are told apart) and a copy of every output.
2. **Compiled.** `torch.compile` runs once; the Inductor output code (what
   `TORCH_LOGS=output_code` prints) is saved, and its loaded module is found in `PyCodeCache`.
   The graph inputs are read from `call`'s frame. `call` is run again with every Triton
   kernel's `run` wrapped (arguments by role: `in_ptr`, `out_ptr`, `in_out_ptr`; each
   storage's bytes before and after the launch) and the extern ATen ops recorded by the same
   dispatch mode.
3. **Data matching.** Each output buffer of a compiled kernel is matched to the eager tensor
   with the same dtype and element count and the smallest relative difference (graph inputs
   and weights match exactly; intermediates within 2% for f32, 8% for bf16, while unrelated
   tensors differ by O(1)). Each input buffer likewise.
4. **The eager sub-sequence** is the backward slice of the eager op graph from the matched
   output to the matched inputs (any tensor lying in a matched input's memory is an input,
   so views need no special case) or graph leaves. Views are replayed (no kernels); the rest
   are the eager kernels. Inductor's FX source nodes ("Topologically Sorted Source Nodes")
   are kept in `meta.json` for reference.
5. **Same bytes on both sides.** Both sides are replayed on the *compiled* kernel's input
   values, and inputs carry the same names in both cases (that is what lines up equivalence
   leaves). The eager side replays the ATen ops (`torch.ops.aten.*`, identical arguments and
   strides, so identical kernels); the compiled side imports the saved output code and calls
   the kernel's `run` (Triton) or the extern op. Each pair is two `capture.py --file`
   targets (`ref.py`, `cand.py`, `data.pt`, `spec.pt`), compared by `tools/compare.py`.
6. A compiled kernel with several outputs (the RoPE kernel writes q and k) gives one pair per
   output. An `in_out_ptr` output (Inductor reuses the input buffer) is an in-place output of
   an input of the same name.

## Results

Columns: the eager counterpart lists the kernels of the eager sub-sequence (views omitted);
verdicts are `ptxv check --certify` (cert. = E-WELLSYNC certificate), "sim = GPU" = the
simulator's output equals the captured GB300 output bitwise; GPU = the two captured GPU
outputs against each other (max |C - R|, bitwise-equal outputs / outputs). u = 2^-24.

### f32

| # | dtype | compiled kernel | eager counterpart (kernel ops) | eager verdict | compiled verdict | equivalence | GPU max \|C-R\| (bitwise =) | blockers |
|---|---|---|---|---|---|---|---|---|
| 00 | f32 | `triton_per_fused_add_mean_mul_pow_rsqrt_0` | pow, mean, add, rsqrt, mul, mul | clean (cert.), sim = GPU | clean (cert.), sim = GPU | **proven L1**: L1 (same function) for every input, \|C - R\| <= 4.18u*y + 4.00u | 4.77e-07 (7581/8192) | - |
| 01 | f32 | `extern aten.mm.out` | mm | clean (cert.), sim = GPU | clean (cert.), sim = GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (24576/24576) | outputs with no dataflow history over the inputs: reference 24576/24576, candidate 24576/24576 (e.g. output 0 has no dataflow history (va... |
| 02 | f32 | `triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_vi...` (out_ptr0) | mul, mul, sub, mul, mul, add, stack | clean (cert.), sim = GPU | clean (cert.), sim = GPU | **proven L2**: L2 (same function) for every input, \|C - R\| <= 4.00u * sum\|terms\| | 1.19e-07 (6523/8192) | - |
| 03 | f32 | `triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_vi...` (out_ptr1) | mul, mul, sub, mul, mul, add, stack | clean (cert.), sim = GPU | clean (cert.), sim = GPU | **proven L2**: L2 (same function) for every input, \|C - R\| <= 4.00u * sum\|terms\| | 2.38e-07 (6543/8192) | - |
| 04 | f32 | `extern aten._scaled_dot_product_efficient_attention.default` | _scaled_dot_product_efficient_attention | not lifted | not lifted | equivalence not attempted | 0.00e+00 (8192/8192) | reference: kernel _Z39fmha_cutlassF_f32_aligned_64x64_rf_sm80N22PyTorchMemE... does not lift, 6 kind(s) of unlifted instruction: `ATOMS v...; candidate: kernel _Z39fmha_cutlassF_f32_aligned_64x64_rf_sm80N22PyTorchMemE... does not lift, 6 kind(s) of unlifted instruction: `ATOMS v...; equivalence not attempted: a kernel does not lift |
| 05 | f32 | `extern aten.mm.out` | mm | clean (cert.), sim = GPU | clean (cert.), sim = GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (8192/8192) | outputs with no dataflow history over the inputs: reference 8192/8192, candidate 8192/8192 (e.g. output 0 has no dataflow history (value ... |
| 06 | f32 | `triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_2` | add, pow, mean, add, rsqrt, mul, mul | clean (cert.), sim = GPU | clean (cert.), sim = GPU | **proven L1**: L1 (same function) for every input, \|C - R\| <= 4.18u*y + 6.00u | 4.77e-07 (7540/8192) | - |
| 07 | f32 | `extern aten.mm.out` | mm | clean (cert.), sim = GPU | clean (cert.), sim = GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (16384/16384) | outputs with no dataflow history over the inputs: reference 16384/16384, candidate 16384/16384 (e.g. output 0 has no dataflow history (va... |
| 08 | f32 | `extern aten.mm.out` | mm | clean (cert.), sim = GPU | clean (cert.), sim = GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (16384/16384) | outputs with no dataflow history over the inputs: reference 16384/16384, candidate 16384/16384 (e.g. output 0 has no dataflow history (va... |
| 09 | f32 | `triton_poi_fused__unsafe_view_mul_silu_3` | silu, mul | clean (cert.), sim = GPU | clean (cert.), sim = GPU | **proven EW (f(x)*y)**: elementwise: every output is rn(f(x) * y), y another input element; over every f32 input (all 2^32 values), \|f_C - f_R\| <= 9.537e-7 (max 2 ulp, bit... | 2.38e-07 (12146/16384) | - |
| 10 | f32 | `extern aten.mm.out` | mm | clean (cert.), sim = GPU | clean (cert.), sim = GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (8192/8192) | outputs with no dataflow history over the inputs: reference 8192/8192, candidate 8192/8192 (e.g. output 0 has no dataflow history (value ... |
| 11 | f32 | `triton_poi_fused__unsafe_view_add_4` | add, add | clean (cert.), sim = GPU | clean (cert.), sim = GPU | **proven L2**: L2 (same function) for every input, \|C - R\| <= 4.00u * sum\|terms\| | 0.00e+00 (8192/8192) | - |

Counts: not proven: 6, proven EW (f(x)*y): 1, proven L1: 2, proven L2: 3

Blocker classes (unproven pairs): MMA (tensor-core) numerics / lifting: f32 04; lift gap: f32 04; missing normal form / numerics tracking: f32 01, f32 05, f32 07, f32 08, f32 10

### bf16

| # | dtype | compiled kernel | eager counterpart (kernel ops) | eager verdict | compiled verdict | equivalence | GPU max \|C-R\| (bitwise =) | blockers |
|---|---|---|---|---|---|---|---|---|
| 00 | bf16 | `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | _to_copy, pow, mean, add, rsqrt, mul, _to_copy, mul | clean (cert.), sim = GPU | clean (cert.), sim = GPU | NOT PROVEN (not shown to be the same function) | 1.56e-02 (6122/8192) | reference: 2 data-dependent branch(es)/guard(s) or address(es) (first: guard at _ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_1...; outputs with no dataflow history over the inputs: reference 4096/4096, candidate 4096/4096 (e.g. output 0 has no dataflow history (value ... |
| 01 | bf16 | `extern aten.mm.out` | mm | contract, violation, sim != GPU | contract, violation, sim != GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (24576/24576) | reference: violation: TMA#4 (issued by cta1.t128) with 128B swizzle: destination 0x2600 must be 1024-byte aligned and the inner box at mo...; reference: exploration not finished in 200019 states (no proof of race / deadlock freedom); reference: the simulator's output differs from the captured GPU output (out: MISMATCH vs the captured GPU output: 40865 of 49152 bytes di...; +4 more |
| 02 | bf16 | `triton_poi_fused__scaled_dot_product_cudnn_attention__unsafe_view_a...` (out_ptr0) | mul, mul, sub, mul, mul, add, stack | clean (cert.), sim = GPU | not lifted | equivalence not attempted | 7.81e-03 (6415/8192) | candidate: kernel triton_poi_fused__scaled_dot_product_cudnn_attention__uns... does not lift, 2 kind(s) of unlifted instruction: `operand...; equivalence not attempted: a kernel does not lift |
| 03 | bf16 | `triton_poi_fused__scaled_dot_product_cudnn_attention__unsafe_view_a...` (out_ptr1) | mul, mul, sub, mul, mul, add, stack | clean (cert.), sim = GPU | not lifted | equivalence not attempted | 7.81e-03 (6381/8192) | candidate: kernel triton_poi_fused__scaled_dot_product_cudnn_attention__uns... does not lift, 2 kind(s) of unlifted instruction: `operand...; equivalence not attempted: a kernel does not lift |
| 04 | bf16 | `extern aten._scaled_dot_product_cudnn_attention.default` | _scaled_dot_product_cudnn_attention | clean (cert.), sim = GPU | clean (cert.), sim = GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (8192/8192) | outputs with no dataflow history over the inputs: reference 4096/4096, candidate 4096/4096 (e.g. output 0 has no dataflow history (value ... |
| 05 | bf16 | `triton_poi_fused_clone_transpose_2` | clone | clean (cert.), sim = GPU | clean (cert.), sim = GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (8192/8192) | outputs with no dataflow history over the inputs: reference 4096/4096, candidate 4096/4096 (e.g. output 0 has no dataflow history (value ... |
| 06 | bf16 | `extern aten.mm.out` | mm | contract, violation, sim != GPU | contract, violation, sim != GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (8192/8192) | reference: violation: TMA#4 (issued by cta1.t128) with 128B swizzle: destination 0x2600 must be 1024-byte aligned and the inner box at mo...; reference: exploration not finished in 200018 states (no proof of race / deadlock freedom); reference: the simulator's output differs from the captured GPU output (out: MISMATCH vs the captured GPU output: 8174 of 16384 bytes dif...; +4 more |
| 07 | bf16 | `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3` | add, _to_copy, pow, mean, add, rsqrt, mul, _to_copy, mul | clean (cert.), sim = GPU | clean (cert.), sim = GPU | NOT PROVEN (not shown to be the same function) | 1.56e-02 (5426/8192) | reference: 2 data-dependent branch(es)/guard(s) or address(es) (first: guard at _ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_1...; outputs with no dataflow history over the inputs: reference 4096/4096, candidate 4096/4096 (e.g. output 0 has no dataflow history (value ... |
| 08 | bf16 | `extern aten.mm.out` | mm | contract, violation, sim != GPU | contract, violation, sim != GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (16384/16384) | reference: violation: TMA#4 (issued by cta1.t128) with 128B swizzle: destination 0x2600 must be 1024-byte aligned and the inner box at mo...; reference: exploration not finished in 200019 states (no proof of race / deadlock freedom); reference: the simulator's output differs from the captured GPU output (out: MISMATCH vs the captured GPU output: 24527 of 32768 bytes di...; +4 more |
| 09 | bf16 | `extern aten.mm.out` | mm | contract, violation, sim != GPU | contract, violation, sim != GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (16384/16384) | reference: violation: TMA#4 (issued by cta1.t128) with 128B swizzle: destination 0x2600 must be 1024-byte aligned and the inner box at mo...; reference: exploration not finished in 200020 states (no proof of race / deadlock freedom); reference: the simulator's output differs from the captured GPU output (out: MISMATCH vs the captured GPU output: 24513 of 32768 bytes di...; +4 more |
| 10 | bf16 | `triton_poi_fused__unsafe_view_mul_silu_4` | silu, mul | clean (cert.), sim = GPU | clean (cert.), sim = GPU | NOT PROVEN (not shown to be the same function) | 7.81e-03 (12031/16384) | reference: 8 data-dependent branch(es)/guard(s) or address(es) (first: guard at _ZN2at6native29vectorized_elementwise_kernelILi8EZZZNS0_5...; candidate: 4 data-dependent branch(es)/guard(s) or address(es) (first: guard at sass:0x01c0 `FMUL R6, R6, 0.25`); the taint check refuses...; outputs with no dataflow history over the inputs: reference 8192/8192, candidate 8192/8192 (e.g. output 0 has no dataflow history (value ... |
| 11 | bf16 | `extern aten.mm.out` | mm | contract, violation, sim != GPU | contract, violation, sim != GPU | NOT PROVEN for every input (L2 holds on the captured inputs only) | 0.00e+00 (8192/8192) | reference: violation: TMA#4 (issued by cta1.t128) with 128B swizzle: destination 0x2600 must be 1024-byte aligned and the inner box at mo...; reference: exploration not finished in 200020 states (no proof of race / deadlock freedom); reference: the simulator's output differs from the captured GPU output (out: MISMATCH vs the captured GPU output: 8177 of 16384 bytes dif...; +4 more |
| 12 | bf16 | `triton_poi_fused__unsafe_view_add_5` | add, add | clean (cert.), sim = GPU | clean (cert.), sim = GPU | NOT PROVEN (not shown to be the same function) | 3.12e-02 (6046/8192) | outputs with no dataflow history over the inputs: reference 4096/4096, candidate 4096/4096 (e.g. output 0 has no dataflow history (value ... |

Counts: not proven: 13

Blocker classes (unproven pairs): data-dependent control: bf16 00, bf16 07, bf16 10; lift gap: bf16 02, bf16 03; missing normal form / numerics tracking: bf16 00, bf16 01, bf16 04, bf16 05, bf16 06, bf16 07, bf16 08, bf16 09, bf16 10, bf16 11, bf16 12; other: bf16 01, bf16 06, bf16 08, bf16 09, bf16 11

bf16 notes: the RMSNorm, silu * mul and residual-add pairs differ by up to one bf16 ulp on
the GPU (1.6e-2, 7.8e-3, 3.1e-2) because eager rounds intermediates to bf16 between kernels
(`type_as`, the silu output, `x + a`) where Inductor keeps f32 inside the fused kernel: these
are different functions, provable only as bounds. The "other" class is the nvjet TMA model
gap below.

## Blockers, by class

| class | pairs | what exactly |
|---|---|---|
| **16-bit outputs** (missing normal forms) | bf16: both RMSNorms, silu * mul, residual add, the transpose copy, cuDNN SDPA (6) | The numerics shadow and every normal form express f32 output words; elementwise mode takes 16-bit outputs of one input element only. bf16 RMSNorm also changes function, not only rounding: eager rounds `type_as(x)` to bf16 before the weight multiply, Inductor keeps f32 (GPU max diff 1.6e-2 = one bf16 ulp near 2-4), so a proof there is a bound, not L0. The copy kernel (`clone`) has no floating-point operation at all, so its outputs have no dataflow history (data-movement leaves, ROADMAP rung 2e). |
| **cp.async + FFMA2 numerics** | f32 mm x5 | cuBLAS picks `cutlass3x_sm100_simt_sgemm_f32_..._32x32x16` (both sides, identical launch). It lifts, is certified clean and equals GB300 bitwise (after the `.E` addressing fix below), but its operands reach registers through `LDGSTS` (cp.async into smem, then `LDS`), which gives them no leaf provenance, and the dot products are `FFMA2`, which plain mode tracks only as x*1 + 0 (num.rs). Every output is "no dataflow history". |
| **lift gaps** | f32 SDPA; bf16 RoPE (q, k) | f32 SDPA (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`, both sides): `HMMA.1688.F32.TF32` x192 (f32 attention runs on TF32 tensor cores), `FSET.BF.GT.AND` x4, `ATOMS` variant x8, a float immediate operand (`1.4426950216293334961`) x1, predicate side output x2, sub-word local access x4. bf16 RoPE (Inductor): `FFMA2 R12, R19.F32, R21.F32, -R16.F32` (negated FFMA2 addend) x2: needs a negation flag on `Op::Ffma2` in exec.rs / num.rs. f32 mm at small M (8 rows): cuBLAS's `gemmSN_NN_kernel` needs `VIMNMX3` (3-input integer min; a new IR op in exec.rs). |
| **bf16 MMA numerics** (agent H) | bf16 mm x5, cuDNN SDPA | HMMA / tcgen05 results are not tracked. cuDNN's bf16 SDPA kernel lifts, is certified clean and = GB300; it is blocked on 16-bit outputs and MMA numerics. |
| **simulator model gap** | bf16 mm x5 | `nvjet_sm103_tst_64x8_64x16_2x4_h_bz_TNT` (cluster 8, multicast TMA; 64x64 nvjet tiles in fixtures/c_mm_bf16_* are fine): the simulator reports `TMA#4 (cta1.t128) with 128B swizzle: destination 0x2600 must be 1024-byte aligned` and its output differs from GB300 in half the bytes; exploration does not finish in 200k states. The destination is a per-CTA multicast slice (stage base 0x2400 + 0x200), so either the S-TMA alignment rule is too strict for slices or the slice offset is mis-modeled (exec.rs `complete_tma`). |
| **data-dependent control** | none on its own | The eager rsqrt kernel's subnormal guard (`@!P0 FMUL R18, R18, 16777216` under a `PLOP3`-combined predicate) is now inside instruction slices (N-SEG), see fixes. The bf16 RMSNorm / silu reports still list data-dependent guards (rsqrtf, expf range checks, Triton's `FMUL R6, R6, 0.25`) because their outputs stay untracked (16-bit), so the guards are not inside materialized slices. |

So for f32, the remaining blockers are all in library kernels that both sides share; for bf16
the dominant blocker is 16-bit outputs (6 of 13), then MMA numerics and the nvjet model gap
(6, overlapping), then two lift gaps.

## Fixes made (all validated on GB300)

| fix | where | what it unblocked | validation |
|---|---|---|---|
| warm-up restores inputs changed in place | `tools/capture.py` | every Inductor `in_out_ptr` kernel was captured on its own warm-up result (silu * mul: GPU max diff 2.19 before, 2.4e-7 after) | captured outputs = GPU |
| launchcap re-hooks `cuGetProcAddress` for every runtime (not only `LAUNCHCAP_VMM`) | `probes/launchcap/launchcap.c` | cuBLAS's statically linked cudart launched `gemmSN_NN_kernel` (8x128 @ 128x384 f32 mm) unrecorded: "no kernel launches recorded" | captures 1 launch; softmax capture unchanged |
| `.E` global / generic accesses use a 64-bit address pair even when nvdisasm prints `[R10]` (as REDG already did) | `crates/sass` `wide_e_addr` (L-NVDISASM / L-ROSETTA) | cuBLAS's SIMT sgemm faulted (`global address 0x600 outside every allocation`, sim != GPU) | all 5 f32 mm pairs: certified clean, sim = GB300 bitwise |
| a `SHFL` cannot overwrite a register an earlier memory-queue instruction has not read, also when that read has a read scoreboard | `crates/sass/src/scoreboard.rs` (L-OPREAD) | Inductor's persistent RMSNorm was a `violation` (`SCOREBOARD WAR: R41 overwritten at SHFL.BFLY while STG.E.128 R40 (SB0) may be outstanding`); ptxas's own schedule | fixtures/c_cve_addnorm_inductor: `hw` 1000/1000 = simulator |
| `PLOP3` in instruction slices | `crates/sim/src/seg.rs` (N-SEG) | torch's rsqrtf (eager RMSNorm) had no slice ("Plop3 is not modeled in a segment"), so every RMSNorm output was untracked | fixtures/c_cve_addnorm_eager: `hw` 1000/1000; 20 random trials (163,840 outputs), 0 over the bound |
| in-place outputs are inputs too (leaves, taint, trials, witnesses) | `crates/cli/src/equiv.rs` `out_only`, `compare.rs` | residual add (in_out_ptr) was "NOT EQUIVALENT" with a bogus witness: its initial elements had no origin and were not tainted | proven L2, GPU bitwise equal |
| same-named workspaces (`W0`) are not compared | `equiv.rs` pre_pair, `compare.rs` inputs_differ | RoPE: "pair: buffer W0 is u8 [2048] in the reference but u8 [4096] in the candidate" | proven L2 |
| elementwise **product outputs** rn(f(x) * y) | `crates/sim/src/elementwise.rs` `product_of`, `equiv.rs`, `compare.rs` (N-ELEMENTWISE) | silu * mul: both sides are `fma(slice(a) * b)`, slices differ, two leaves | f swept over all 2^32 f32 values; outputs = GB300 |
| fresh outputs with `torch.empty` (a `zeros` adds a fill kernel to the candidate) | `tools/compile_vs_eager.py` | - | - |

The product claim: if f_C = f_R bitwise for every x, the outputs are bitwise equal for every
(x, y) (the same correctly rounded product); otherwise |C - R| <= |y| max|f_C - f_R| +
2^-24 (|C| + |R|) + 2^-149 (each rounding is relative 2^-24 of its result, or absolute
2^-150 below the normal range).

## Fixtures and tests

Captured with `record --dtype f32 --seq 8` (1024 / 2048 outputs, so the test is quick), each
library cubin slimmed by `tools/cubin_slim.py`:

| fixture | what | check | GB300 |
|---|---|---|---|
| `c_cve_addnorm_eager` | eager residual add + RMSNorm: 7 kernels (add, pow, mean, add, rsqrt, mul, mul) | clean (cert.), 246 states, = captured GPU output | `hw` 1000/1000 |
| `c_cve_addnorm_inductor` | `triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_2` | clean (cert.), 7 states | `hw` 1000/1000 |
| `c_cve_silumul_eager` | eager silu kernel + mul kernel | clean (cert.), 98 states | `hw` 1000/1000 |
| `c_cve_silumul_inductor` | `triton_poi_fused__unsafe_view_mul_silu_3` (in_out_ptr) | clean (cert.), 64 states | `hw` 1000/1000 |
| `c_mm_bf16_mcast` | `a @ b.t()`, 64x128 @ 128x128 bf16: nvjet `64x8_64x16_2x4` (cluster 8, multicast TMA slices) | contract E-BULK-EXIT (cert.), 2514 steps | `hw` 1000/1000, also scrubbed |
| `c_cve_rope_bf16_inductor` | bf16 RoPE Triton kernel (seq 64, `FFMA2 -R.F32`) | clean (cert.) | `hw` 1000/1000 |
| `c_cve_sdpa_f32` | `fmha_cutlassF_f32_aligned_64x64_rf_sm80` (seq 64, TF32 HMMA) | clean (cert. only: its atomics interleave past 200k states when explored) | `hw` 1000/1000 |
| `c_mm_f32_gemmsn` | `torch.mm` 8x128 @ 128x384 f32: `gemmSN_NN_kernel` (`VIMNMX3`, `STS.128 RZ`) | clean, 1585 states | `hw` 1000/1000 |
| `c_mm_f32_gemmsn_tn` | seq-8 pair 05 mm: `gemmSN_TN_kernel` (partly stale `LDS.128`) | clean, 1025 states | `hw --scrub` 1000/1000 |

- `crates/cli/tests/compare.rs`:
  - `compare_inductor_add_rmsnorm_vs_eager_is_l1`: L1 proven (norm form, |C - R| <= 4.18u*y
    + 6u), 1024 of 1024 outputs bounded, no blockers;
  - `compare_inductor_silu_mul_vs_eager_is_elementwise_product`: proven as product outputs,
    f identified as silu, max |f_C - f_R| < 2e-6 (the sweep is committed in
    `fixtures/c_equiv_elementwise/sweeps.elementwise.json`).
- `crates/sim/tests/fixtures.rs` `captured_outputs_match_the_gpu` covers these fixtures;
  `s_lift_k_matches_hardware` (fixtures/s_lift_k) and `probe_tmma_matches_hardware`
  (probes/tmma) cover the forms lifted for them.

## Limits of the tool

- One block, one stream, forward only, no dynamic shapes; Inductor's `default` mode (no
  CUDA graphs, no autotuning of mm templates).
- The candidate re-imports the output code in a fresh process; Inductor may pick another
  config for a kernel with several (the recorded launcher config is reapplied when the
  kernel has several).
- Data matching needs the compiled intermediate close to the eager one; a kernel whose output
  matches no eager tensor (or matches a graph input) is reported, not paired. None occurred.
