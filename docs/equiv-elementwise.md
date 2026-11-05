# Elementwise equivalence: exhaustive over every input value

**Status (2026-11-03).** `ptxv equiv` (and `ptxv compare`) prove elementwise kernels (gelu, silu,
tanh-GELU, exp, ...) equivalent over **every** input of their element type. For each pair the
report gives the exact largest difference between the two kernels and each kernel's exact
largest error against the true function in f64. The sweeps cover all 2^32 f32 bit patterns or
all 2^16 bf16 / f16 patterns. There is no normal form for these kernels, so nothing else
proves them. Before this, `compare` reported such pairs as blocked ("no dataflow history",
"not the same function").

| pair (torch vs Triton) | input | slice ops R / C | bitwise equal | max \|C - R\| | R vs true: \|err\|, ulp (\|f\| >= 2^-10) | C vs true: \|err\|, ulp (\|f\| >= 2^-10) | GB300 check |
|---|---|---|---|---|---|---|---|
| `F.gelu(x, 'tanh')` vs Triton on `tanh.approx` | f32 | 25 / 8 | 97.13% | 5.61e-6 (x = -1.404) | 4.33e-7, 440 | 5.66e-6, 31222 | 73,728 / 73,728 each |
| `F.gelu(x)` vs Triton `tl.math.erf` | f32 | 47 / 40 | 99.89% | 4.77e-7 (x = 4.0002) | 4.86e-7, 762 | 4.48e-7, 457 | 73,728 / 73,728 each |
| `F.silu(x)` vs Triton `x / (1 + exp(-x))` | f32 | 9 / 40 | 96.60% | 1.91e-6 (x = 12.12), max 65 ulp | 1.27e-6, 3.06 | 1.40e-6, 9.05 | 73,728 / 73,728 each |
| `F.gelu(x)` vs Triton erf-GELU | bf16 | 95 / 81 | 99.98% | 1.53e-5 (x = -3.14) | 7.77e-3, 0.503 | 7.77e-3, 0.500 | 65,536 / 65,536 each (every bf16 input) |

- **Columns.**
  - "bitwise equal": the share of input values where both outputs have the same bits.
  - "\|err\|": the largest absolute error over finite inputs.
  - "ulp": the largest |K - T| / ulp(T) where |T| >= 2^-10 (output format).
  - "GB300 check": outputs where each kernel, run on GB300, equals its element function bit
    for bit (below).
- **Where the inputs are.** The fixtures are in `fixtures/c_equiv_elementwise/{gelu_tanh,
  gelu_erf, silu, gelu_bf16}.toml`. They use the captured cases `c_gelu_tanh`,
  `c_triton_gelu_tanh`, `c_gelu_erf`, `c_triton_gelu_erf`, `c_silu`, `c_triton_silu`,
  `c_gelu_bf16` and `c_triton_gelu_bf16` (all 4x256, `randn * 4`).
- **Cost.** One f32 pair takes 20-40 s on 144 threads per distinct pair of functions. A bf16
  pair takes under a second.
- **Sweep cache.** The results are cached in `sweeps.elementwise.json` next to the
  `sweep_cache`.

The largest *ulp* distance between the two kernels is meaningless where an output is near
zero. In these places one kernel cancels to 0 and the other does not:
- tanh-GELU: 891,036,724 ulp at x = -4.88 (-5.8e-7 vs -0);
- erf-GELU: 875,651,124 ulp at x = -5.54 (-0 vs -1.65e-7).

`0.5 x (1 + tanh(u))` cancels in f32 for x < -4.9, and Triton's `tanh.approx` returns exactly
-1 there. The report therefore gives the absolute and relative differences, and ulp errors
against the true function where |f| >= 2^-10. The largest ulp error over all inputs is
also reported, at x around -9. Every kernel reaches about 2^24 ulp there, by flushing a true
value near 2^-126 to 0.

## Method

1. **Element functions.** In segment mode (`docs/equiv-reassoc.md` §3) an output's numerics
   node is a small DAG made of these parts:
   - instruction slices (`Seg`: everything between tracked nodes, data-dependent guards as
     both outcomes);
   - tracked products and sums (`Fma`, `Add`);
   - constants;
   - `Input` leaves.

   `elementwise::extract` turns each output's DAG into an `ETree` and requires a single
   input leaf. The taint check gives control flow and addresses independent of input data,
   so the DAG is that output's function for every input. Outputs whose trees are equal up to
   register names and pcs share one function. The structural key renames each slice
   operation's registers by their role. Vectorized kernels (4 or 8 elements per thread, each
   on its own registers) therefore have 1 or 2 functions, not one per lane.
2. **The pairing.** Output i must read the same (buffer, element) in both kernels.
3. **16-bit outputs.** A bf16 / f16 output word packs two results:
   `F2FP.BF16.F32.PACK_AB d, a, b` puts a in the high half and b in the low half.
   - Each half is the operand's slice. Its dependence on the slice arguments is computed by
     dataflow over the slice: FFMA2 lanes count only their own registers.
   - That operand must depend on one leaf. The other leaf is then bound to 0, which is
     sound because the half does not read it.
   - bf16 outputs are now read element by element (docs/equiv-mma.md §2). Each element's node
     is the bf16 rounding (N-ROUND) of the F2FP operand's slice, so the function is
     `round bf16` of that slice, with the same dependence rule. The F2FP itself is no longer
     part of the slice, so the slice-op counts in the table above dropped (the bf16 GELU pair
     is now 47 / 79); the swept results are unchanged.
4. **The sweep.** `elementwise::sweep` evaluates both functions on every input pattern,
   threaded, and accumulates:
   - bitwise-equal count;
   - both-NaN count;
   - special-value mismatches (NaN / inf on one side only);
   - the largest absolute, relative and ulp distance where both are finite, with the input
     where it occurs;
   - the outputs at +inf, -inf and NaN;
   - each kernel against the true function in f64. Inputs whose true value overflows the
     output format are counted apart.
5. **The true function.** It is `function = "gelu"` in the equiv.toml, `--function` for
   `compare`, or identified from the kernels:
   - identification picks the library function nearest both kernels on about 900 sample
     points, if it is within 2^-5 and 8x nearer than any other;
   - gelu and gelu_tanh differ by about 0.2 on this measure, the kernels by about 0.004;
   - the library is `identity, neg, abs, relu, exp, exp2, log, sqrt, rsqrt, reciprocal, tanh,
     sigmoid, silu, gelu, gelu_tanh, erf, sin, cos, softplus`.

   The f64 forms avoid cancellation:
   - gelu = 0.5 x erfc(-x/√2);
   - gelu_tanh = x · sigmoid(2u);
   - erf / erfc: a Taylor series below 3 and a continued fraction above, to less than 1e-13
     relative.
6. **The level.**
   - `L0` if the two kernels are bitwise equal on every input.
   - Otherwise `EW` ("elementwise, exact largest difference over every input"). It is proven
     when the usual preconditions hold: clean on every schedule, and the taint check except
     guards inside slices (N-SEG).
   - `compare` headline: "PROVEN elementwise for every f32 input (all 2^32 values): |C - R| <=
     ..., max N ulp apart, bitwise equal on P% of inputs; vs gelu (f64): reference |err| <= ...,
     candidate |err| <= ...".

### What changed to make the slices complete

These changes are additive, in `seg.rs` and `num.rs`. They are listed under N-SEG in
ASSUMPTIONS.md.
- **Guard decisions.** torch's erfcf writes `@P MUFU.EX2 R; @P FADD R, -R, 1; @P LOP3 S, R,
  ...`, so R is defined only where P holds. Before, the slicer also built R's "P false"
  side, which reads R before any write, and gave up. Now the `new` side of a `Guard` assumes
  its predicate: another guard on the same predicate value (the same write) takes its `new`
  side. Guard predicates are sliced without assumptions, and each write is one `SVal`, so
  "the same predicate" is exact.
- **div.rn.** ptxas's `div.rn.f32` sequence is also recognized when the slow-path callee
  writes the quotient register itself. torch's vectorized silu has no `MOV q, res` after the
  call.
- **New ops in slices.** `F2FP.BF16/F16.F32.PACK_AB`, `PRMT`, FFMA2 / FMUL2 / FADD2 per lane,
  and the bf16 widening op.
- **FFMA2 immediates.** `FMUL2 / FADD2` with an f32 immediate (`FMUL2 R4, R2.F32x2.HI_LO, 0.5`)
  are lifted as a broadcast of the encoding's immediate word. `c_triton_gelu_bf16` matches
  GB300 bitwise with it.

## Validation on GB300

`elementwise_gpu = N` in an equiv.toml runs each kernel on a free GPU (`hw-worker`,
`PTXV_GPUS` restricts which) for N launches, with its input buffer replaced each time:
- launch b holds bf16 bit patterns 1024b .. 1024b + 1023, widened for f32 inputs, so all
  2^16 bf16 patterns are covered in 64 launches of 1024 elements;
- later launches hold random patterns.

Every output is compared with its element function's value at that input, bit for bit. For
the four pairs above, with N = 72 for f32 and 64 for bf16, no output differs in either kernel.
For bf16 this covers every input, so the bf16 GELU pair's sweep is reproduced on hardware
in full.

## bf16 and f16 inputs in the numerics

Before this, the numerics shadow tracked f32 input words only. A bf16 kernel's outputs had
no dataflow history ("the inputs are not f32 ..."). The new `halfin.rs` (N-HALF-LEAF)
follows the provenance of each byte of registers loaded from an unwritten bf16 / f16 input.
That includes packed words and `LDG.E.U16`. It tracks them through the widening idioms:
- `SHF.L.U32 d, r, 0x10`;
- `IMAD.U32 d, r, 0x10000, RZ`;
- `LOP3.LUT d, r, 0xffff0000, RZ, 0xc0`;
- `PRMT` byte moves;
- the bf16 conversion op;
- `HADD2.F32 d, -RZ, r.H0_H0` for f16.

A register holding exactly one element widened to f32 gets an `Input` leaf (allocation,
element index). Its value is exact. Sums, products and slices over it then work as for f32.
`equiv` trials and the captured-input bound draw and read bf16 inputs.

Shown on two pairs:

| pair | level | bound | trials | GPU (captured) |
|---|---|---|---|---|
| `x.sum(-1, dtype=torch.float32)` vs `x.float().sum(-1)`, x bf16 4x128 (torch reduce kernel vs elementwise copy + reduce, 1 vs 2 kernels) | **L1** proven | 19u · Σ\|x\| | 20, 0 over the bound | 4/4 bitwise equal |
| `F.gelu(x)` vs Triton erf-GELU, bf16 | **EW** proven | table above | - | 1006/1024 bitwise equal |

## Limitations

- **One output, one input element.** Binary elementwise ops (`a + b`, `x * y`) have two
  leaves and are refused. They are a 2^64 sweep in f32, or 2^32 in bf16, which would be
  feasible but is not implemented. One exception: **product outputs** rn(f(x) * y), one f32
  multiply by another input element y (the same on both sides): f is swept, and equal f gives
  equal outputs; otherwise |C - R| <= |y| max|f_C - f_R| + 2^-24 (|C| + |R|) + 2^-149. This
  proves Inductor's fused `silu(a) * b` against eager's silu kernel + mul kernel
  (docs/compile-vs-eager.md, N-ELEMENTWISE).
- **Unsupported node kinds.** Elementwise `Max` nodes (relu written as `FMNMX`) are refused.
  Their NaN semantics are not recorded on the node.
- **Untracked stores.** 16-bit outputs stored by `STG.E.U16` (one element per store) are not
  materialized (`num_st` tracks 4-byte words).
- **Runtime.** The sweeps take seconds to minutes per pair, which is why the results are
  cached.

## Assumptions

- **N-ELEMENTWISE** (validated): elementwise mode's claim, the true-function accuracy, and
  identification.
- **N-HALF-LEAF** (argued): bf16 / f16 leaves.
- **N-SEG** (updated): guard decisions, the div.rn slow-path variant, and the new slice ops.
- **L-ROSETTA** (updated): FFMA2 immediates.
