# `ptxv compare`: one report per kernel pair

**Status (2026-11-03).** One command compares a candidate kernel with a reference. It reports:
- each side's execution verdict;
- the equivalence level;
- a worst-case error bound for each output;
- how much of that bound the GPU's actual difference uses on the captured inputs;
- why anything is unproven.

It runs on 18 pairs. Of those, 14 are proven:
- one L0;
- nine L1 with bounds, among them a bf16 sum;
- four elementwise pairs (gelu, tanh-GELU, silu in f32; gelu in bf16), swept over every input
  value (docs/equiv-elementwise.md).

One pair is shown not equivalent, with a witness input. Three are blocked, and the report
names the reason: layer_norm, sort vs topk (sorting has no normal form), and the bf16 matmul
pair (bf16 outputs, transposed layout).

## Usage

```sh
source env.sh
# Capture both sides on GPU 3 (the same seeded inputs for both), then compare on the CPU:
$PTXV_PYTHON tools/compare.py --ref 'torch.softmax(x,-1)' \
    --cand tools/capture_examples/triton_softmax.py:softmax_kernel \
    --inputs 'x=randn(4,128,f32)' --out /tmp/cmp_softmax
# Two existing case directories (fixtures, or earlier captures); no GPU needed:
ptxv compare fixtures/c_cat fixtures/c_triton_cat --out /tmp/cmp_cat
```

`tools/compare.py --ref X --cand Y`. Each of X and Y is one of:
- a case directory (it has a `case.toml`);
- `FILE.py:KERNEL`, a Triton kernel (`capture.py --triton --binary`);
- `FILE.py`, everything `run()` launches;
- any other text, which is treated as a torch expression (`capture.py --op`).

`--inputs` is passed to both captures with the same `--seed`, so both sides read the same bytes.
`capture.py --inputs` now also overrides a file's `inputs()`. Options:
- `--gpu` (default 3; 2 is refused);
- `--kernels REGEX`;
- `--max-states` (default 200000);
- `--trials` (default 20);
- `--segments true|false`;
- `--sweep-cache`;
- `--function NAME` (elementwise pairs: the real function, e.g. `gelu`, `gelu_tanh`, `silu`;
  default: identified from the kernels);
- `--ptxv PATH`.

The comparison step runs with `CUDA_VISIBLE_DEVICES=""`.

`ptxv compare REF CAND [--out DIR] [--max-states N] [--trials N] [--segments B] [--sweep-cache F]
[--hw] [--ref-label S] [--cand-label S]` writes `DIR/report.json` and `DIR/report.md`, and prints
a short summary. Long runs print progress on stderr.

Equivalence covers the first declared output buffer, the final result of a multi-launch
region. Additional buffers can hold intermediates and are not compared; `output_scope` in
the report names the checked and unchecked buffers, and the headline identifies partial
output scope. Cases with no output buffer are rejected. Values that depend on inputs must
retain numeric history even when their captured value is zero; missing history cannot
establish equivalence.

## What it does

1. **Each side** (`crates/cli/src/compare.rs`, in process; a panic becomes a blocker).
   - **Lift.** It lifts on the SASS front: the captured cubin, or ptxas's output for a PTX
     case. A PTX case falls back to the PTX front if that fails. If neither lifts, it lists
     every kind of instruction the lifter does not lower, per kernel, with counts
     (`ptxv_sass::lift_report`).
   - **Execution verdict.** First the E-WELLSYNC certificate (`check --certify`).
     - One CTA: if the certificate fails, it explores up to `--max-states`.
     - Several CTAs: if the certificate fails, it does not explore the whole launch. It
       explores one cluster at a time (`explore_symmetric`: symmetry, else every cluster; fp32
       atomic adds are reductions).
   - **Simulator vs GPU.** The simulated output is compared with the captured GPU output. For a
     nondeterministic launch, it checks whether one of the simulated outputs equals the GPU's.
2. **Equivalence.** It runs `ptxv equiv` as a child process on a generated `equiv.toml`, and
   keeps the log as `equiv.log`.
   - Preconditions use the certificate (new `certify = true` per kernel), or `slices` when step 1
     needed them.
   - Plain numerics run first. If a kernel uses MUFU, or outputs have no dataflow history, and
     some outputs are left without a proven bound, it reruns in segment mode (softmax, norms)
     and keeps the better result.
   - In segment mode, elementwise mode is tried too: every output one input element's
     function, both kernels swept over every input value (docs/equiv-elementwise.md). The
     headline then gives the exact largest difference and each kernel's error against the true
     function.
   - The sweep cache defaults to `DIR/sweeps.json`, seeded from
     `fixtures/c_equiv_softmax/sweeps.json`.
3. **Captured GPU outputs.** The two captured GPU outputs are compared element by element:
   bitwise-equal count, max absolute and relative difference, max ulp. bf16 and f16 outputs
   are widened to f32. This needs the inputs of both cases to hold the same bytes; if they
   don't, the report says so.
   - When a bound is proven, `equiv` evaluates it at the captured inputs (`captured` in
     equiv.json), which gives the fraction of the bound the GPU difference uses.

### The report

- **headline:**
  - "PROVEN L0 / L1 ... |C - R| <= bound";
  - "NOT EQUIVALENT: with every input 0 except x[127] = 1, out[0] is 1 vs 0";
  - "NOT PROVEN ...", followed by the GPU numbers.
- **per side** (`reference.side`, `candidate.side`):
  - kernels, lifted or not, and `unlifted` (instruction, count, example);
  - `check`: verdict, complete, method, certified, states, assumptions, notes, findings;
  - `sim_vs_gpu`.
- **`equivalence`:**
  - level and proven, plus each of L0, L1 and L2;
  - `per_output_bound`: max / median / min, a histogram (exact values if 8 or fewer
    distinct, else 8 bins), and each kernel's own maximum against the exact value;
  - for softmax, the form `rel u * y + add u`;
  - simulator trials;
  - `captured` (the bound at the captured inputs, per output);
  - witness;
  - `relies_on`.
- **`gpu`:** outputs, bitwise_equal, max_abs_diff, max_rel_diff, max_ulps, worst output and
  values, and `max_fraction_of_bound`. Display only: no verdict, level or headline status reads
  them. Non-finite elements do not enter the maxima: an element whose difference is NaN (a NaN
  on either side, or the same infinity on both) adds nothing to max_abs_diff and max_rel_diff,
  and max_ulps compares only elements finite on both sides. An infinity against a finite value
  gives max_abs_diff = inf. Read bitwise_equal against outputs for the rest: an element that
  is NaN on one side only is counted as differing there and nowhere else.
- **`assumptions`:** the execution assumptions used (ASSUMPTIONS.md IDs).
- **`relies_on`:** what the verdicts rely on beyond those, per side: E-BULK-EXIT, N-ATOMIC,
  N-SEG.
- **`blockers`:** why something is unproven. One line per cause:
  - unlifted instructions (`unsupported SASS opcode REDUX.OR` x1, ...);
  - findings, grouped by kind, with the pc;
  - an exploration that did not finish;
  - data-dependent branches, guards or addresses (count and the first pc);
  - outputs with no dataflow history (count per side, and whether the inputs are not f32);
  - "not the same function" (the differing monomial);
  - outputs without a bound, grouped by reason;
  - nondeterminism that equiv did not excuse;
  - inputs or shapes that differ between the cases;
  - the simulator disagreeing with the GPU.

## Sample report (`tools/compare.py --ref 'torch.softmax(x,-1)' --cand triton_softmax.py:softmax_kernel --inputs 'x=randn(4,128,f32)'`)

Summary (stdout):

```
reference (torch.softmax(x,-1)): ref [sass]: clean (certified, E-WELLSYNC, 26 states), sim = GPU bitwise
candidate (triton_softmax.py:softmax_kernel): cand [sass]: clean (certified, E-WELLSYNC, 172 states), sim = GPU bitwise
=> PROVEN L1 (same function) for every input, |C - R| <= 44.95u*y + 1.24u; GPU: 174/512 outputs bitwise equal, max |C - R| 1.490e-8 (0.034 of the bound)
   per-output bound: max 46.19u, median 46.19u, min 46.19u (512 of 512 outputs bounded)
   relies on: reference: N-SEG (8 data-dependent guards inside instruction slices); candidate: N-SEG (6 data-dependent guards inside instruction slices)
```

report.md (52 s end to end, including both captures):

```
# ptxv compare

**PROVEN L1 (same function) for every input, |C - R| <= 44.95u*y + 1.24u; GPU: 174/512 outputs bitwise equal, max |C - R| 1.490e-8 (0.034 of the bound)**

| | reference | candidate |
|---|---|---|
| what | `torch.softmax(x,-1)` | `triton_softmax.py:softmax_kernel` |
| kernels | `_ZN43_GLOBAL__N__a4dace70_10_SoftMax_cu_9f978f6320softmax...` | `softmax_kernel` |
| lifts | yes (sass) | yes (sass) |
| execution verdict | clean, certified, E-WELLSYNC (26 states) | clean, certified, E-WELLSYNC (172 states) |
| sim vs GPU (captured input) | out: bitwise equal to the captured GPU output (2048 bytes) | out: bitwise equal to the captured GPU output (2048 bytes) |

## Equivalence

- level: **L1** (proven for every input)
  - L0 bitwise: fails at 512 outputs
  - L1 same function: holds
  - L2 no less accurate: fails at 512 outputs
- per-output bound (|C - R| <= rel_u u y + add_u u, y the exact output (<= 1)): max 46.19u, median 46.19u, min 46.19u; 512 of 512 outputs bounded
- each kernel against the exact value (largest over outputs): reference 19.61u, candidate 26.59u
- softmax form: |C - R| <= 44.95u * y + 1.24u
- histogram of the bound over outputs:

  | bound (u) | outputs |
  |---|---|
  | 46.19 | 512 |

- simulator trials: 20 random inputs, 0 over the bound, at most 0.0501 of the bound used
- softmax: softmax normal form exp(+x[0] -max(128 values)) / sum of 128 exp(x_j - max); contracts exp r0=2.36u r1=0.110u/|d| ...; rcp r0=1.65u ...

## Captured GPU outputs

- 174 of 512 outputs bitwise equal; max |C - R| = 1.490e-8 (out[393]: 0.19045305252075195 vs 0.19045306742191315), max relative 4.326e-7, max 5 ulp
- largest |C - R| / (bound at the captured input): 0.0342 (out[69])

## Rests on

- reference: N-SEG (8 data-dependent guards inside instruction slices)
- candidate: N-SEG (6 data-dependent guards inside instruction slices)
- execution assumptions (ASSUMPTIONS.md): E-ALIGNED E-BAR-EXIT E-BSYNC E-CONST-BRA E-CTA-INDEP E-DRF E-GRID E-LOCKSTEP E-RET E-WARPCOLL E-WELLSYNC L-CTRL L-MEMSB L-NVDISASM L-OPCLASS L-OPREAD L-ROSETTA L-STALL N-FADD N-FALU N-FCHK N-MUFU S-BRA-COND S-CBANK S-GLOBAL S-INIT S-SMEM-STRONG

## Not proven, and why

Nothing: every step is proven.
```

In the cumsum report, the bound histogram has several bins. Output j of a row is a sum of
j + 1 terms, so its bound grows with j:

```
  | 0.00 – 1.88 | 4 |   | 7.50 – 9.38 | 32 |
  | 1.88 – 3.75 | 4 |   | 9.38 – 11.25 | 64 |
  | 3.75 – 5.63 | 8 |   | 11.25 – 13.13 | 128 |
  | 5.63 – 7.50 | 16 |  | 13.13 – 15.00 | 256 |
```

## Demos

u = 2^-24.
- **Captured on GPU 3 by `tools/compare.py`** (both sides on the same inputs): softmax,
  sum vs transpose+sum, sum bug, gelu, layernorm, sort/topk, mm bf16.
- **Committed fixture pairs:** cat, sums, cumsum, softmax_2pass_fix, softmax_online.
- **Column meanings:**
  - "bound": the per-output bound's max / median;
  - "GPU =": captured outputs bitwise equal / total;
  - "frac": the GPU |C - R| over the bound at the captured input.

| pair | reference | candidate | level | bound max / median | GPU = | GPU max \|C − R\| | frac |
|---|---|---|---|---|---|---|---|
| torch.cat vs Triton cat (`c_cat`, `c_triton_cat`) | clean (cert.) | clean (cert.) | **L0** | 0 | 512/512 | 0 | - |
| torch.sum vs Triton butterfly (`r4_sum_torch`, `r4_rowreduce_triton`) | clean (cert.) | clean (cert.) | **L1** | 15u · Σ\|x\| | - (inputs differ) | - | - |
| torch.sum vs Triton split-K, 2 launches | clean (cert.) | clean (cert.) | **L1** | 15u · Σ\|x\| | 0/4 | 9.5e-7 | 0.011 |
| torch.sum vs Triton fp32 atomics | clean (cert.) | nondeterministic, relies on N-ATOMIC | **L1** | 16u · Σ\|x\| | 0/4 | 9.5e-7 | 0.010 |
| torch.cumsum vs Triton tl.cumsum | clean (cert.) | clean (cert.) | **L1** | 15u / 13u · Σ\|x\| | 351/512 | 1.9e-6 | 0.593 |
| torch.softmax vs Triton two-pass (fixtures) | clean (cert.) | clean (cert.) | **L1** | 44.95u·y + 1.24u | - (inputs differ) | - | - |
| torch.softmax vs Triton online | clean (cert.) | clean (cert.) | **L1** | 53.85u·y + 1.24u | 155/512 | 3.0e-8 | 0.041 |
| `torch.softmax(x,-1)` vs `triton_softmax.py` (captured, f32) | clean (cert.) | clean (cert.) | **L1** | 44.95u·y + 1.24u | 174/512 | 1.5e-8 | 0.034 |
| `torch.sum(x,-1)` vs `x.t().contiguous().sum(0)` (torch vs torch; 1 vs 2 kernels) | clean (cert.) | clean (cert.) | **L1** | 42u · Σ\|x\| (8u ref, 34u cand) | 1/4 | 2.9e-6 | 0.015 |
| `torch.sum(x,-1)` vs `x[:, :127].sum(-1)` (a bug) | clean (cert.) | clean (cert.) | **not equivalent** | - | 0/4 | 1.66 | - |
| `F.gelu(x,'tanh')` vs Triton gelu on `tanh.approx` (`c_gelu_tanh`, `c_triton_gelu_tanh`) | clean (cert.) | clean (cert.) | **EW** (all 2^32 f32) | \|C − R\| ≤ 5.61e-6; vs gelu_tanh: 4.3e-7 / 5.7e-6 | 311/1024 | 5.2e-6 | - |
| `F.gelu(x)` vs `triton_gelu_erf.py` (f32) | clean (cert.) | clean (cert.) | **EW** (all 2^32 f32) | \|C − R\| ≤ 4.77e-7; vs gelu: 4.9e-7 / 4.5e-7 | 927/1024 | 4.8e-7 | - |
| `F.silu(x)` vs `triton_silu.py` (f32) | clean (cert.) | clean (cert.) | **EW** (all 2^32 f32) | \|C − R\| ≤ 1.91e-6 (65 ulp); vs silu: 3.1 / 9.0 ulp | 599/1024 | 9.5e-7 | - |
| `F.gelu(x)` vs `triton_gelu_erf.py` (bf16) | clean (cert.) | clean (cert.) | **EW** (all 2^16 bf16) | \|C − R\| ≤ 1.53e-5; vs gelu: 0.503 / 0.500 ulp | 1006/1024 | 2.4e-7 | - |
| `x.sum(-1, dtype=torch.float32)` vs `x.float().sum(-1)` (x bf16; 1 vs 2 kernels) | clean (cert.) | clean (cert.) | **L1** | 19u · Σ\|x\| | 4/4 | 0 | 0 |
| `F.layer_norm` vs Triton layernorm (f32) | clean (cert.) | clean (cert.) | blocked | - | 376/512 | 7.2e-7 | - |
| `torch.sort(x,-1).values` vs `torch.topk(x,128,largest=False).values` | clean (cert.) | clean (cert., 1878 states), sim = GPU | blocked | - | 512/512 | 0 | - |
| `torch.mm(a,b)` vs `(b.t() @ a.t()).t()` (bf16, cuBLAS nvjet; `c_mm_bf16_nnt`, `c_mm_bf16_ttn`) | clean (cert.), relies on E-BULK-EXIT | clean (cert., 1098 states), relies on E-BULK-EXIT, sim = GPU | blocked | - | 66/4096 (layouts differ) | - | - |

Two pairs show "inputs differ": `r4_rowreduce_triton` and `r4_softmax_triton` were captured
with other inputs than the torch fixtures. The proofs hold for every input, but those GPU
outputs cannot be compared.

### What blocks, as reported

- **Sum bug.** The report says `NOT EQUIVALENT: with every input 0 except x[127] = 1, out[0] is
  1.0 (reference) vs 0.0 (candidate)`. Blocker: "not the same function at 4 of 4 outputs: x[127]:
  1 vs 0".
- **tanh-GELU** (and erf-GELU, SiLU, bf16 GELU) are proven in elementwise mode. The earlier
  blockers were:
  - torch's tanhf range split (data-dependent guards), now covered by slices (N-SEG);
  - "no dataflow history", now fixed: vector stores materialize every word, and slices are
    complete (guard decisions, the div.rn slow-path variant);
  - "not the same function", answered by the exhaustive sweep.

  `ptxv compare fixtures/c_gelu_tanh fixtures/c_triton_gelu_tanh` reports:

  ```
  => PROVEN elementwise for every f32 input (all 2^32 values): |C - R| <= 5.610e-6, max 891036724 ulp apart, bitwise equal on 97.1325% of inputs; vs gelu_tanh (f64): reference |err| <= 4.332e-7, candidate |err| <= 5.656e-6; GPU: 311/1024 outputs bitwise equal, max |C - R| 5.223e-6
     relies on: reference: N-SEG (8 data-dependent guards inside instruction slices)
  ```

  The huge ulp distance is a cancellation near zero: at x = -4.88 the reference gives -5.8e-7
  and Triton's `tanh.approx` gives -0. docs/equiv-elementwise.md explains it.
- **layer_norm** (agent C3 is adding norms):
  - data-dependent guards in both kernels: torch at sass:0x1e10 `BRA`, and Triton's sqrt
    range scaling at sass:0x05f0 `FMUL R9, R9, 0.25`;
  - outputs not expressed over the inputs (512 of 512 on each side): rsqrt of a variance is
    not a normal form yet.
  - Both kernels are certified clean, with sim = GPU. On the GPU, 376 of 512 outputs are
    bitwise equal, and the largest difference is 7.2e-7.
- **sort vs topk.**
  - Before: topk's `gatherTopK` did not lift. The missing pieces were `ENDCOLLECTIVE` x7,
    `REDUX.OR`, `SR_LEMASK` and `BRA.DIV UR4` (the "bad predicate UR4").
  - What they are: ptxas's collective lowering of `__ballot_sync` with a run-time member
    mask. Its fast path is `REDUX.OR; R2UR P; @!P BRA.DIV`; its slow path is
    `WARPSYNC.COLLECTIVE Rm` ... `ENDCOLLECTIVE`, one member-mask group at a time.
  - They are now lowered (E-COLLECTIVE) and validated bitwise on GB300 by
    `fixtures/s_warpcoll`.
  - Both kernels are now certified clean and equal to the GPU bitwise.
  - Equivalence stays blocked: radix sort's addresses depend on the data (`LDS.U16` at
    sass:0x2fc0), and sorting has no normal form. The GPU outputs are bitwise equal on the
    captured input.
- **bf16 matmul, two cuBLAS kernels.** Both are now certified clean (1074 and 1098 states)
  and equal to the GPU bitwise. Both rely on E-BULK-EXIT.
  - **The old violation was a model gap (S-TMA).**
    - The candidate `nvjet_sm103_tst_8x64_64x16_4x1_v_bz_TTN` stores its 64x64 bf16 tile
      through an 8 x 16 TMA box (16-byte rows) with SWIZZLE_64B.
    - Its `STSM.16.M88.2` writes rows 64 bytes apart.
    - The simulator laid out a swizzled box densely (row r at 16r), so the store read smem
      nothing wrote (0x24410 = row 1) and the output was wrong in 7852 of 8192 bytes.
    - On GB300, a swizzled box narrower than its span has every smem row padded to the span:
      row r at 64r, loads and stores alike. `probes/tma_narrow_box` showed this for all 13
      swizzle x width configurations (the dense layout fails all 6 narrow ones).
    - `exec.rs` `tma_box_off` now uses that layout, and `fixtures/c_mm_bf16_ttn` matches the
      GPU bitwise.
  - **Equivalence is still blocked.**
    - The outputs are bf16, and the numerics express f32 outputs. The bf16 inputs are now
      leaves (N-HALF-LEAF).
    - The candidate's output `.t()` is a transposed view of a [4096] buffer: "pair: buffer
      out is bf16 [64, 64] vs bf16 [4096]".

## Test

`crates/cli/tests/compare.rs` (`cargo test --release`, no GPU) runs three pairs from the
committed fixtures:
- cat: L0, both certified, sim = GPU, 512/512 GPU outputs bitwise equal, no blockers;
- cumsum: L1, bounds from 0 to 15u, GPU fraction in (0, 1];
- gelu (tanh): proven elementwise over all 2^32 f32 inputs (the committed sweep is reused),
  identified as gelu_tanh, nothing blocking, and the GPU difference still reported.

## Changes outside `compare.rs`

All additive:
- `ptxv_sass::lift_report(cubin, kernel) -> Vec<Unlowered>`: the `PTXV_LIFT_REPORT` listing as
  data. The env-var path now uses it.
- `ptxv equiv`:
  - per kernel `certify = true` (certificate first, as `check --certify`);
  - equiv.json gains `captured`, `witness`, `pre_reference` / `pre_candidate` / `pre_pair`,
    `notes_reference` / `notes_candidate`, and `untracked_reference` / `untracked_candidate`;
  - it prints a "captured GPU outputs" line;
  - random trials are skipped when outputs are untracked. Their "bounds" are constants, and
    the trials used to report every output as over its bound and fail the run.
- `tools/capture.py --inputs` with `--file` / `--triton` replaces the module's `inputs()`.
- No change to `crates/sim/src/equiv.rs` or `num.rs`.

Since then (agent F, 2026-11-03):
- `ptxv compare` / `tools/compare.py`:
  - `--function`;
  - segment mode also when outputs have no dataflow history;
  - elementwise results in the headline and in `equivalence.elementwise`;
  - the elementwise sweep cache, seeded from `fixtures/c_equiv_elementwise`.
- `ptxv equiv`: `elementwise`, `function` and `elementwise_gpu` (docs/equiv-elementwise.md);
  bf16 / f16 input leaves; bf16 trials and captured bounds.
- New `crates/sim/src/elementwise.rs` and `halfin.rs`.
- Small additive changes:
  - `seg.rs`: guard decisions, the div.rn slow-path variant, new ops in slices;
  - `num.rs`: bf16 hooks, two accessors.

Since then (agent G, 2026-11-03, docs/compile-vs-eager.md):
- An output updated in place (its buffer starts from captured bytes, `init = "file"`: an
  Inductor `in_out_ptr` kernel) is also an input: its initial elements are leaves, tainted,
  randomized in trials and set in witnesses (`equiv::out_only`). Before, its reads had no
  origin and were not tainted.
- Same-named workspaces of the two cases (`W0`: captured by neither as an input) are no longer
  compared ("pair: buffer W0 is u8 [2048] in the reference but u8 [4096] in the candidate").
- Elementwise product outputs rn(f(x) * y) (docs/equiv-elementwise.md).
