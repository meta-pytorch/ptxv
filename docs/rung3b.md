# Rung 3b: flashinfer trtllm-gen fp8 block-scale MoE (gather4, dynBatch, the whole call)

**Status (2026-11-04):** done for the whole `trtllm_fp8_block_scale_moe` call of flashinfer
0.5.3: routing, gemm1, activation, gemm2 and finalize, captured as one 5-launch case
(`fixtures/r3b_trtllm_moe`, section "The whole MoE" below). Summary:

- gemm2's binary has a race under the PTX model (T3b-2): warp 0 frees TMEM after a barrier of
  only half the epilogue. With that barrier widened (variant `fixed`), all five launches are
  certified by one run (E-WELLSYNC), with contracts only.
- The simulator equals GB300 bitwise on the MoE output, 1000/1000 runs, for both variants.
  The routing and gemm1 intermediates also equal the GPU's snapshots.
- The numerics claims hold at every stage, and end to end against the exact f64 MoE. The
  routing dependency claim holds.
- B70-B72 are caught (a routing atomic, a finalize weight, a finalize index). On GB300 each
  equals the simulator's mutant output.
- gemm1 at K = 2048 (8 stages wrapping) is certified and equals GB300 bitwise; B73, a parity
  bug that only shows on the wrap, is caught.

The gemm1-alone case below came first. Its kernel is a prebuilt, source-free cubin:

- the whole grid is certified (E-WELLSYNC), and its findings are contracts only
  (E-BULK-EXIT, E-TC-EXIT);
- the simulator equals GB300 bitwise, on the bytes captured inside the MoE call and on 1000
  standalone runs;
- the numerics and dependency claims hold, and B25, B27, B28 and B30 are caught;
- with partial tiles the kernel reads shared memory nothing wrote (a violation), but the
  result does not depend on it.

Tests are in `crates/sim/tests/r3b.rs` and `crates/sim/tests/r3b_moe.rs`.

## The case: `fixtures/r3b_trtllm_bmm`

`tools/capture_examples/trtllm_fp8_moe.py` runs the DeepSeek-style fp8 MoE:

- data: e4m3;
- scales: f32, one per 128 x 128 weight block and one per 128 hidden channels of a token;
- routing: Renormalize, top-2 of E = 4 experts;
- shape: T = 16 tokens, hidden H = 256, intermediate 2I = 256.

The logits are chosen so that every expert gets exactly 8 tokens, which is one full tile.
The call makes five launches: routing, gemm1, activation, gemm2 and finalize.
`tools/capture.py --kernels '^bmm_E4m3'` keeps gemm1:

`bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_16dp256b_rM_TN_transOut_noShflA_dsFp8_schedS_bN_tma_tmaOpt_clmp_lbW2_dynBatch_sm100f`

| name part | meaning |
|---|---|
| `t128x8x128` | a 128-channel x 8-token tile, K = 128 per block; 8 smem stages |
| `bN` + `dynBatch` | tokens are the N side, routed |
| `tma` | the token rows are gathered with `UTMALDG.2D.GATHER4` |
| `dsFp8` | per-128-K block scaling in software |
| e4m3 output | per-token amax scales |

The launch is grid 2 x 7, 416 threads (13 warps) and 147,952 bytes of dynamic smem. The
warp roles were read off the SASS:

- warps 0-7: epilogue, two M = 64 halves (TMEM lanes 0-15 and 16-31);
- warp 8: MMA (`UTCQMMA`, kind::f8f6f4, M = 64, N = 8);
- warp 9: weights (`UTMALDG.3D`);
- warps 10-11: token gather, 4 rows each;
- warp 12: scales (`LDGSTS` + `cp.async.mbarrier.arrive`).

Only 4 of the 7 y-tiles hold tokens. The other three exit at entry (`numNonExitingCtas` = 4).
The kernel's buffers come from the routing kernel's outputs:

| buffer | content |
|---|---|
| `W1` | permuted row -> token |
| `W2` | tile -> expert |
| `W3` | tile -> token limit |
| `W0` | padded rows |
| `W6` | e4m3 output, rows of 2I bytes |
| `W7` | output scales, `[2I/128][rows]` |

The cubin is flashinfer-cubin's file, slimmed by `tools/cubin_slim.py` (58,912 -> 56,760
bytes: debug frames emptied). Regenerating it is the one command in case.toml.

`fixtures/r3b_trtllm_bmm_partial` is the same kernel with partial tiles: `PTXV_MOE_PARTIAL=1`,
8 tokens routed at random, so each tile holds 1-4 tokens.

**Capturing one kernel out of a sequence (new).** gemm1 reads buffers that the routing kernel
writes during the same call. Before this change, `capture.py` started such buffers as zeros.
Now `probes/launchcap` takes `LAUNCHCAP_SNAP=regex`. For a matching launch it synchronizes
and dumps every device allocation a parameter word points into (`cuMemGetAddressRange`),
once before the kernel and once after. With `--kernels`, `capture.py` then:

- starts the buffers from the first kept launch's pre-snapshot;
- takes as outputs what the kept launches changed, as of the last one's post-snapshot;
- adds allocations outside the caching allocator as buffers.

docs/capture.md has the details.

## Lifting and how each new piece was validated

| piece | model | validation |
|---|---|---|
| `UTMALDG.2D.GATHER4` (`cp.async.bulk.tensor.2d...tile::gather4`; PTX front too) | 5 coordinates (column, 4 rows); the rows land as a `box[0] x 4` box would (S-TMA layout, swizzle phase from the address); out-of-bounds rows and columns zero-filled; complete_tx = 4 rows; the map must be 2-D with box height 1. A column whose byte offset is not a multiple of 16 is a `fault` (S-TMA-GATHER4) | **fixtures/s_gather4**: 6 gathers through 128B / 64B / 32B / none / narrow-128B maps. Rows include 64, 1000, -1 and repeats; columns run past both edges; the starts are at 128-byte offsets that are not 8-row aligned. PTX and SASS = GB300 1000/1000, and the recorded bytes are in cargo test. The 16-byte column rule was found by bisection: columns 3 and 100 (u16), and 4 (u16 with the 128B map), are illegal instructions; variant `misaligned` |
| `SYNCS.ARRIVE.TRANS64.RED.A0T1` + `ARRIVES.LDGSTSBAR.64.TRANSCNT` / `.ARVCNT` | what ptxas emits for `cp.async.mbarrier.arrive` / `.noinc` (Rosetta: compiled both and read the SASS). A0T1 = expect one transaction unit, no arrival. LDGSTSBAR = an async op that lands the thread's cp.async writes, then completes that unit (TRANSCNT) or arrives (ARVCNT) (E-CPASYNC-MBAR) | **fixtures/s_cpasync_mbar** (both forms) = GB300 1000/1000, and the recorded bytes |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT PT, ...` | result discarded: a no-op (E-SPIN) | argued: it neither blocks nor writes. Dropping its acquire can only add races |
| peeks whose result is kept as a token (`SEL R4, RZ, 0x1, !P1` ... `ISETP.NE.AND P1, PT, R4, 0x1` ... wait) | `spin::is_peek` tracks predicates and registers known from failed peeks through SEL and integer EQ/NE compares | the certified run equals GB300 bitwise. Without this, the try_waits were `unmodeled` |
| TMA element addresses | wrap modulo 2^64. The output map has dims `[2I, 8, 2^31, 2^31]` and strides `[2I, 2^35 - 256, 2I]`, with coordinates `(ch, (-n) & 7, 2^30, row0 + 2^30 - ((-n) & 7))`. The `2^30 x stride` terms cancel mod 2^64, and the 8-row dim clips rows past the tile's n valid tokens | output bitwise = GB300 |
| tcgen05.commit nobody waited for before exit | `contract` (E-TC-EXIT), not a violation | **probes/tc_commit_exit**: 4864 CTAs x 200 runs, one CTA per SM. The next CTA's mbarrier at the committed offset never saw an early phase, for trtllm-gen's shape or with 64 MMAs still in flight. The positive control was 200/200 |
| tensor maps (S-TMAP) | nothing new to decode: the gather map is a tiled 2-D map (u8, box 128 x 1, 128B), and the output map is an ordinary 4-D one | decoded by `tmap::decode`; `hw` encodes u16 / u32 / i32 maps now (s_gather4) |

## Verdicts

```
ptxv check fixtures/r3b_trtllm_bmm --certify            # whole grid, 4714 steps, ~2 s
PASS good   sass [contract]
    note: well-synchronized: one run covers all schedules (E-WELLSYNC, 4714 steps)
    CONTRACT: TMA store at sass:0x2f40 / 0x3f50 ... (E-BULK-EXIT)
    CONTRACT: tcgen05 pipe at sass:0x4cb0 / 0x4cd0 `UTCBAR` ... (E-TC-EXIT)
    expect (output 0): W6: bitwise equal to the captured GPU output (262144 bytes)
    expect (output 0): W7: bitwise equal to the captured GPU output (448 bytes)
```

- `--slices --certify` certifies every one of the 14 CTAs.
- `PTXV_GPUS=3 ptxv hw fixtures/r3b_trtllm_bmm --variant good --runs 1000`: bitwise equal to
  the simulator 1000/1000.

The two contracts come from the end of the kernel:

- The epilogue's TMA stores are never waited for (E-BULK-EXIT, as cuBLAS).
- The MMA warp's last two `UTCBAR` arrive on the A/B stage "empty" mbarriers. The loader
  warps have finished and never wait on them. Under the PTX model the arrival may land after
  the CTA exits. GB300 drains the tcgen05 pipe first.

## Numerics and the dependency claim

**Claim (`r3b_bmm_good`).** For every routed row p and channel ch, take:

- the exact value V = sum over k-blocks of w1s[g][ch/128][kb] * hss[kb][tok] * sum_k
  w1[g][ch][k] * hs[tok][k], where g = expert(p / 8) and tok = perm[p];
- the kernel's dequantized output, e4m3(W6[p][ch]) * W7[ch/128][p].

The two differ by at most:

- half an e4m3 ulp: 2^-4 |V| relative, plus 2^-10 x the scale absolute (subnormals);
- plus 2^-18 x sum |terms| for the f32 and tensor-core accumulation.

This holds for all 32 x 256 outputs. The scale is the block's amax / 448, from MUFU.RCP;
`FMUL.FTZ ... 448` is in the SASS.

**Dependency (`r3b_bmm_dependency_claim`).** Each case perturbs one input and runs again.
Exactly the outputs the claim names change, plus their scales, and nothing past the routed
rows changes:

| perturbed input | outputs that change |
|---|---|
| token 5's activation row | exactly its 2 routed rows (all channels) and their 4 scales |
| token 9's K-block-1 scale | exactly its rows |
| weight row (expert 2, ch 37) | channel 37 in all 8 of expert 2's rows; other changes only in ch 37's 128-channel block (the amax), in expert 2's rows |
| weight block scale (expert 1, block 1, K-block 0) | only channels 128-255 of expert 1's rows |

## Mutants (SASS patches in case.toml)

| id | patch | ptxv | GB300 |
|---|---|---|---|
| B25 | the token-tile mbarrier's `expect_tx` 1024 -> 512 (`IMAD.MOV.U32 R0, RZ, RZ, 0x400` at 0x15d0); the two gather4 loads still deliver 1024 | race + violation. The phase completes after one gather, so the MMA reads stage bytes the other gather has not written, and that gather may land after the CTA exited (async-after-exit). The certificate refuses (C-PHASE: a gather can land in phase 0 or 1); `--slices` exploration (160k states) finds no deadlock | **hangs** (watchdog), every run |
| B27 | first epilogue half: token offset `ctaid.y << 2` instead of `<< 3` (0x2870) | race: overlapping scale `STG`s inside a CTA, and clusters 4 / 6 and 5 / 7 write the same output bytes (`--slices` cross-cluster check) | wrong 100/100, 6 distinct outputs |
| B28 | k-block 1's first `UTCQMMA` accumulates (`!UPT` -> `UPT`, 0x4ad0) | violation: accumulates into TMEM never written | wrong 99/100 (run 0 correct: TMEM starts zeroed, as B42) |
| B30 | the scale loader does not advance the weight-scale pointer after k-block 0 (`IADD3 R4, P0, PT, R4, 0x4` -> `0x0`, 0x1c40), so k-block 1 uses k-block 0's weight scale | execution unchanged (contract) and certified. The numerics claim fails for > 1000 of 8192 outputs (`r3b_bmm_mutants`) | wrong 100/100; GB300 = the simulator's B30 output bitwise 1000/1000 (`PTXV_HW_EXPECT=variant`) |

B25's hang is not a deadlock in the model. The arming warp issues its own gather after the
arm, so the tx count cannot end negative in phase 0. The hang is consistent with the
async-after-exit violation: a TMA landing on an exited CTA keeps the kernel from retiring.
That is not probed. Over-delivery alone does not hang GB300: one 512-byte gather against
`expect_tx(256)`, or two against `expect_tx(512)`, both complete (s_gather4's first gather,
modified). A related model gap: for the single over-delivering TMA, the model applies the
whole complete_tx at once (tx = -256) and reports a deadlock, while GB300 completes the
phase. Evidently the hardware drops the count in parts.

Not applicable:

- **B26** (empty group): the schedule is static, one tile per CTA, so no barrier phase
  carries from one tile to the next. Experts without tokens get no CTA: the CTA exits at
  entry, `numNonExitingCtas`.
- **B29** (tensormap update without fence): the maps are kernel parameters, not built on the
  device.
- **B31** (no promotion, or promotion too rarely): the kernel promotes once per 128-K block
  because each block has its own scale. A "longer interval" mutant must apply one block's
  scale to another block's products, which is B30's failure. Precision is not the issue on
  Blackwell either:
  - `UTCQMMA` accumulates in f32 TMEM, aligned to the largest exponent and truncated 25 bits
    below it (N-MMA-F8);
  - the f32 FFMA2 promotion is no more precise;
  - Hopper's ~14-bit fp8 accumulator, the reason for DeepSeek's promotion, has no
    counterpart here.

  BUGS.md says so.

## Finding: partial tiles read unwritten shared memory (`fixtures/r3b_trtllm_bmm_partial`)

With 1-4 tokens per tile:

- The scale loader issues `LDGSTS` only for valid tokens (`ISETP.GE.AND P0, PT, R7, R2`).
- The epilogue still loads all 8 token scales (`LDS.64 [R4+0x23c00]`), multiplies, and
  publishes the per-token amax with `ATOMS.MAX` (sass:0x29c0). ptxv reports a violation
  there (S-INIT: undefined bits reaching memory).
- The garbage stays in the padded columns. The TMA store clips those rows, and the scale
  `STG`s are guarded.
- GB300 and the simulator agree bitwise (100/100, also with shared memory scrubbed).

The verdict is a true statement about the binary: it reads uninitialized smem. It is benign
for this launch's outputs. Proving it benign would need undefined-bit tracking through shared
memory and TMA stores.

## The whole MoE: `fixtures/r3b_trtllm_moe`

### Capture

```
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py --file tools/capture_examples/trtllm_fp8_moe.py \
    --out fixtures/r3b_trtllm_moe
```

The inputs are those of the gemm1 case (T = 16, H = 256, I = 128, top-2 of E = 4). No
`--kernels`, so every launch is kept: the buffers start from the named tensors, and buffers
allocated during the call start as zeros (flashinfer issues no memset before routing). The
cubins were slimmed with `tools/cubin_slim.py`, keeping the launched kernels (k0 8.8 MB ->
0.78 MB; k1 is byte-identical to the gemm1 case's cubin).

| # | kernel | grid x block | PDL | reads | writes |
|---|---|---|---|---|---|
| 0 | `routingIndicesClusterKernel` (Renormalize) | 8 x 1024, cluster 8 | yes | logits | W0 rows, W1 (t, k) -> row, W2 row -> token, W5 tile -> expert, W6 tile -> limit, W7 non-exiting CTAs, W8 bf16 weights |
| 1 | gemm1 `bmm_E4m3_..._dynBatch` | 2 x 7 x 416 | no | hs, hss, w1, w1s, W0 W2 W5 W6 W7 | W9 e4m3 [rows][2I], W10 scales [2][rows] |
| 2 | `activationDeepSeekKernel` | (1, 16, 2) x 128 | yes | W9, W10, W1, W0 | W12 e4m3 [rows][I], W11 scale per row |
| 3 | gemm2 `bmm_Bfloat16_E4m3E4m3_..._dynBatch` | 2 x 7 x 384 | no | w2, w2s, W12, W11, W0 W5 W6 W7 | W13 bf16 [rows][H] |
| 4 | `finalizeKernel` | (1, 16) x 256 | yes | W13, W1, W8 | out bf16 [T][H] |

PDL is on despite `enable_pdl=False`: launchcap records the attribute on launches 0, 2 and 4.

- Routing, gemm1 and gemm2 trigger their dependents (`PREEXIT`, sass 0x9d50 / 0x1820 /
  0x38a0).
- Activation and finalize run `griddepcontrol.wait` (`ACQBULK`) before touching data, so
  every dependent kernel reads its producer's results after the producer completed (E-PDL).
- gemm1 and gemm2 are launched without PDL, so they wait for completion anyway.

The routing kernel's cluster shape is compiled in (`__cluster_dims__`): the launch names no
cluster, and the cubin's `.nv.info` holds `EIATTR_CTA_PER_CLUSTER` (0x3d) = 8 x 1 x 1. Before
this change `capture.py` wrote no cluster. It now reads the attribute
(`cubin_cluster`).

The kernel semantics were read off the SASS and checked against the simulator's
intermediates:

- **routing:** top-2 by a `CREDUX.MAX` over packed (bf16 score, ~index) keys, so ties go to
  the lower expert. The softmax runs over the top 2 (`MUFU.EX2`, `MUFU.RCP`) and is stored
  as bf16. Expert histograms come from `ATOMS.ADD` and `ATOMS.POPC.INC`; each (token, k)
  gets its slot within the expert from the add's return value. Peer CTAs' top-k are read
  through the generic shared window (`LD.E.U16` on `SR_SWINHI`-based addresses).
- **activation:** `act = silu(x[I + i]) * x[i]` (the gate is the second half of gemm1's
  channels), quantized to e4m3 with one scale per row, amax / 448.
- **gemm2:** `y = sum_i w2[e][h][i] act[i] * s_act * w2s[e][h / 128]`, rounded to bf16.
- **finalize:** `out[t][h] = bf16(sum_k bf16(w_k) * y[row(t, k)][h])` (`FFMA.FTZ` from 0).

### Lifting and validation (`fixtures/s_routing_forms`)

| form | model | validation |
|---|---|---|
| `F2FP.SATFINITE.BF16.F32.PACK_AB` (gemm2's epilogue) | `CvtPack` to bf16, RN, satfinite: overflow and +-Inf -> +-max (0x7f7f), NaN -> 0x7fff (lowp.rs, as the other satfinite packs) | s_routing_forms word 0: f32 pairs with overflow, +-Inf, NaNs, ties (low half 0x8000), random bits |
| `HSETP2.BF16_V2.<cmp>.AND Pd, PT, Ra.Hx_Hx, Rb.Hy_Hy / -INF , -INF , Pc` | new `Hsetp2Bf16`: one bf16 lane per operand, widened exactly to f32, `f32_cmp`. ptxas emits NE, NEU, GT, GTU, GE and GEU and negates the predicate for the rest | word 1: all 14 `setp.<cmp>.bf16` on random and edge pairs (+-0, +-Inf, NaNs, equal, one ulp apart), plus the immediate `>= -inf` form |
| `ATOMS.POPC.INC.32 RZ, [Ra]` | what ptxas emits for `atom/red.shared.add.u32 [a], 1` with the result unused: each active lane adds 1 | word 6: divergent lanes (y & 3 != 0) on 16 colliding counters |
| `S2R SR_SWINHI` and generic shared addresses | `SReg::SharedWindowHi`. A generic address whose high word is the window's is the shared::cluster address in its low word (`cvta.shared` = `{SWINHI, rank << 24 \| off}`; generic `mapa` = PRMT of the rank byte). `resolve` maps it, and `Cvta` / `Mapa` handle the window (S-SWIN). GB300's window high word is 0xfffa; the model uses 0x7c00 | words 2-5, 7: `LD.E` / `ST.E` / `LD.E.U16` through generic addresses into this CTA's and the peer CTA's shared memory (2-CTA cluster); the mapa'd address's low word, and its high word xor the cvta's |

`s_routing_forms` is a SASS case (ptxas output of `kernel.ptx`): the PTX front does not take
the bf16 compares. Simulator = GB300 bitwise, 1000/1000; the bytes are in expected/ and in
cargo test. The forms came from the Rosetta method: one PTX line per form, compiled with
ptxas, then the SASS read.

### Verdicts

```
ptxv check fixtures/r3b_trtllm_moe --variant fixed --certify     # 15018 steps, ~26 s
PASS fixed  sass [contract]
    note: well-synchronized: one run covers all schedules (E-WELLSYNC, 15018 steps)
    CONTRACT: TMA stores (gemm1 / gemm2 epilogues) not waited for before exit (E-BULK-EXIT)
    CONTRACT: tcgen05.commit arrivals nobody waits for (gemm1 / gemm2 MMA warps) (E-TC-EXIT)
    expect (output 0): out: bitwise equal to the captured GPU output (8192 bytes)
```

**Hardware:**

- `PTXV_GPUS=3 ptxv hw fixtures/r3b_trtllm_moe --variant good --runs 1000`: bitwise equal to
  the simulator, 1000/1000. `fixed` likewise.
- The routing and gemm1 intermediates of the simulated call equal the GPU snapshots in
  `fixtures/r3b_trtllm_bmm/data` (W1, W2, W3, W5) and `expected` (gemm1's output and scales),
  in their meaningful regions (test `r3b_moe_good_races_in_gemm2_and_matches_the_gpu`).

Two changes to the certificate were needed:

- **C-ORDER, the lanes of one SASS atomic step.** These are ordered by lane order
  (E-LOCKSTEP), and C-WARP keeps such lanes together in every schedule. Routing assigns token
  slots with `ATOMS.ADD`, whose return values the lanes of one warp read. Before this change
  the certificate refused the routing kernel.
- **Peeks** (rung 3b's K = 2048 case, below).

**T3b-2: gemm2 frees TMEM after half its epilogue.**

- gemm2 has two epilogue halves: warps 0-3 read TMEM lanes 0-15, and warps 4-7 read lanes
  16-31 (`tmem + 0x100000`).
- The MMA warp issues each half's 4 MMAs and commits them to different mbarriers: 0x241b0,
  which warps 0-3 wait on, and 0x241d0, which warps 4-7 wait on.
- After its TMA store, warp 0 frees the TMEM columns (`UTCATOMSWS.AND`, sass:0x1cc0). This
  happens after `BAR.SYNC 0x9, 0x80`, a barrier of warps 0-3 only.
- So neither the second half's MMAs nor warps 4-7's `LDTM` are ordered before the free. The
  run reports write-write races on TMEM (MMA at 0x3040 vs the free).

gemm1 has no such race: its two halves meet at `BAR.SYNC 0x1, 0x100`, because the per-token
amax needs both. Variant `fixed` makes the halves meet before the free:
warps 0-3's `BAR.SYNC 0x9, 0x80` and warps 4-7's last `BAR.SYNC 0x7, 0x80` become
`BAR.SYNC 0x9, 0x100`.

The race is benign on GB300:

- the 147,952-byte shared memory allows one CTA per SM, so nothing can allocate the freed
  columns while warps 4-7 still run;
- freed TMEM keeps its data, as in B07.

### Numerics claims (`r3b_moe_fixed_certified_and_claims_hold`)

Each stage is checked against the exact function (f64) of the dequantized values it read,
for all outputs. The intermediates come from the certified run.

| stage | exact reference | tolerance |
|---|---|---|
| routing | top-2 of the bf16 logits (ties: lower expert); softmax over the two | slot consistency: W2[W1[t, k]] = t and W5[W1[t, k] / 8] = expert. Weight: half a bf16 ulp (2^-8 relative) + 2^-18 |
| gemm1 | as `r3b_bmm_good` | e4m3 half ulp (2^-4 relative, 2^-10 x scale absolute) + 2^-18 x sum of the terms' magnitudes |
| activation | silu(gate) x up of gemm1's dequantized output | e4m3 half ulp of the row scale + 2^-17 relative (MUFU exp, rcp) |
| gemm2 | sum over i of w2 x act x scales | bf16 half ulp (2^-8 relative) + 2^-18 x sum of the terms' magnitudes |
| finalize | sum over k of w_k(exact) x y[row(t, k)][h] | 2^-8 relative + (2^-8 + 2^-18) x sum of the terms' magnitudes (bf16 weights, f32 FFMA) |
| end to end | the exact MoE of the dequantized inputs: no intermediate rounding, exact softmax weights | each stage's rounding propagated in the worst case (\|silu'\| <= 1.1) |

The stage claims are tight: B71 and B72 break the finalize claim for more than half the
outputs.

The end-to-end claim holds for all 4096 outputs, but it is loose:

- the bound is 2.3 x |exact| at the median, because worst-case propagation through the
  128-term sums ignores cancellation;
- the observed error is at most 0.085 of the bound;
- the median relative error against the exact MoE is 3.8%, the price of the two e4m3
  intermediates.

### The routing dependency claim (`r3b_moe_token_depends_exactly_on_its_experts`)

Token t's output row depends exactly on its top-k experts:

- Replacing expert 3's w1 and w2 changes exactly the rows of the 8 tokens routed to expert 3.
- Replacing token 6's activations changes exactly row 6.

### Atomic lane order (`r3b_moe_output_does_not_depend_on_atomic_lane_order`)

PTX leaves open the order in which the lanes of one atomic on one address are performed. The
model takes lane order, which is GB300's here: the slots equal the GPU snapshot.

With `Machine::atom_lanes_rev` (`PTXV_ATOM_LANES=rev`) the lanes go in reverse:

- the permutation W1 changes;
- the MoE output is bitwise unchanged, and every claim still holds.

### Mutants

| id | patch | ptxv | GB300 |
|---|---|---|---|
| B70 | routing: the expert-count `ATOMS.ADD` at 0x2af0 adds 0 (`IMAD.MOV.U32 R10, RZ, RZ, 0x0` at 0x2ac0), so all of an expert's (token, k) get slot 0 | race: several threads store the slot-0 entries of the permutation (sass:0xa070); the routing claim fails | wrong 1000/1000; = the simulator's B70 output bitwise 1000/1000 (`PTXV_HW_EXPECT=variant`) |
| B71 | finalize: the second expert's weight is loaded from the first's slot (`LDG.E.U16 [R6.64+0x2]` -> `[R6.64]`, 0x08c0) | executes like `fixed` (certified, contracts); the finalize claim fails for > 2048 of 4096 outputs, every earlier stage holds | wrong 1000/1000; = the simulator's output bitwise 1000/1000 |
| B72 | finalize: the second expert's row is read through the first's permutation index (`LDG.E [R8.64+0x4]` -> `[R8.64]`, 0x0840) | likewise | likewise |

B70-B72 are applied on top of `fixed`, so that the only findings are the mutants'.

### Simulator speed

The case has 27,584 threads. Three O(threads^2) paths made one run take more than 15 minutes:

- every `griddepcontrol.wait` rejoined the previous grid's 8192 clocks;
- every cluster-barrier arrival rescanned all threads;
- every waiter rescanned all threads again.

The fixes:

- `grid_done_vc_cached`: a completed grid's clock is computed once, so waiters share its
  base. `grid_complete` short-cuts on it too.
- `cluster_threads`: only the cluster's own threads are scanned.
- The forgetting of old generations runs once per completion.

One certified run now takes about 26 s.

## Larger K: `fixtures/r3b_trtllm_bmm_k2048`

`PTXV_MOE_H=2048` (in `trtllm_fp8_moe.py`) with `--kernels '^bmm_E4m3'`. flashinfer picks the
same gemm1 cubin. There are 16 K-blocks through the 8 stages, so every stage's full and empty
mbarriers complete phases 0 and 1, and the producers wait on empty barriers for real.

```
ptxv check fixtures/r3b_trtllm_bmm_k2048 --certify      # 16362 steps, ~18 s
PASS good   sass [contract]    (E-BULK-EXIT, E-TC-EXIT)
    expect: W6, W7 bitwise equal to the captured GPU output
```

The numerics claim of `r3b_bmm_good` holds for all outputs (`r3b_bmm_k2048_stages_wrap`).

The certificate first refused the case:

- The loaders use CUTLASS's try-acquire. The weight loader peeks the next stage's empty
  barrier, keeps the result as a token (`SEL R4, RZ, 0x1, !P1`), and branches on it in the
  next iteration (`ISETP.NE.AND P1, PT, R4, 0x1` -> `@P1 BRA` to the wait stub).
- The token loaders and the scale loader peek with a guard (`@P2 TRYWAIT P0` after the first
  wrap) and branch right away.
- The peek-use analysis now follows a GPR token through `SEL`.
- At the meet point, the register the two sides leave different (the token, or the flipped
  predicate) is now proven dead by walking on (`walk_dead`). Static liveness alone loops
  back to the read through the guarded peek or the last-iteration exit.
- The walk runs ALU ops and branches on a copy, steps over effects without a register
  result, and treats memory results as unknown, forking on unknown branches. ASSUMPTIONS.md
  E-WELLSYNC has the rule.

**B73.** The weight loader does not flip its empty-barrier parity on the wrap
(`LOP3.LUT R6, R6, UR13, RZ, 0x3c` -> `0xf0` at 0x14c0). From K-block 8 on it waits for a
phase that already completed, then re-arms a full barrier whose phase is still pending: a
fault ("more arrivals than the expected count") and a deadlock. With K = 256 the patch is
invisible, because no stage is reused. **GB300 faults too:** the first run fails with
`cuCtxSynchronize: unspecified launch failure`.

**Hardware, good:** `PTXV_HW_EXPECT=variant ptxv hw ... --variant good --runs 1000` gives
simulator = GB300 bitwise, 1000/1000. `num` is too slow at this K, so the run is compared
with a deterministic simulator run instead.

## Open

- Why B25 hangs GB300. A probe of a TMA landing in an exited CTA would settle it.
- Over-delivered TMA: the model's all-at-once complete_tx deadlocks where GB300 completes the
  phase (above). The class differs, but both report a bug.
- The end-to-end numerics bound is worst-case and loose (2.3 x |exact| at the median). A
  tight one needs error terms carried with signs through the sums, as `equiv`'s L3 does,
  across five launches.
- Exhaustive exploration of the good MoE (not needed for the verdict: its race is a witness)
  runs out of memory at 96 GB. On a bigger machine or a Slurm cluster, if wanted:
  `ptxv check fixtures/r3b_trtllm_moe --variant good --max-states 50000000`.
- Not covered: the `cga2x1x1` variants; the `ldgsts` routing variant; more than 16 tokens
  (several tiles per expert); the PTX front for `setp.bf16` and `mov.b32 {h, h}`.
