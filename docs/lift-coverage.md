# SASS lifter coverage: rungs 4 and 5 (softmax, norms, reductions, flash attention)

**Status (2026-11-03, round 2):** all 10 rung-4 kernels (Triton's and torch's own) lift
completely, run to the end and match GB300 bit for bit. That now includes both flash-attention
kernels:
- torch's sm80 `flash_fwd_kernel` (HMMA, ldmatrix, a 3-stage cp.async pipeline, register
  spills): `clean`, = GB300.
- Triton's tcgen05 flash attention (S read back from TMEM in lane halves, P stored to TMEM
  and used as the MMA's A operand, V as an MN-major B, masked cp.async prefetch): = GB300; its
  verdict is `violation` only for Triton's known descriptor contract issue (bits 46-48 left
  0, as in rung 2).

D's torch-op smoke set: all 33 ops lift and match GB300 bitwise (table below; the cuDNN
SDPA, index_add, sort, the nvjet matmuls, where, to_bf16 and the mixed capture are new).

Every new instruction form is validated on GB300:
- `fixtures/s_alu2` checks every pair of 128 operands (64 special bit patterns, 64 random)
  through 44 output words, 720,896 words in all, bitwise (N-FALU).
- `probes/fchk` checks FCHK's predicate (N-FCHK).

`cargo test` checks both against the hash of GB300's output.

## Kernels

| fixture | source | grid x block | good | mutant |
|---|---|---|---|---|
| `r4_rowreduce_triton` | Triton row sum + max, 4x128 f32 | 4 x 64 | clean, **= GB300** | `nosb`: violation |
| `r4_sum_torch` | `x.sum(-1)`, torch `reduce_kernel` | 1 x (32,4) | clean, **= GB300** | `nosb`: violation |
| `r4_softmax_triton` | Triton row softmax, 4x128 f32 | 4 x 64 | clean, **= GB300** (MUFU.EX2) | `nosb`: violation |
| `r4_softmax_torch` | `torch.softmax(x, -1)`, `softmax_warp_forward` | 1 x (32,4) | clean, **= GB300** (MUFU.EX2) | `nosb`: violation |
| `r4_layernorm_triton` | Triton layer norm | | clean, **= GB300** (MUFU.SQRT) | `nosb`: violation |
| `r4_layernorm_torch` | `F.layer_norm`, bf16, `vectorized_layer_norm_kernel` | | clean, **= GB300** (MUFU.RSQ) | `nosb`: violation |
| `r4_rmsnorm_triton` | Triton rms norm | | clean, **= GB300** (MUFU.RSQ) | `nosb`: violation |
| `r4_rmsnorm_torch` | `F.rms_norm`, bf16, `vectorized_layer_norm_kernel<.., rms=true>` | | clean, **= GB300** (MUFU.RSQ) | `nosb`: violation |
| `r4_attn_triton` | Triton flash-attention forward (tcgen05), N_CTX 128, HEAD 64 | 2 x 128 | violation (descriptor bits only), **= GB300** | `nosb` (LDTM): violation from the scoreboard check |
| `r4_sdpa_flash_torch` | `F.scaled_dot_product_attention`, flash backend (sm80 `flash_fwd_kernel`) | 1 x 128 | clean, **= GB300** | `nosb` (LDSM): violation |

How each fixture was made:
- **Triton fixtures:** `tools/r4_triton_capture.py`. It writes kernel.ptx and Triton's own
  kernel.cubin, plus inputs in data/ and the GPU's outputs in expected/.
- **Torch fixtures:** D's `tools/capture.py`, driven by `tools/r4_torch_capture.sh`. Each
  library cubin is then shrunk to the launched kernel by `tools/cubin_slim.py`. That script
  empties the other functions' sections and re-lays out the ELF; the slimmed cubins give
  identical GPU outputs.

The `nosb` mutant is a cheap one: one variable-latency producer (a `SHFL` of the reduction,
or the first input `LDG`) loses its write scoreboard.

Tests:
- `tests/fixtures.rs` `r4_kernel_classes` checks every variant's classes.
- `captured_outputs_match_the_gpu` (D's test, which now honours the variant's expectation)
  checks the GPU bytes of the clean ones.
- An `unlifted` fixture must still fail to load. If it starts lifting, the test fails so that
  its expectation gets updated.

## Gap inventory: before

`PTXV_LIFT_REPORT=1 ptxv check <case> --front sass --max-states 1`, base commit. Counts are
static instruction counts.

| kernel | not lowered |
|---|---|
| rowreduce_triton | FMNMX (7), FSEL (2) |
| sum_torch | SR_LTMASK, POPC (2), VIMNMX.U32 (2) |
| softmax_triton | FMNMX (7), FSETP.GEU/GT.AND (4), `-R` f32 operands, MUFU.EX2 (2) |
| softmax_torch | FCHK (8), FFMA.RM/RP/RZ (11), FFMA.SAT (8), FSEL (24), FSETP.{GEU,GT,NEU}.AND, FSETP.{GTU,NEU}.FTZ.AND (30), FADD.FTZ/SAT, `-R`, MUFU.EX2 (8), MUFU.RSQ |
| layernorm_triton | FSEL (3), FSETP (4), MUFU.SQRT |
| layernorm_torch | sub-word LDG (80), `-R` (110), FADD.FTZ (18), FCHK, FFMA.RM/RP/RZ, FSETP (16), VIMNMX[.U32] (5), MUFU.RSQ (2): 23 kinds |
| rmsnorm_triton | FMUL2 (3), FSETP (2), MUFU.RSQ |
| rmsnorm_torch | sub-word LDG (60), `-R` (20), FADD.FTZ, FCHK, FFMA.RM/RP/RZ, FSETP (7), VIMNMX[.U32] (5), MUFU.RSQ (2): 14 kinds |
| attn_triton | FMNMX3 (32), FMNMX, FSETP, `-R`, MUFU.EX2 (66), LDTM/STTM .16dp32bit shapes (20), UTCHMMA with a TMEM A operand (8), LDGSTS (36), LDGDEPBAR (9), LDSM (4): 19 kinds |
| sdpa_flash_torch | FSETP (277), FMNMX3 (124), FMNMX (20), FSEL (12), `-R` (~280), FADD.FTZ, FFMA.RM/RP, MUFU.EX2 (260), MUFU.LG2, HMMA.16816.F32.BF16 (512), LDSM (144), LDGSTS (48), LDGDEPBAR (5), STL/LDL (64), LEA `-R`: 30+ kinds |

## Gap inventory: after

| kernel | not lowered | run |
|---|---|---|
| rowreduce_triton | none | clean, = GB300 |
| sum_torch | none | clean, = GB300 |
| softmax_triton, softmax_torch | none | clean, = GB300 |
| layernorm_triton | none | clean, = GB300 |
| layernorm_torch, rmsnorm_triton, rmsnorm_torch | none | clean, = GB300 |
| attn_triton | LDTM (10) / STTM (10) `.16dp32bit_t0_t15` shapes; UTCHMMA with A in TMEM (`expected gdesc[..], got tmem[..]`, 8); LDGSTS.E.BYPASS.128 (36); LDGDEPBAR (9); LDSM.16.M88.4 (4): 12 kinds | |
| sdpa_flash_torch | HMMA.16816.F32.BF16 (512); LDSM.16.M88.4 / .MT88.4 (144); LDGSTS.E.BYPASS.LTC128B.128 (48); LDGDEPBAR (5); STL (35) / LDL[.LU] (29) local memory; `LEA` with a negated first operand (1): 9 kinds | |

## Round 2 (rung 5): instructions added and how each was validated on GB300

| form | validation |
|---|---|
| `HMMA.16816.F32.BF16` (warp-collective; PTX ISA fragments) | **probes/hmma**: `ptxv calib-mma` over 820 candidate models, 6 input sets, 98,304 outputs: exactly `g16-f25-rz-accin-tzero-einputs` matches all (the tcgen05 model; N-HMMA); expected/ in cargo test; torch flash attention = GB300 |
| `LDSM.16.M88.{1,2,4}` / `.MT88.{1,2,4}` (ldmatrix) | **fixtures/s_sm80** (all 6 forms, lane-dependent row addresses), bitwise |
| `LDGSTS.E.BYPASS[.LTC128B].128` (+ source predicate: zero fill), `LDGDEPBAR`, `DEPBAR.LE SB0, n` | s_sm80 (2 groups, wait 1 / 0, zero fill); both flash attentions (E-CPASYNC) |
| `STL[.128]`, `LDL[.LU]` (local memory) | s_sm80 (dynamically indexed array) |
| `LEA Rd, P, -Ra, Rb, s` | s_sm80 (64-bit pointer decrement) |
| `IDP.2A.{LO,HI}.{U16,S16}.{U8,S8}` (+ PTX `dp2a`) | **fixtures/s_dp2a**: all 8 forms on 128x128 operand pairs, both front-ends; cuDNN SDPA |
| `LDTM` / `STTM.16dp32bit_t0_t15` / `_t16_t31`; `UTCHMMA` with A in TMEM; MN-major B descriptors | Triton flash attention = GB300 (S-TMEM) |
| `LDTM.16dp256bit[.xN]`, `STSM.16.MT88.N` | cuBLAS nvjet matmuls (matmul_bf16, matmul_relu, mixed) = GB300 |
| `ATOM.E.ADD.F32.FTZ.RN` (P = global), `ATOM.E.CAST.SPIN`, `ATOMS.CAST.SPIN`, `QSPC.E.{S,D}` | torch index_add = GB300 (only the global path runs) |
| `SHFL.* P, ...` (in-range predicate; + PTX `d\|p`), `BMSK[.W]` / `UBMSK` (+ PTX `bmsk`) | **fixtures/s_bmsk**, both front-ends; torch radix sort = GB300 |
| `REDUX[.OR/.XOR/.SUM]`, `CREDUX.MIN/.MAX[.S32]`, `SR_LEMASK/GEMASK/GTMASK/EQMASK`, `BRA.DIV`, `WARPSYNC.COLLECTIVE` ... `ENDCOLLECTIVE` (run-time member masks; E-COLLECTIVE) | **fixtures/s_warpcoll**: data-dependent active sets, a shrinking-mask loop, 1/2/3 member-mask groups per warp, bitwise; torch.topk's gatherTopK certified clean, = GB300 |
| `STSM.16.M88.N` + a TMA store of a box narrower than its swizzle span (smem rows padded to the span, S-TMA) | **probes/tma_narrow_box** (13 swizzle x width configs, load and store); cuBLAS `nvjet_..._TTN` (fixtures/c_mm_bf16_ttn) = GB300 |
| `LDG.E.ENL2.256 Rhi, Rlo` | operand order checked against ptxas's `ld.global.v8.b32`; torch where / to_bf16 = GB300 |
| `F2F.F32.F64` / `.F64.F32` / `.F16.F32`, `F2FP.F16.F32.PACK_AB`, `HADD2.F32` (f16 -> f32), `F2I.S64/.U64.TRUNC`, `I2F.S64/.U64/.S16/.U16.H1/.S8`, `LDG.E.S8/.S16`, `LEA.HI -Rb` | **fixtures/s_cvt** (the s_alu2 operands and every pair as f64 / i64), bitwise; the NaN rules were measured there (payload-preserving F2F, 2^63 for F2I.S64(NaN)) |
| `DFMA` / `DMUL` / `DADD` `[.RZ/.RM/.RP]` (`-x`, `\|x\|`, f64 immediates), `DSETP` (16 comparisons), `DSETP.MIN/.MAX` (fmin / fmax) | **fixtures/s_f64**: 128x128 operand pairs (special f64 patterns + random), 11 words each, bitwise; measured the f64 NaN rules (propagation b, c, a with payload, invalid = 0xfff8000000000000) |
| PTX `add/sub/mul/fma.f64`, `div.{rn,rz,rm,rp}.f64`, `sqrt.{rn,rz,rm,rp}.f64` (correctly rounded) | **probes/f64_ptx**, PTX front, bitwise; the SASS expansion of div / sqrt needs agent B's MUFU.RCP64H / RSQ64H (merge) |
| PTX `div.full.f32`, `cvt.{rn,rz,rm,rp}.f32.{s32,u32}`, `selp.b64`, `st.v2` of virtual registers | **fixtures/s_ptxdiv**, both front-ends; Triton softmax / layer norm now pass on the PTX front |

Model changes the kernels forced:
- **S-PROXY credits any thread's fence** that observed the writes: `probes/proxy_fence`
  (rung 1b's pipeline with the fence after the barrier, by all threads or by the issuer only:
  1000/1000 correct on GB300; without a fence 1000/1000 wrong). Triton's
  `bar.sync; fence.proxy.async; tcgen05.mma` was a false proxy race.
- **E-CPASYNC**: cp.async writes land at the group's wait, race-checked with the issue clock.
- **L-OPREAD**: more in-order units (`LDSM`, `S2UR`, the conversion unit MUFU / F2I / I2F /
  F2F; STTM before LDTM; loads after any earlier queue read; a generic ATOM's predicate on its
  read scoreboard); the parser no longer reads `idesc[URx]` as 64-bit or `[UR7+..]` as a
  destination.
- **S-INIT**: spills keep undefined bits; a compare decided by its combining predicate does
  not use its operands (torch where).

## Round 3 (rung 3b, flashinfer trtllm-gen gemm1): instructions added

| form | validation |
|---|---|
| `UTMALDG.2D.GATHER4` (+ PTX `.tile::gather4`) | **fixtures/s_gather4** (6 gathers, 4 swizzles, out-of-bounds rows / columns, unaligned starts), both front-ends = GB300 1000/1000; unaligned columns are illegal instructions (variant `misaligned`, S-TMA-GATHER4) |
| `SYNCS.ARRIVE.TRANS64.RED.A0T1`, `ARRIVES.LDGSTSBAR.64.TRANSCNT` / `.ARVCNT` (cp.async.mbarrier.arrive [.noinc]) | **fixtures/s_cpasync_mbar** = GB300 1000/1000 (E-CPASYNC-MBAR) |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT PT, ...` (no-op), peek tokens through `SEL` / `ISETP` (E-SPIN) | fixtures/r3b_trtllm_bmm certified, = GB300 bitwise |

## Round 4 (rung 3b, the whole flashinfer MoE): instructions added

| form | validation |
|---|---|
| `F2FP.SATFINITE.BF16.F32.PACK_AB` (CvtPack, RN, satfinite) | **fixtures/s_routing_forms** = GB300 1000/1000 (overflow, +-Inf, NaN, ties) |
| `HSETP2.BF16_V2.<cmp>.AND` (register halves, immediate `-INF , -INF`) | s_routing_forms: all 14 `setp.<cmp>.bf16` on edge pairs |
| `ATOMS.POPC.INC.32 RZ, [Ra]` (each active lane adds 1) | s_routing_forms: divergent lanes, colliding counters |
| `S2R SR_SWINHI`, generic addresses in the shared window (S-SWIN): `LD.E` / `ST.E` / `LD.E.U16`, generic `mapa`, `cvta.shared` | s_routing_forms: this and the peer CTA's shared memory in a 2-CTA cluster |


## Round 1 (rung 4): instructions added (all through `sass` `lower_line`, semantics in `fp.rs` / `exec.rs`)

The second column says how each form was checked on GB300.

| form | validation |
|---|---|
| FADD / FMUL / FFMA `.RZ/.RM/.RP/.FTZ/.SAT`; `-x`, `\|x\|` operands (also with `.reuse`) | s_alu2, bitwise |
| FMNMX `[.FTZ/.NAN]`, FMNMX3 | s_alu2 |
| FSETP: 16 comparisons, `.FTZ`, `.AND/.OR/.XOR` | s_alu2 |
| FSEL (Sel), FSEL with `-x` / `\|x\|` (Fsel) | s_alu2 |
| FCHK | probes/fchk: patched cubin, 16,384 + boundary pairs; plus div.rn end to end in s_alu2 |
| FMUL2, FFMA2 with canonical NaN | s_alu2 |
| MUFU.{EX2,RCP,RSQ,SQRT,LG2,SIN,COS,TANH,RCP64H,RSQ64H}[.F16/.BF16/x2]: `Op::Mufu`, executed by the N-MUFU model | exhaustive sweep, probes/mufu; s_alu2 (the `MUFU.RSQ R, -QNAN` NaN idiom) |
| NaN immediates (`+QNAN`): bits taken from the encoding | s_alu2 |
| VIMNMX[.U32], POPC, SR_LTMASK | s_alu2; sum_torch = GB300 |
| LDG.E.U8 / .U16 (zero-extended) | s_alu2 |
| STG / ST.E `.U8/.S8/.U16/.S16`, ST.E (generic, global) | s_alu2 (STG.E.U8/U16); D's embedding / index_select captures |
| `@P EXIT Q` (conditional branch to a plain EXIT) | sum_torch = GB300 (wrote zeros before) |
| BREAK.RELIABLE (E-BSYNC) | D's add / fill captures, bitwise |
| VIADD with `-R` / `-UR` | D's captures |
| R2P | lowered, not exercised on hardware |

Model changes the kernels forced:
- **S-INIT V-bits.** Undefined bits propagate bit-precisely through register-only operations
  and are reported at sinks. This fixes false alarms on Triton's predicated `LDS` + `SHFL` and
  on the bf16 `PRMT` unpack.
- **Nested BSSY/BSYNC regions** (E-BSYNC).
- **Three L-OPREAD rules:**
  - a memory op's reads complete with its write scoreboard;
  - an untracked `LDS` lands before a later `SHFL`;
  - a load writes only after the earlier reads in the same in-order queue.

  Each rule was forced by a false positive on a kernel that matches GB300.

PTX front:
- Added:
  - `add/sub/mul/fma.f32` with `.ftz/.sat/.rz/.rm/.rp`;
  - `min/max.f32`;
  - `mad.wide.s32/u32`;
  - `shfl.sync.up/down/bfly`;
  - `{ %r }` brace operands;
  - `ex2/lg2/rsqrt/sqrt/sin/cos/tanh.approx`, `rcp/rsqrt.approx.ftz.f64` (to `Op::Mufu`).
- Fixed: `= { ... }` array initializers no longer swallow the kernel body. That bug caused
  the "Triton layernorm PTX parses to nothing" panic.
- Added in round 2: `div.full.f32`, `cvt.{rn,rz,rm,rp}.f32.{s32,u32}`, `dp2a`, `bmsk`,
  `shfl ... d|p`, `tcgen05.mma` with A in TMEM, `ld.s8/.s16` sign extension; fixed `selp.b64`
  (it truncated to 32 bits) and `st.v2.b32` of virtual registers (it stored the first only).
- Triton's cumsum and cat now pass on the PTX front bitwise.

## D's torch-op smoke set

`tools/capture_smoke.py` (captures on GB300, GPU 0, then `ptxv check`), 2026-11-03, round 2:
**all 33 ops lift, and on every one the simulator's output equals the GPU's bitwise.** 30 are
`clean`. The three cuBLAS nvjet matmuls report `contract`: the TMA store is still reading
shared memory at exit, which the PTX ISA forbids and GB300 drains (E-BULK-EXIT, as in rung 2e).
The Triton kernels also pass on the PTX front. (Round 1: 24 of 32 ran; base: 3.)

| op | kernels (cubins) | lifts (SASS) | missing SASS instructions | verdict | sim = GPU bitwise |
|---|---|---|---|---|---|
| softmax | softmax_warp_forward (1) | yes |  | sass: [clean] | sass: yes |
| log_softmax | softmax_warp_forward (1) | yes |  | sass: [clean] | sass: yes |
| layer_norm | at::native::vectorized_layer_norm_kernel (1) | yes |  | sass: [clean] | sass: yes |
| rms_norm | at::native::vectorized_layer_norm_kernel (1) | yes |  | sass: [clean] | sass: yes |
| sum_dim | at::native::reduce_kernel (1) | yes |  | sass: [clean] | sass: yes |
| max_dim | at::native::reduce_kernel (1) | yes |  | sass: [clean] | sass: yes |
| mean_dim | at::native::reduce_kernel (1) | yes |  | sass: [clean] | sass: yes |
| cumsum | at::native::tensor_kernel_scan_innermost_dim (1) | yes |  | sass: [clean] | sass: yes |
| gelu | at::native::vectorized_elementwise_kernel (1) | yes |  | sass: [clean] | sass: yes |
| silu | at::native::vectorized_elementwise_kernel (1) | yes |  | sass: [clean] | sass: yes |
| sdpa | cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_k (1) | yes |  | sass: [clean] | sass: yes |
| embedding | at::native::indexSelectSmallIndex (1) | yes |  | sass: [clean] | sass: yes |
| index_add | at::native::indexFuncSmallIndex (1) | yes |  | sass: [clean] | sass: yes |
| matmul_bf16 | nvjet_sm103_tst_64x8_64x16_1x4_h_bz_NNT (1) | yes |  | sass: [contract] | sass: yes |
| matmul_relu | nvjet_sm103_tst_64x8_64x16_1x4_h_bz_NNT<br>at::native::vectorized_elementwise_kernel (2) | yes |  | sass: [contract] | sass: yes |
| sort | elementwise_kernel_with_index<br>at::native::elementwise_kernel<br>at::native::radixSortKVInPlace (3) | yes |  | sass: [clean] | sass: yes |
| triton_softmax | softmax_kernel (1) | yes |  | sass: [clean]; ptx: [clean] (after adding cvt.rn.bf16.f32) | sass: yes |
| triton_layernorm | layernorm_kernel (1) | yes |  | sass: [clean]; ptx: [clean] (after adding cvt.rn.bf16.f32) | sass: yes |
| triton_cumsum | cumsum_kernel (1) | yes |  | sass: [clean]; ptx: [clean] | sass: yes; ptx: yes |
| triton_cat | cat_kernel (1) | yes |  | sass: [clean]; ptx: [clean] | sass: yes; ptx: yes |
| cute_add | kernel_cutlass_add_kernel_tensorptrf32gmemo41281281_tensorpt (1) | yes |  | sass: [clean]; ptx: [clean] | sass: yes; ptx: yes |
| mixed_3_modules | nvjet_sm103_tst_64x8_64x16_1x4_h_bz_NNT<br>at::native::unrolled_elementwise_kernel<br>scale_kernel (3) | yes |  | sass: [contract] | sass: yes |
| add_f32 | at::native::vectorized_elementwise_kernel (1) | yes |  | sass: [clean] | sass: yes |
| transpose_copy | at::native::elementwise_kernel (1) | yes |  | sass: [clean] | sass: yes |
| index_select | at::native::indexSelectSmallIndex (1) | yes |  | sass: [clean] | sass: yes |
| cat | at::native::CatArrayBatchedCopy_vectorized (1) | yes |  | sass: [clean] | sass: yes |
| mul_scalar_f32 | at::native::vectorized_elementwise_kernel (1) | yes |  | sass: [clean] | sass: yes |
| where | at::native::vectorized_elementwise_kernel<br>at::native::vectorized_elementwise_kernel (2) | yes |  | sass: [clean] | sass: yes |
| to_bf16 | at::native::vectorized_elementwise_kernel (1) | yes |  | sass: [clean] | sass: yes |
| flip | at::native::index_elementwise_kernel (1) | yes |  | sass: [clean] | sass: yes |
| gather | at::native::_scatter_gather_elementwise_kernel (1) | yes |  | sass: [clean] | sass: yes |
| fill | at::native::vectorized_elementwise_kernel (1) | yes |  | sass: [clean] | sass: yes |
| add_i32 | at::native::vectorized_elementwise_kernel (1) | yes |  | sass: [clean] | sass: yes |

## MUFU (N-MUFU)

- **Lifting.** `MUFU.<func>[.F16|.BF16|.F16x2|.BF16x2]` lifts as `Op::Mufu { func, fmt, hi,
  neg, abs, ptx: None }` (`-` / `|x|` operand modifiers and the `.H1` half selector kept); the
  PTX front maps every `<func>.approx[.ftz]` to the same op with `ptx: Some(ftz)`.
- **Execution.** `mufu.rs`: the units reconstructed from exhaustive GB300 sweeps, bit-exact on
  every input (NaN in, canonical NaN out, as before).
- **Fixtures.** The `good` variants of r4_softmax_*, r4_layernorm_*, r4_rmsnorm_* are
  `clean` and equal expected/ (`captured_outputs_match_the_gpu`); the `nosb` mutants are
  `violation`.
- **MUFU uses in the rung-4 kernels** (static counts):

  | kernel | MUFU uses |
  |---|---|
  | softmax_triton | EX2 2 |
  | softmax_torch | EX2 8, RSQ 1 |
  | layernorm_triton | SQRT 1 |
  | layernorm_torch | RSQ 2 |
  | rmsnorm_triton | RSQ 1 |
  | rmsnorm_torch | RSQ 2 |
  | attn_triton | EX2 66 |
  | sdpa_flash_torch | EX2 260, LG2 4 |

  Every kernel also uses RCP, either in ptxas's div.rn sequence or in integer division. RCP is
  modelled as the correctly rounded reciprocal (N-MUFU). The FFMA refinement then makes
  div.rn's quotient bitwise equal to GB300's in s_alu2.
- **Earlier dev result.** With a correctly rounded RSQ/SQRT stub (not committed), the four
  norms matched GB300 bitwise. EX2 needs the hardware's approximation.

## Still not lifting

- Nothing in the rung-4 / rung-5 fixtures or D's smoke set.
- Known unmodeled paths: `MUFU.TANH` inside 2^-8 <= |x| < 8 (N-MUFU); HMMA / tcgen05.mma with
  subnormal / Inf / NaN operands (N-MMA, N-HMMA); generic addresses in the shared window
  (`QSPC` returns false, `cvta.shared` is rejected); `LDGSTS` other than `.128`; HMMA other
  than `.16816.F32.BF16`.
