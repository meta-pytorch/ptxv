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
| `HMMA.16816.F32.BF16` (warp-collective; PTX ISA fragments; PTX `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` and the m16n8k8 tf32 form since 2026-11-07) | **probes/hmma**: `ptxv calib-mma` over 820 candidate models, 6 input sets, 98,304 outputs: exactly `g16-f25-rz-accin-tzero-einputs` matches all (the tcgen05 model; N-HMMA); expected/ in cargo test; torch flash attention = GB300 |
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
| `SYNCS.ARRIVE.TRANS64.ART0` / `.RED.ART0` (mbarrier.arrive with a count) | probes/mbar_count = GB300, both front-ends (S-MBAR) |
| `SYNCS.ARRIVE.TRANS64.RED.A0T1`, `ARRIVES.LDGSTSBAR.64.TRANSCNT` / `.ARVCNT` (cp.async.mbarrier.arrive [.noinc]) | **fixtures/s_cpasync_mbar** = GB300 1000/1000 (E-CPASYNC-MBAR) |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT PT, ...` (no-op), peek tokens through `SEL` / `ISETP` (E-SPIN) | fixtures/r3b_trtllm_bmm certified, = GB300 bitwise |

## Round 4 (rung 3b, the whole flashinfer MoE): instructions added

| form | validation |
|---|---|
| `F2FP.SATFINITE.BF16.F32.PACK_AB` (CvtPack, RN, satfinite) | **fixtures/s_routing_forms** = GB300 1000/1000 (overflow, +-Inf, NaN, ties) |
| `HSETP2.BF16_V2.<cmp>.AND` (register halves, immediate `-INF , -INF`) | s_routing_forms: all 14 `setp.<cmp>.bf16` on edge pairs |
| `ATOMS.POPC.INC.32 RZ, [Ra]` (each active lane adds 1) | s_routing_forms: divergent lanes, colliding counters |
| `S2R SR_SWINHI`, generic addresses in the shared window (S-SWIN): `LD.E` / `ST.E` / `LD.E.U16`, generic `mapa`, `cvta.shared` | s_routing_forms: this and the peer CTA's shared memory in a 2-CTA cluster |


## Round 5 (kernel DB: msl_ops / mslk, agent L2): instructions added

Lowered in `crates/sass/src/l2.rs` (and the PTX forms in `crates/ptx/src/l2.rs`); every other
modifier of these opcodes is refused.

| form | validation |
|---|---|
| `FRND[.F64\|.F16\|.BF16][.FTZ][.TRUNC\|.FLOOR\|.CEIL]` (+ PTX `cvt.r{n,z,m,p}i[.ftz]` between equal float types) | **probes/fcvt**, 8192 edge / tie / random inputs per form, both front-ends bitwise (N-FCVT) |
| `F2F.{BF16,F16}.F64[.RZ\|.RM\|.RP]`, `F2F.F64.{BF16,F16} Rd, Ra[.H1]`, `F2F.BF16.F16[.R*]`, `F2F.F16.BF16[.R*]` (+ PTX `cvt`) | probes/fcvt (single rounding from f64, f64 NaN payloads kept, f16 <-> bf16 NaN canonical) |
| `VIMNMX.{U16x2,S16x2}`, `VIMNMX3.{U16x2,S16x2}` (+ PTX `min` / `max .u16x2 / .s16x2`) | **probes/int16x2**, both front-ends (N-INT16X2) |
| `LEA.HI.X.SX32 Rd, Ra, ~Rb, s, P`; `LEA Rd, Pc, Ra, -Rb, s` | probes/int16x2: ptxas's `IMAD.X ~Ry` rewritten into the LEA.HI.X form (cubins/lea.cubin), same GB300 bytes |
| `FCHK P, -Ra, b` | **probes/fchk_neg**: probes/fchk with the negate bit, identical predicates (N-FCHK) |
| `FCHK P, \|Ra\|, \|Rb\|` | **probes/fchk_abs**: probes/fchk with both abs bits, identical predicates (N-FCHK) |
| `RPCMOV.32 Rpc.LO, Ra` / `RPCMOV.32 Rd, Rpc.LO` | **probes/rpcmov** (patched cubin; L-RPCMOV): a per-thread slot past the register count; kernels with a non-trap CALL refused |
| `F2I[.FTZ].{S16,U16}[.TRUNC\|.FLOOR\|.CEIL].NTZ` (+ PTX `cvt.r*i[.ftz].{s16,u16}.f32`) | **probes/f2i16**: saturated, NaN 0, extended in the register (N-F2I16) |
| `CCTL.E.PF1` / `CCTL.E.PF2` (prefetch.global.L1 / .L2): no effect | probes/f2i16 runs both |
| `BREAK[.RELIABLE] Pc, Bb` | **probes/div_ftz**: div.rn.ftz.f32 slow path on edge pairs (E-BSYNC) |
| `UTCQMMA` / `tcgen05.mma` block-scaled M = 128 N = 8 (shape check relaxed) | **probes/mma_bs expected/n8**, both front-ends (N-MMA-BS, S-TMEM) |
| `ATOMG.E.INC[.STRONG.GPU]` (+ PTX `atom.inc.u32`) | **probes/atom_inc**, both front-ends (L-ROSETTA) |
| `BMOV.32[.CLEAR] Rd, Bb`, `BMOV.32 Bb, Ra` | the parallel_attn_bwd_dkv case (E-BSYNC); not probed alone |
| `[R.U32+UR]` generic addresses, `UTMAREDG ... desc[URZ]`, device-built tensor maps (words 0x40/0x48 = 0) | operand syntax; **probes/tmap_device**, **probes/tmap_param_copy** (S-TMAP; raw tensor-map parameter bytes visible to device code) |
| `LDSM.8.MT1616[.2]` (ldmatrix.m16n16.x1 / .x2.trans.b8) | **probes/ldsm_b8**, SASS (L-ROSETTA) |
| PTX `tcgen05.st / ld .16x32bx2 [a], split` | **probes/tmem_x2**, both front-ends (S-TMEM) |
| `LDTM.STAT.16dp32bit_t0_t15 / _t16_t31 .xN.MIN/MAX[ABS].F32[.NAN]` | **probes/tmem_ldred16**, SASS (S-TMEM) |
| `STTM.16dp256bit[.xN]`, `STTM.16dp128bit[.xN]` (+ PTX `tcgen05.st.16x256b / .16x128b`) | **probes/tmem_st16**, both front-ends (S-TMEM) |
| `F2FP[.RELU].F16.E2M1.UNPACK_B Rd, Ra.B1 / .B2 / .B3` (+ PTX `cvt.rn[.relu].f16x2.e2m1x2` of a `mov.b32 {..}` byte) | **probes/e2m1_bytes**, both front-ends (N-CVT-LP) |
| `VIMNMX.RELU`, `VIMNMX.S16x2.RELU`, `VIADDMNMX.RELU` (+ PTX `min / max.relu.s32 / .s16x2`) | **probes/imnmx_relu**, both front-ends (N-INT16X2) |
| `UTMAREDG.<N>D.ADD` (+ PTX `cp.reduce.async.bulk.tensor...add.tile.bulk_group`) on f32 / f16 / bf16 tensors | **probes/tma_red**, edge pairs per type, both front-ends (N-BULK-RED) |
| `UBLKCP.S.S [URd], [URs], URn` (+ PTX `cp.async.bulk.shared::cluster.shared::cta`) | **probes/bulk_s2s**: 2-CTA cluster, peer and self copies, both front-ends (S-TMA) |
| `UBLKRED.G.S.ADD.F32.RN [URg], [URs], URn` (+ PTX `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`) | **probes/bulk_red**: per-word atomic f32 add (RN, no flush, canonical NaN), four CTAs on the same words, both front-ends (N-BULK-RED) |

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

## Round 5 (mslk / msl_ops kernel DB): instructions added, each with a GB300 probe

Every form below fails closed on modifiers its probe does not cover. Since 60b9a5f a
per-opcode modifier gate (`crates/sass/src/modset.rs`, test `tests/modifier_audit.rs`) refuses
any modifier outside the audited set before an arm runs.

| SASS form | PTX source | semantics (GB300) | probe |
|---|---|---|---|
| `FMNMX.XORSIGN` | `min` / `max.xorsign.abs.f32` | min/max of \|a\|, \|b\| (FMNMX NaN rules); sign = sign(a) xor sign(b) of the stored operands, a NaN operand's sign included and `\|UR\|` ignored for the sign; a NaN result is 0x7fffffff | probes/fmnmx_xorsign |
| `HFMA2` / `HMUL2` / `HADD2.BF16_V2[.RELU]` | `fma` / `mul` / `add` / `sub.rn.bf16x2`, `neg.bf16x2` | per lane: one rounding to bf16 (RNE), subnormals kept, every NaN result 0x7fff (neg of a NaN too); `-RZ` addend keeps -0 | probes/bf16x2 |
| `FHFMA` / `FHADD.BF16`, `HMNMX2.BF16_V2[.NAN][.XORSIGN]` | `fma` / `add.f32.bf16`, `min` / `max.bf16x2` | f32 FMA of the widened halves, one rounding in the instruction's mode; min/max as FMNMX on the widenings, NaN 0x7fff | probes/bf16_ext |
| `REDG.E.ADD.BF16x2.RN`, `ATOMG.E.ADD.BF16x2.RN` | `red` / `atom.add.noftz.bf16x2` | per lane bf16 add, RNE, NaN 0x7fff; not commutative-exact (excluded from C-ORDER) | probes/bf16x2 |
| `MUFU.RSQ` + Newton step + slow-path `CALL` (ptxas) | `sqrt.rn.f32` | IEEE square root, round to nearest even; negative / NaN inputs give 0x7fffffff | probes/sqrt_rn |
| `LD.E.U8` / `LD.E.S8` / `ST.E.U8`; `ATOMG` / `REDG .E.ADD.STRONG.SM` | `ld` / `st.u8`, `ld.s8` (generic); `atom` / `red.global.cta` | byte accesses (S8 sign-extended); `.SM` = CTA scope | probes/generic8 |
| `REDG.E.ADD.F32x4` / `F32x2.FTZ.RN` | `red.global.add.v4` / `.v2.f32` | one FADD.FTZ atomic per element, each on its own | probes/red_vec |
| `BRX Ra -off` (BRANCH_TARGETS) | `brx.idx[.uni] i, $L` / `$L: .branchtargets ...` | per-thread jump to target i; an index outside the list is a fault | probes/brx_idx |
| `REDUX.SUM.S32` | `redux.sync.add.s32` | wrapping sum over the active lanes (partial masks) | probes/redux_s32 |
| `LOP3.LUT ..., 0x80000000, ..., 0xb8` | `copysign.f32` | b with the sign bit of a (bit operation) | probes/copysign |
| `REDUX` / `CREDUX` | `redux.sync` with a partial immediate or register membermask | reduction over the lanes the mask names; executing lanes != mask is a violation (PTX) | probes/redux_mask |
| `SHFL` / `VOTE` | `shfl.sync` / `vote.sync` with a partial immediate or register membermask | per mask group, combined with the group's lanes at another instruction of the same kind; executing lanes != mask is a violation (PTX) | probes/coll_split, probes/shfl_mask (PTX front; SASS `WARPSYNC.EXCLUSIVE` refused) |
| `FFMA2` / `FMUL2` / `FADD2 .RZ/.RM/.RP` | `.rz/.rm/.rp.f32x2` | one rounding in the mode | probes/bf16_ext |
| `IMAD.WIDE R, a, b, -c` | `mul.wide` + `sub.s64` | a*b - c mod 2^64 | probes/int_misc |
| `POPC` | `popc.b32` | variable latency, in-order queue (scoreboard cover) | probes/int_misc |
| `UP2UR` | (CuTe predicate spill) | uniform P2R | lift_l_forms |
| `VIADD.16x2` | `add.u16x2` / `add.s16x2` | two 16-bit lanes mod 2^16, no carry between lanes | probes/viadd16x2 |
| `LDG.E.ENL2.256 Rhi, Rlo` / `STG.E.ENL2.256 [a], Rlo, Rhi` | `ld` / `st.global.v8.b32` | 32-byte access, 32-byte aligned; bytes 0..16 in the `lo` quad (second operand of the load, first data operand of the store) | probes/ldst256 |
| `LDG` / `STG .E.ELL2.256`, `.E.EFL2.256` | `ld` / `st.global.L2::evict_last` / `L2::evict_first .v8.b32` | as ENL2 (eviction hint dropped, L-PTX-HINT) | probes/ldst256_l2 |
| `FFMA` / `FFMA.FTZ` / `FFMA.SAT` / `DFMA` | `mul` + `add` / `sub` `.f32` / `.f64` without rounding modifiers | ptxas 13.0's default contraction where unambiguous (one reader, one block, same `.ftz`, unguarded); ambiguous shapes refused (PTX-FMAD) | probes/fmad |
| `HFMA2.BF16_V2 d, -RZ.H0_H0, RZ.H0_H0, \|R\|[.H1_H1]` | `abs.bf16x2` / `abs.bf16` | sign bits cleared, then -0*0 + \|x\|: NaN 0x7fff, -0 to +0 | probes/bf16_abs |
| `HFMA2` / `HMUL2` / `HADD2` (f16) with `.H0_H0` / `.H1_H1`, `.reuse`, immediate pairs | `fma` / `mul` / `add` / `sub .rn.f16[x2]`, `mul.f16` that cannot be contracted | per lane one f16 rounding; HMUL2 = a*b + -0, HADD2 = a*1 + c | probes/f16_arith |
| `LDSM.16.M88.N` / `.MT88.N` | `ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16` | (PTX front added) | probes/ldmatrix_ptx |
| `LDG.E.NA[...]`, `ST.E.EF.128`, `LD` / `ST` / `ATOM` `.STRONG.SM` | `.L1::no_allocate`, `.L1::evict_first`, `.cta` scope | cache hints dropped; `.SM` = CTA scope | probes/memhints |
| `STG.E.64.MMIO.SYS` | `st.mmio.relaxed.sys` | strong relaxed `.sys` store; other `.MMIO` forms refused | probes/memhints |
| `UBLKPF.L2`, `R2UR.OR P, UR, R` | `cp.async.bulk.prefetch.L2` | no-op; `.OR` ORs the lane's "differs" bit into P | probes/memhints |
| `FLO` scoreboards | - | FLO shares POPC's in-order queue (L-OPCLASS) | probes/flo_popc |
| `F2F.F32.F64` / `F2F.F64.F32`, `FADD.SAT d, RZ, a`, `F2I...NTZ` | `cvt.rn.f32.f64` / `cvt.f64.f32`, `cvt.sat.f32.f32`, `cvt.r{n,m,p,z}i[.ftz].{s,u}{32,64}.f32` (PTX front added) | as the SASS ops | probes/cvt_misc |
| `FADD[.FTZ] d, \|a\|, -RZ`, `IABS` | `abs[.ftz].f32`, `abs.s32` | \|a\| + -0 (canonical NaN); two's-complement abs | probes/abs_f32 |
| `S2R SR_LTMASK` etc. | `%lanemask_{lt,le,ge,gt,eq}`; `%`-less scope registers; `[%rd+-N]`; `{...}` operands continued after a comma | PTX syntax only | probes/ptx_syntax |
| `FMUL2` / `FADD2` | `mul` / `add` / `sub.f32x2` with no rounding modifier | .rn where no use lets ptxas contract (a mul only multiplied, stored, converted, an fma addend, or unpacked into halves read by scalar ops or repacked across lanes; an add / sub of no such mul) | probes/f32x2_norn, probes/f32x2_unpack |
| `HADD2.BF16_V2`, `HFMA2.BF16_V2 d, a, b, -RZ` | `add` / `sub` / `mul` `.bf16[x2]` with no rounding modifier | .rn under the f32x2 rule | probes/bf16_norn |
| ptxas's div.rn.f32 expansion (MUFU.RCP, FFMA refinement, slow-path CALL) | `rcp.rn.f32` | IEEE 1 / a, NaN canonical | probes/rcp_rn |
| `F2F.F32.F64.{RZ,RM,RP}` | `cvt.{rz,rm,rp}.f32.f64` | directed rounding | probes/cvt_f64_rnd |
| `BAR.RED.{POPC,AND,OR}[.DEFER_BLOCKING] id[, n], {!}P`, `B2R.RESULT Rd[, Pd]` | `bar.red.popc.u32`, `bar.red.{and,or}.pred` | reduction over the generation's waiters (E-BAR-RED) | probes/bar_red |
| `I2F[.F64][.S16\|.U16\|.S64\|.U64][.RZ\|.RM\|.RP]` | `cvt.<rnd>.{f32,f64}.{s,u}{16,32,64}` | as the SASS ops | probes/cvt_i2f |
| negate + select (64-bit), `BREV`, ptxas's f64 division expansion | `abs.s64`, `brev.b32`, `rcp.<rnd>.f64` | two's-complement abs; bit reverse; 1 / a | probes/ptx_misc2 |
| `UTCHMMA` with an f16 or tf32 instruction descriptor (A / B format 0 / 2) | `tcgen05.mma.kind::f16` (f16 A / B), `.kind::tf32` | N-MMA over 16 f16 / 8 tf32 (truncated) products per step | probes/tc_f16, probes/tc_tf32 |
| `STTM` with a negative TMEM offset (`tmem[UR+-0x60]`) | `tcgen05.st [taddr - n]` | offset added to the 32-bit TMEM address | probes/tmem_stneg |
| `UTCHMMA` with offset TMEM addresses (ptxas adds them into URs) | `tcgen05.mma [taddr + off], [a + off], ...` | offset added to the TMEM address word | probes/tc_mma_off |
| `FFMA2 d, a, -URn.F32[x2.HI_LO], c`, `FMUL2 d, \|R\|.F32x2.HI_LO, imm` | `neg.f32` of parameters / `abs.f32` of halves feeding `.f32x2` ops | negated uniform operand; absolute value | probes/ffma2_ur |
| `UTMAPF.L2.<n>D` | `cp.async.bulk.prefetch.tensor.<n>d.L2.global[.tile]` | no-op (cache hint) | probes/tma_prefetch |
| `ATOMG` / `REDG` `.E.ADD.F64.RN` | `atom` / `red` `.add.f64` | one f64 add, nearest (as DADD) | probes/atom_f64 |
| `IMAD` on the extended operand | `mad.wide.{s,u}16 d, a, imm, c` | 32-bit ext16(a) * ext16(imm) + c | probes/mad_wide16 |
| `HSET2[.BF16_V2][.BF].<cmp>.<bop>` | `set.<cmp>.{u32,bf16x2,f16x2}.{bf16x2,f16x2}` | per half 0xffff / 0, `.BF` 1.0 / 0 | probes/hset2 |
| `MOVM.16.MT88` | `movmatrix.sync.aligned.m8n8.trans.b16` | warp transpose of 8x8 16-bit elements | probes/movm |
| `HMMA.1688.F32.BF16` | `mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32` | N-HMMA over 8 products (calibrated) | probes/bmma8 |
| `LDGSTS...ZFILL [d], [a + size - s], P` | `cp.async ..., size, s` (s immediate or register) | first s bytes from the rounded-down address, the rest zero; !P = s 0 | probes/cpasync_srcsize |
| `REDUX`, `FRND` scoreboards | - | REDUX is one in-order queue; FRND joins the MUFU / conversion queue (L-OPCLASS) | msl_ops captures |
| f64 results without a scoreboard | - | complete 128 counted stall cycles after issue (`F64_LAT`, ptxas's NOP padding) | msl_ops captures |
| `HMNMX2.BF16_V2 d, a, hi, lo, p`, `VHMNMX ... RZ.Hx_Hx` | `min` / `max.bf16x2` with constants | bf16 immediate pair from the encoding (checked against the print); RZ = +0 | msl_ops captures |
| SASS `SHFL` in a lane-subset BSSY region | (warp-aggregated atomics) | runs over the lanes present; lanes parked at a BSYNC are not waited for | DeepEP hybrid, Triton MoE |
| f64 pipe scoreboards | - | DADD / DMUL / DFMA / DSETP complete in order (L-OPCLASS) | probes/f64_queue, probes/f64_dsetp |
| `DSETP.<cmp>.AND` | `setp.<cmp>.f64` (all 14 compares; no p\|q, no combining predicate) | as SASS DSETP | probes/f64_dsetp |
| `STTM` then `LDTM` on the same registers | - | the TMEM queue reads the store's registers first (L-OPREAD) | probes/sttm_ldtm |
| `FENCE.VIEW.ASYNC.S` / `.G` / `.T` | `fence.proxy.async[.shared::*]` / `.global` / `tcgen05.wait::st` | each orders only its state space; `.G` covers no shared-memory check, `.T` is `tcgen05.wait::st` (S-PROXY) | probes/proxy_fence `global` |

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
