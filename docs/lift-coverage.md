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
| `UTMALDG.2D.GATHER4.MULTICAST[.2CTA]` (+ PTX `.tile::gather4 ... .multicast::cluster[.cta_group::2]`) | **probes/tma_g4_mcast** (56 configs of ctaMask, mbarrier and destination rank, cluster of 4), both front-ends = GB300 (S-TMA-2CTA-MCAST) |
| `SYNCS.ARRIVE.TRANS64.ART0` / `.RED.ART0` (mbarrier.arrive with a count) | probes/mbar_count = GB300, both front-ends (S-MBAR) |
| `SYNCS.ARRIVE.TRANS64.RED.A0TR` / `.A0TX` (mbarrier.expect_tx / complete_tx, PTX `[.relaxed] [.cta / .cluster]`) | probes/mbar_expect_tx = GB300, both front-ends; relaxed: publishes only what a prior fence released (S-MBAR) |
| `UREDGR.E.ADD.NA[.S32 / .64].STRONG.GPU / .SYS` (red.async.release.{gpu,sys}.global.add) | probes/red_async = GB300, both front-ends; release order: probes/redasync_litmus (S-RED-ASYNC). `.relaxed` on `.global` and the other operations, which the PTX ISA does not define there, are refused on the PTX front (ptxas compiles them to the release add: probes/red_async_ops = GB200) |
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
| `UTCQMMA` / `tcgen05.mma.kind::mxf8f6f4` with A in TMEM or MN-major A / B (SWIZZLE_128B), 8-bit formats, M = 128 | **probes/mma_mxf8_tmem** = the K-major path on GB300 bitwise, both front-ends (N-MMA-BS, S-TMEM); the other swizzles, unswizzled MN-major and cta_group::2: **probes/mma_bs/mn**, **probes/mma_bs/ts** below |
| `LDC.U16` (zero-extending 16-bit constant load, odd halfwords of a word) | **probes/ldc_u16**, both front-ends = GB300 |
| `LDC` / `LDCU` `c[0x0][R+-off]` (register-indexed, negative offset) | **probes/cbank_neg**, both front-ends = GB300 |
| `ATOMG.E.INC[.STRONG.GPU]` (+ PTX `atom.inc.u32`) | **probes/atom_inc**, both front-ends (L-ROSETTA) |
| `BMOV.32[.CLEAR] Rd, Bb`, `BMOV.32 Bb, Ra` | the parallel_attn_bwd_dkv case (E-BSYNC); not probed alone |
| `[R.U32+UR]` generic addresses, `UTMAREDG ... desc[URZ]`, device-built tensor maps (words 0x40/0x48 = 0) | operand syntax; **probes/tmap_device**, **probes/tmap_param_copy** (S-TMAP; raw tensor-map parameter bytes visible to device code) |
| PTX `tensormap.replace.tile.*.shared::cta.b1024`, `tensormap.cp_fenceproxy`, `fence.proxy.tensormap::generic.acquire` | ptxas's LDS / LOP3 / STS read-modify-write per field, per-lane release copy, TmapFence; **probes/tmap_replace** (every field / ordinal / value), tmap_device and tmap_param_copy on the PTX front (S-TMAP) |
| `LDSM.8.MT1616[.2]` (ldmatrix.m16n16.x1 / .x2.trans.b8) | **probes/ldsm_b8**, SASS and PTX fronts (L-ROSETTA) |
| `STSM.8.MT168[.2\|.4]` (stmatrix.m16n8.x1 / .x2 / .x4.trans.b8) | **probes/stsm_b8**, SASS and PTX fronts (L-ROSETTA) |
| PTX `tcgen05.st / ld .16x32bx2 [a], split` | **probes/tmem_x2**, both front-ends (S-TMEM) |
| `LDTM.STAT.16dp32bit_t0_t15 / _t16_t31 .xN.MIN/MAX[ABS].F32[.NAN]` | **probes/tmem_ldred16**, both fronts (S-TMEM) |
| `LDTM.STAT.xN.MIN/MAX[ABS].F32[.NAN]` (`tcgen05.ld.red.32x32b`) | **probes/tmem_ldred**, both fronts (S-TMEM) |
| `STTM.16dp256bit[.xN]`, `STTM.16dp128bit[.xN]` (+ PTX `tcgen05.st.16x256b / .16x128b`) | **probes/tmem_st16**, both front-ends (S-TMEM) |
| `F2FP[.RELU].F16.E2M1.UNPACK_B Rd, Ra.B1 / .B2 / .B3` (+ PTX `cvt.rn[.relu].f16x2.e2m1x2` of a `mov.b32 {..}` byte) | **probes/e2m1_bytes**, both front-ends (N-CVT-LP) |
| `VIMNMX.RELU`, `VIMNMX.S16x2.RELU`, `VIADDMNMX.RELU` (+ PTX `min / max.relu.s32 / .s16x2`) | **probes/imnmx_relu**, both front-ends (N-INT16X2) |
| `VIADDMNMX.RELU d, a, -Rb, c, sel` (ptxas's fusion of `sub.s32` with `min / max.relu.s32`) | **probes/iaddmnmx_relu** (4 forms), both front-ends |
| `UTMAREDG.<N>D.ADD` (+ PTX `cp.reduce.async.bulk.tensor...add[.tile].bulk_group[.L2::cache_hint]`) on f32 / f16 / bf16 tensors | **probes/tma_red**, edge pairs per type, both front-ends (N-BULK-RED) |
| `UBLKCP.S.S [URd], [URs], URn` (+ PTX `cp.async.bulk.shared::cluster.shared::cta`) | **probes/bulk_s2s**: 2-CTA cluster, peer and self copies, both front-ends (S-TMA) |
| `UBLKRED.G.S.ADD.F32.RN [URg], [URs], URn` (+ PTX `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`) | **probes/bulk_red**: per-word atomic f32 add (RN, no flush, canonical NaN), four CTAs on the same words, both front-ends (N-BULK-RED) |
| `UTCOMMA[.2CTA].4X` (sm_100a's spelling of `kind::mxf4nvf4.block_scale.block16`; sm_103a: `.BLOCK16`) | **probes/tc_pair_bs/sm100** = GB200 = GB300's output, both front-ends (N-MMA-BS) |
| `UTCHMMA tmem[A], ...` with M = 128, N % 16 == 8 (outside the PTX ISA table) | **probes/tc_mma_ts_n8** (N = 8, 24) = GB200, both front-ends; a `contract` finding (S-TMEM) |
| `UTCQMMA[.2CTA] tmem[A], gdesc[B], ...` (narrow / block-scaled kinds with A in TMEM) | **probes/mma_bs/ts** = GB200 (= A from shared memory), both front-ends (S-TMEM) |
| `UTCQMMA[.2CTA]` idesc bit 15 / 16 with 8-bit operands (`tcgen05.mma` kind::f8f6f4 / mxf8f6f4 with MN-major A / B: SWIZZLE_128B / 64B / 32B or none, a start inside a swizzle atom) | **probes/mma_bs/mn**: 9 one-CTA and 7 cta_group::2 layouts (A in TMEM or shared memory) = GB200 = the K-major D, both front-ends (S-TMA, N-MMA-BS) |

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
| `FMNMX3 \|a\|, \|b\|, \|c\|` `[.FTZ/.NAN]` | `min` / `max [.ftz] [.NaN] .abs.f32` with three sources | min / max of the absolute values (FMNMX3 NaN rules) | probes/fmnmx3_abs |
| `HFMA2` / `HMUL2` / `HADD2.BF16_V2[.RELU]` | `fma` / `mul` / `add` / `sub.rn.bf16x2`, `neg.bf16x2` | per lane: one rounding to bf16 (RNE), subnormals kept, every NaN result 0x7fff (neg of a NaN too); `-RZ` addend keeps -0 | probes/bf16x2 |
| `FHFMA` / `FHADD.BF16`, `HMNMX2.BF16_V2[.NAN][.XORSIGN]` | `fma` / `add.f32.bf16`, `min` / `max.bf16x2` | f32 FMA of the widened halves, one rounding in the instruction's mode; min/max as FMNMX on the widenings, NaN 0x7fff | probes/bf16_ext |
| `QMUL4.E4M3x4.E2M1x4.E4M3x4` (`.N3210` / `.N7654`, `.B0`) | (no PTX form found) | e2m1 x e4m3 per byte, nearest even, overflow / NaN scale 0x7f | probes/qmul4 (exhaustive, patched-in encoding) |
| `UTMAREDG.ND.ADD` bf16 / f16, unordered reductions into one element | `cp.reduce.async.bulk.tensor...add` (16-bit) | any sequential order, one 16-bit rounding per add; nondeterministic, bounded over every order | probes/tma_red_order (N-RED16) |
| `F2FP.F16.F32.PACK_AB.RZ` | `cvt.rz.f16x2.f32` | f32 -> f16 toward zero, \|x\| >= 65536 to 0x7bff, NaN 0x7fff | probes/f2fp_rz (every f32 per lane, on the GPU) |
| `FHFMA` / `FHADD` (f16, no `.BF16`; `.RZ/.RM/.RP/.SAT`, `.H1`, `-c`) | `fma` / `add` / `sub .f32.f16` | f32 FMA of the exactly widened f16 halves, one rounding in the instruction's mode, NaN 0x7fffffff, `.SAT` to [0, 1] | probes/fhadd (exhaustive per half) |
| `HMNMX2[.NAN][.XORSIGN]` (f16; `\|R\|`, `.Hx_Hx`, immediate pairs) | `min` / `max` [.NaN] [.xorsign.abs] `.f16x2` / `.f16` | FMNMX's rules per f16 lane, `\|R\|` without `.XORSIGN` clears the sign, NaN 0x7fff | probes/hmnmx2 (exhaustive lane 0) |
| `REDG.E.ADD.BF16x2.RN`, `ATOMG.E.ADD.BF16x2.RN` | `red` / `atom.add.noftz.bf16x2` | per lane bf16 add, RNE, NaN 0x7fff; not commutative-exact (excluded from C-ORDER) | probes/bf16x2 |
| `MUFU.RSQ` + Newton step + slow-path `CALL` (ptxas) | `sqrt.rn.f32` | IEEE square root, round to nearest even; negative / NaN inputs give 0x7fffffff | probes/sqrt_rn |
| `LD.E.U8` / `LD.E.S8` / `ST.E.U8`; `ATOMG` / `REDG .E.ADD.STRONG.SM` | `ld` / `st.u8`, `ld.s8` (generic); `atom` / `red.global.cta` | byte accesses (S8 sign-extended); `.SM` = CTA scope | probes/generic8 |
| `REDG.E.ADD.F32x4` / `F32x2.FTZ.RN` | `red.global.add.v4` / `.v2.f32` | one FADD.FTZ atomic per element, each on its own | probes/red_vec |
| `ATOMG.E.ADD.F32x4` / `F32x2.FTZ.RN` | `atom.global.add.v4` / `.v2.f32` | as REDG, each element's old value returned; destination overlapping a later source refused | probes/atom_vec |
| `BRX Ra -off` (BRANCH_TARGETS) | `brx.idx[.uni] i, $L` / `$L: .branchtargets ...` | per-thread jump to target i; an index outside the list is a fault | probes/brx_idx |
| `REDUX.SUM.S32` | `redux.sync.add.s32` | wrapping sum over the active lanes (partial masks) | probes/redux_s32 |
| `LOP3.LUT ..., 0x80000000, ..., 0xb8` | `copysign.f32` | b with the sign bit of a (bit operation) | probes/copysign |
| `LOP3.LUT d, a, b, c, lut` | `lop3.b32 d, a, b, c, immLut` | the LUT on (a, b, c) = (0xF0, 0xCC, 0xAA); register or immediate operands; `d\|p` and BoolOp forms refused | probes/lop3_ptx |
| `HMMA.16816.F32` | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | N-HMMA model with f16 elements; f16 subnormal / Inf / NaN refused | probes/hmma_f16 |
| `F2F.BF16.F16` / `F2F.F16.BF16` | `cvt.bf16.f16` / `cvt.f16.bf16` without a rounding modifier | nearest | probes/cvt_bf16_f16 |
| `MUFU.RSQ64H` + `DMUL` / `DFMA` + slow-path `CALL` | `rsqrt.approx.f64` | ptxas's refinement and special cases | probes/rsqrt_f64 |
| `DSETP.MIN/MAX.AND` + `SEL` / `FSEL` + `LOP3 0x80000`; `DADD -RZ, -a / \|a\|` | `min` / `max.f64`; `neg` / `abs.f64` | as the SASS ops | probes/f64_misc |
| ptxas's div.rn / sqrt.rn FTZ expansions | `div.rn.ftz.f32`, `sqrt.rn.ftz.f32` | IEEE on flushed operands, result flushed | probes/ftz_div_sqrt |
| `STL.64` / `.128`, `LDL.64` / `.128` | `ld` / `st` `.local` `.b64` / `.f64` / `.v2` / `.v4` | per element; a 64-bit element is one register | probes/local_vec |
| `S2R SR_CTAID` + `LDC c[0x0][0x2a0.. - 0xa0 * SR_CgaSize]` | `%cluster_ctaid`, `%cluster_nctaid`, `%clusterid`, `%nclusterid` | ctaid mod / div the cluster shape | probes/cluster_sreg |
| `PRMT` | `mov.b32 r, {b0, b1, b2, b3}`, `mov.b16 h, {lo, hi}`, `mov.b16 {lo, hi}, h` | byte pack / unpack | probes/mov_pack |
| `UTCHMMA.2CTA tmem[..], gdesc[..]` at M = 128 | `tcgen05.mma.cta_group::2` M = 128 with A in TMEM | each D lane reads its own lane's A row | probes/tc_pair_atmem |
| `UTCHMMA` idesc bit 15 / 16 with kind::tf32, smem layout 1 | `tcgen05.mma.kind::tf32` MN-major A / B (SWIZZLE_128B_BASE32B) | 4-row 128-byte atoms, 32-byte granule swizzle | probes/tc_tf32_mn |
| `UTCHMMA ... !UPT` / `UPT` | `tcgen05.mma ..., idesc, 0` / `1` (immediate enable-input-d) | constant predicate: overwrite / accumulate | probes/tc_mma_imm |
| `PRMT` + `BMSK` + `SHF.L` + `LOP3 0xe2`; `FLO.U32` + `IADD3 -d, 0x1f` | `bfi.b32`; `clz.b32` | pos / len mod 256, clipped at bit 31; 32 for 0 | probes/bfi_clz |
| `FADD.FTZ.SAT d, RZ, a`, `FADD.FTZ d, a, -RZ` | `cvt.ftz.sat.f32.f32`, `cvt.ftz.f32.f32` | as the SASS ops | probes/cvt_ftz |
| immediates | decimal float literals (`20.`, `1.0`, `.5e1`) and `0d` in `.f32` / `.f64` instructions | f64 value rounded to nearest even for .f32; `0f` raw, zero-extended in .f64 | probes/float_lit |
| `FSETP.GEU \|a\|, +INF` + `IADD 0x1000` + `LOP3 & 0xffffe000` | `cvt.rna.tf32.f32` | ties-away round to 19 bits; Inf / NaN only masked | probes/cvt_tf32 |
| `F2FP [.SATFINITE] [.RELU] .TF32.F32.PACK_B [.RZ]` | `cvt.rn` / `.rz [.relu] [.satfinite] .tf32.f32` | round off 13 bits (RN even / RZ); NaN -> 0x7fffe000; relu: negative -> +0; satfinite: Inf / overflow -> +-0x7f7fe000 | probes/cvt_tf32_rn |
| (`CCTL.E.RML2`, refused) | `discard.global.L2 [a], 128` | weak stores of undefined data over the line (S-INIT) | probes/discard_l2 (PTX front) |
| `IMMA.16832.S8.S8.SAT` | `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32` | exact integer sum, saturated to s32 (N-IMMA) | probes/imma_s8 |
| `HMUL2.BF16_V2` | plain `mul.bf16` / `.bf16x2` reaching only stores, conversions, mma operands, multiplicands through packs / unpacks / prmt / shfl | .rn (ptxas cannot contract) | probes/bf16_sink |
| `FMUL2` + `FADD2` apart | plain `mul.f32x2` halves repacked next to values not from the product (own lane or other); integer reads of the product (`shr`, `xor`, ...) | .rn (ptxas cannot contract; `{p.lo, p.hi}` refused) | probes/f32x2_repack |
| `HMUL2.BF16_V2`, `SEL`, `STSM` / bare `STTM`, `LDTM` (32x32b.x1) | plain `mul.bf16` through `selp` into `stmatrix` / `tcgen05.st`; `tcgen05.{st,ld}.32x32b.x1` | .rn (ptxas cannot contract); x1 TMEM access | probes/bf16_selp |
| `REDUX` / `CREDUX` | `redux.sync` with a partial immediate or register membermask | reduction over the lanes the mask names; executing lanes != mask is a violation (PTX) | probes/redux_mask |
| `SHFL` / `VOTE` | `shfl.sync` / `vote.sync` with a partial immediate or register membermask | per mask group, combined with the group's lanes at another instruction of the same kind; executing lanes != mask is a violation (PTX) | probes/coll_split, probes/shfl_mask (PTX front; SASS `WARPSYNC.EXCLUSIVE` refused) |
| `FFMA2` / `FMUL2` / `FADD2 .RZ/.RM/.RP` | `.rz/.rm/.rp.f32x2` | one rounding in the mode | probes/bf16_ext |
| `FFMA2` / `FMUL2` / `FADD2 .FTZ[.RZ/.RM/.RP]` | `.{rn,rz,rm,rp}.ftz.f32x2` (a `.ftz` mul read by a same-mode add: refused) | subnormal inputs and results flushed per lane | probes/f32x2_ftz (GB200) |
| `IMAD.WIDE R, a, b, -c` | `mul.wide` + `sub.s64` | a*b - c mod 2^64 | probes/int_misc |
| `POPC` | `popc.b32` | variable latency, in-order queue (scoreboard cover) | probes/int_misc |
| `POPC` x 2 + `IADD3` | `popc.b64` | popc of each half, summed | probes/popc64 |
| `UP2UR` | (CuTe predicate spill) | uniform P2R | lift_l_forms |
| `VIADD.16x2` | `add.u16x2` / `add.s16x2` | two 16-bit lanes mod 2^16, no carry between lanes | probes/viadd16x2 |
| `LDG.E.ENL2.256 Rhi, Rlo` / `STG.E.ENL2.256 [a], Rlo, Rhi` | `ld` / `st.global.v8.b32` | 32-byte access, 32-byte aligned; bytes 0..16 in the `lo` quad (second operand of the load, first data operand of the store) | probes/ldst256 |
| `LDG.E.ENL2.256 Rhi, Rlo, [a], mask` | `ld.global.v8.b32` / `.v4.b64` with some elements unused | each half's selected words compacted from its base register (RZ: none); others untouched | probes/ldg256_mask |
| `LDG` / `STG .E.ELL2.256`, `.E.EFL2.256` | `ld` / `st.global.L2::evict_last` / `L2::evict_first .v8.b32` | as ENL2 (eviction hint dropped, L-PTX-HINT) | probes/ldst256_l2 |
| `S2UR` / `S2R` / `CS2R.32` of `SR_CLOCKLO` / `SR_CLOCKHI` / `SR_GLOBALTIMERLO` / `SR_GLOBALTIMERHI` | `%clock`, `%clock_hi`, `%globaltimer_lo` / `_hi` | 0, never an ordering witness (E-CLOCK) | probes/clock32 |
| `CS2R Rd, SR_CLOCKLO` / `SR_GLOBALTIMERLO`, `CALL.ABS.NOINC` vprintf, `c[0x0][0x2f8]` | `%clock64` / `%globaltimer`, `call.uni vprintf`, `cvta.local.u64` | clocks 0, vprintf a no-op (note per site), unmapped generic local base (E-CLOCK, S-CBANK) | probes/printf_clock |
| `IADD3` / `IMAD.IADD` with the window base (folded) | `cvta.to.local.u64` | generic minus the local generic base; outside the window, ld / st `.local` fault (S-CBANK) | probes/local_cvta |
| (moves) | `mov.b128 q, {lo, hi}` / `mov.b128 {lo, hi}, q` | a .b128 register is two 64-bit slots (a cluster-launch-control response for `query_cancel`) | fixtures/r1d_clc v2pack |
| `MOV` / `LOP3` / `SHF` of the response words | `clusterlaunchcontrol.query_cancel.get_first_ctaid::z` / `.v4` | word 1 >> 16 (ctaid.z); .v4 = {x, y, z, 0} (E-CLC) | fixtures/r1d_clc3d |
| `LDTM` (consecutive registers) | `tcgen05.ld` with destinations not consecutive (CuTe DSL lists them in descending order) | register j of the load into the j-th listed register | probes/sttm_ldtm permuted |
| `LDC.64 c[0x4][..]` (`R_CUDA_64` to `.nv.global[.init]`) | `.global` module variables, `mov.b64 %rd, sym`, `[sym+off]` | one buffer per (module, symbol, GPU): captured bytes, else initializer, else zeros (noted) (S-GLOBAL-INIT) | probes/ptx_globals |
| `MEMBAR.SC.<scope>` [+ `ERRBAR` + `CCTL.IVALL`] | `membar.cta` / `.gl` / `.sys` | as `fence.sc.<scope>` (S-SCOPE) | probes/fence_scopes |
| nothing / `CCTL.IVALL` | `fence.acquire.cta` / `.cluster` / `.gpu` | acquire of that scope only (S-SCOPE) | probes/fence_scopes |
| `FMUL [.FTZ] a, RN(1/C)` (or FFMA into an add), `FADD +-a, -RZ` for C = +-1 | `div.full` / `div.approx [.ftz] .f32 a, C`, C an immediate or a dominating `mov` constant | ptxas's constant fold, then PTX-FMAD as a plain mul | probes/div_const |
| `FFMA` / `FFMA.FTZ` / `FFMA.SAT` / `DFMA` | `mul` + `add` / `sub` `.f32` / `.f64` without rounding modifiers | ptxas 13.0's default contraction where unambiguous (one block, same `.ftz`, unguarded); ambiguous shapes refused (PTX-FMAD) | probes/fmad |
| `FFMA` per reader, or `FMUL` + `FADD` | plain `mul` `.f32` / `.f64` with several readers (adds / subs, through `mov` / `neg`, others) | all or nothing: contracted into every reader when all live readers are same-block adds of the same `.ftz`, else none; an add of two products folds the one with no other reader (PTX-FMAD) | probes/fmad_multi |
| `HFMA2` per reader (result in the low half), or `HMUL2` + `HADD2` | plain scalar `mul` + `add` / `sub` `.f16` | the `.f32` rule (all or nothing per mul); a copy / `neg` of the product read by an add, and an add of two products, refused (PTX-FMAD) | probes/fmad_f16 |
| `HFMA2.BF16_V2 d, -RZ.H0_H0, RZ.H0_H0, \|R\|[.H1_H1]` | `abs.bf16x2` / `abs.bf16` | sign bits cleared, then -0*0 + \|x\|: NaN 0x7fff, -0 to +0 | probes/bf16_abs |
| (PTX) | `ld[.param / .global].s32` / `.s16` / `.s8` into a 64-bit register | sign-extended to 64 bits (unsigned: zero) | probes/ld_sext64 |
| (PTX) | `cvt.s8` / `.s16 .s32` / `.s64` narrowing | the kept bits sign-extended to the destination register | probes/cvt_s8_narrow |
| `HADD2 d, \|R\|, -RZ.H0_H0` / `HADD2 d, -R, -RZ.H0_H0` | `abs` / `neg .f16[x2]` | per lane \|x\| or -x plus -0; NaN -> 0x7fff | probes/f16_absneg |
| `HFMA2` / `HMUL2` / `HADD2` (f16) with `.H0_H0` / `.H1_H1`, `.reuse`, immediate pairs | `fma` / `mul` / `add` / `sub .rn.f16[x2]`, `mul.f16` that cannot be contracted | per lane one f16 rounding; HMUL2 = a*b + -0, HADD2 = a*1 + c | probes/f16_arith |
| `LDSM.16.M88.N` / `.MT88.N` | `ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16` | (PTX front added) | probes/ldmatrix_ptx |
| `LDG.E.NA[...]`, `ST.E.EF.128`, `LD` / `ST` / `ATOM` `.STRONG.SM` | `.L1::no_allocate`, `.L1::evict_first`, `.cta` scope | cache hints dropped; `.SM` = CTA scope | probes/memhints |
| `STG.E.64.MMIO.SYS` | `st.mmio.relaxed.sys` | strong relaxed `.sys` store; other `.MMIO` forms refused | probes/memhints |
| `UBLKPF.L2`, `R2UR.OR P, UR, R` | `cp.async.bulk.prefetch.L2` | no-op; `.OR` ORs the lane's "differs" bit into P | probes/memhints |
| `FLO` scoreboards | - | FLO shares POPC's in-order queue (L-OPCLASS) | probes/flo_popc |
| `F2F.F32.F64` / `F2F.F64.F32`, `FADD.SAT d, RZ, a`, `F2I...NTZ` | `cvt.rn.f32.f64` / `cvt.f64.f32`, `cvt.sat.f32.f32`, `cvt.r{n,m,p,z}i[.ftz].{s,u}{32,64}.f32` (PTX front added) | as the SASS ops | probes/cvt_misc |
| `FADD[.FTZ] d, \|a\|, -RZ`, `IABS` | `abs[.ftz].f32`, `abs.s32` | \|a\| + -0 (canonical NaN); two's-complement abs | probes/abs_f32 |
| `S2R SR_LTMASK` etc. | `%lanemask_{lt,le,ge,gt,eq}`; `%`-less scope registers; `[%rd+-N]`; `{...}` operands continued after a comma | PTX syntax only | probes/ptx_syntax |
| `FFMA2[.RZ/.RM/.RP]` | `mul.f32x2` read only by `add` / `sub.f32x2` of its rounding mode (none = `.rn`), explicit modifiers included | contracted into each (PTX-FMAD); any other reader: none | probes/fmad2 |
| `FMUL2` | plain `mul.f32x2` no `add` / `sub.f32x2` can read (selp, fma addend, copies, multiplicands, stores, unpacked halves not repacked; loop-carried too) | .rn (ptxas cannot contract; PTX-FMAD) | probes/f32x2_sinks |
| `FMUL2` / `FADD2` | `mul` / `add` / `sub.f32x2` with no rounding modifier | .rn where no use lets ptxas contract (a mul only multiplied, stored, converted, an fma addend, or unpacked into halves read by scalar ops or repacked across lanes, also next to a direct add / sub reader; an add / sub of no such mul) | probes/f32x2_norn, probes/f32x2_unpack, probes/f32x2_unpack_add |
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
| `ELECT P, UR, PT` (ptxas drops a register membermask) | `elect.sync d\|p, %r` (mask in a register) | full warp checked when executed; partial refused (E-ELECT) | probes/elect_regmask |
| `UBLKCP.S.G` / `UBLKCP.G.S` with the policy in `desc[UR]` | `cp.async.bulk.{shared::cluster.global.mbarrier::complete_tx::bytes,global.shared::cta.bulk_group}.L2::cache_hint` | the copy; hint and policy operand dropped (L-PTX-HINT) | probes/bulk_hint |
| `CALL.ABS.NOINC` (`__assertfail`), `BPT.TRAP` | `call.uni __assertfail, (param0, .., param4)` | trap: the grid is aborted (E-CLOCK) | probes/assertfail |
| `F2FP.SATFINITE.BF16.S2_6.UNPACK_B Rd, Ra[.H1], Rb.H0 / imm.H0` (nvdisasm 13.0 prints nothing; decoded by `undis.rs`, L-NVDISASM) | `cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2` (PTX ISA 9.1, ptxas 13.1) | per lane: int8(code byte i) / 64 * 2^(scale byte i - 127), one RNE rounding to bf16, subnormals kept, satfinite; scale 0xff -> 0x7fff (N-CVT-LP) | **probes/s2f6_cvt**: every (code, scale) pair of both lanes, register / immediate scale, either data half, SASS and PTX bitwise; all 2^32 (data, scale) pairs vs. a C model on GB300 (`run.sh`), 0 mismatches |
| `REDG.E.{OR,AND,XOR}[.64].STRONG.{SM,GPU,SYS}` | `red.global[.cta / .sys].{or,and,xor}.b32 / .b64` | one atomic bit operation per thread on the word (S-SCOPE: `.SM` = CTA) | **probes/red_bitops**: 256 threads, four per word, single-bit / all-ones / zero / random words, both front-ends bitwise |
| `UTCHMMA[.2CTA] ..., idesc[URi], UPd, s` (trailing immediate) | `tcgen05.mma.kind::f16 / kind::tf32 ..., enable_input_d, scale_input_d` | D = A B + D * 2^-s: the calibrated step with the accumulator scaled exactly; a scaled value that is not normal or zero is `unmodeled` (N-MMA) | **probes/mma_scale_d**: bf16 and tf32, 9 scales, repeated and with enable 0, SASS and PTX bitwise; 9 random input sets |
| `UTMALDG.<n>D.IM2COL[.2CTA] [URa], [URb], URo[, desc[..]]`, `UTMASTG.<n>D.IM2COL [URa], [URb]` | `cp.async.bulk.tensor.<n>d.shared::cluster.global.im2col...`, `....global.shared::cta.im2col_no_offs.bulk_group` | im2col walk through the pixel bounding box, unsigned packed offsets, zero fill / skipped stores; unaligned channel starts and stores through padded boxes fault (S-TMA, `sim/im2col.rs`); im2col tensor maps decoded (S-TMAP) | **probes/tma_im2col**: 48 loads / 12 stores through 9 encoder-built maps, SASS bitwise; fault conditions by single-access variants |
