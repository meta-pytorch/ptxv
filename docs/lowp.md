# Low precision: FP8 / FP6 / FP4 conversions and tensor-core MMAs

What ptxv models for narrow formats on GB300 (sm_103a, CUDA 13.0), and how each part was
validated. The details and evidence are in ASSUMPTIONS.md entries N-CVT-LP, N-MMA-F8,
N-MMA-BS, N-MMA-K96 and S-TCCP.

| piece | model | validation |
|---|---|---|
| `cvt` to / from e4m3, e5m2, e2m3, e3m2, e2m1(x2/x4), ue8m0; `.satfinite`, `.relu`, `.rs` packs (SASS `F2FP`) | `lowp.rs`: exact rounding from the input value; the bit routing of the `.rs` random word measured | `ptxv cvt-sweep probes/cvt_lp`: 51 forms; every 16-bit input / code pair, plus 2^20 f32 operand sets per form; 0 mismatches. Per-op hardware hashes are replayed in `cargo test` |
| `tcgen05.mma.kind::f8f6f4` (SASS `UTCQMMA`) | `g32-f25-rz-accin-tzero-efields`: 32 products and the accumulator are aligned to the largest exponent-field sum, truncated 25 bits below it, summed exactly and rounded toward zero (an f16 D rounds to nearest) | `ptxv calib-mma probes/mma_f8`: 3,624 candidates; 9 format / D configurations x 10 adversarial input sets x 8,192 outputs, each 4 chained steps; the fit is unique where the data can separate the candidates |
| block-scaled `mxf8f6f4` (`UTCQMMA` with scale operand) | the f8f6f4 model on a·b·sa·sb terms | `probes/mma_bs`: 4 format pairs, unique fit |
| block-scaled `mxf4`, `mxf4nvf4` (`UTCOMMA`, `.BLOCK16`) | `u16-f35-live-sfields`: exact sums over 16-element units, each scaled; aligned to the largest scale-exponent sum over units with a nonzero product; truncated 35 bits below; rounded toward zero | `probes/mma_bs`: 2X ue8m0, 4X ue8m0, 4X ue4m3; unique within the family (360 candidates) |
| `mxf4nvf4` block16 with K = 96 (idesc bit 31, sm_103; `UTCOMMA.BLOCK16`, `.2CTA`) | the same `u16-f35-live-sfields` in one 96-wide step (six units of 16): not two steps of 64 + 32. Operands: 48 bytes per row (three 16-byte chunks LBO apart without swizzle; contiguous within the 128-byte row with SWIZZLE_128B, or split at the row end with the absolute-address LBO mode, descriptor bit 52) | `ptxv calib-mma probes/mma_k96/calib`: ue8m0 and ue4m3, 19 sets x 8,192 outputs, 4 chained steps; `u16-f35-live-sfields` is the only member matching both; the best 64 + 32 split matches 1,825 to 8,192 per set. Layouts: 22 exact identification sets (`probes/mma_k96`), both front-ends = GB300 |
| K = 96 scale factors | block b reads byte p = sf_id + b (sf_id 0 or 2) of the row's two words: p < 4 in column base + i, else byte p - 4 in column base + g + i (i = sub-partition for A, n / 32 for B; g = 4 for A, max(4, N / 32) for B) | per-(lane, column, byte) distinct ue4m3 scales with exact sums, N = 64 / 128 / 256, sf_id 0 and 2 |
| scale factors in TMEM | SFA row m: lane m, column + m/32. SFB row n: lane 32w + n%32, column + n/32. Byte selected by the idesc sf_id; address bits 30-31 ignored | measured with per-lane / per-column distinct scales; illegal-instruction cases recorded |
| `tcgen05.cp.32x128b.warpx4` (`UTCCP.T.S.4x32dp128bit`) | 32 x 16 B smem rows to 4 columns, replicated to the 4 sub-partitions | end to end in `fixtures/r3_cute_mxf8` |
| `cta_group::2` block-scaled MMA (`UTCOMMA.2CTA.BLOCK16`) and `tcgen05.cp.cta_group::2` (`UTCCP.T.S.2CTA.4x32dp128bit`) | the one-CTA arithmetic; A / B split as E-PAIR; each CTA's D rows use the scale factors in its own TMEM (each CTA holds all of SFB); the pair copy moves each CTA's smem into its own TMEM | `probes/tc_pair_bs` (every operand differs between the CTAs; both front-ends = GB300); cuBLASLt's NVFP4 kernel end to end (below) |
| `tcgen05.ld.16x128b` (`LDTM.16dp128bit`), `LOP3.LUT.PAND` | register 2j + h of lane l: TMEM lane l/4 + 8h, column 4j + l%4; Pd = (r != 0) AND Pq (default form: OR Pq) | `probes/tmem_16x128b`; `probes/lop3_pmode` (patched cubins) |
| error contracts of these MMA steps (numerics, equivalence) | derived from the models: e4m3 / e5m2 steps 17.5u to 130u, FP4 kinds ~2u (`mma_lp::lp_eps`, `fp4_eps`; docs/equiv-mma.md §5); FP8 operand and ue8m0 scale bytes are leaves | 60,000 adversarial steps per contract test, the model's error never above it, reached within 0.9; the MXF8 GEMM's own bound 255.5u x sum \|terms\| for every input (`fixtures/c_equiv_mxf8`) |

## Real kernel: `fixtures/r3_cute_mxf8`

The CUTLASS 4.6.1 CuTeDSL example `dense_blockscaled_gemm_persistent.py` (flashinfer's bundled
copy). It is an MXF8 GEMM: A, B e4m3, ue8m0 scales per 32 K elements, C f32, M = N = 128,
K = 256, one persistent CTA with 6 warps. Its pipeline is TMA (A, B, SFA, SFB), then
`tcgen05.cp`, then `tcgen05.mma...block_scale.block32`, then the TMEM epilogue and a TMA
store. `capture.py` compiles it, records the launch with probes/launchcap, matches the five
tensor maps and writes the case; `variants.py` adds the mutant.

| variant | ptxv (PTX / SASS) | spec (exact MXF8 GEMM) | GB300 vs simulator |
|---|---|---|---|
| good | race + violation / clean | dataflow = spec | bitwise equal on all 16,384 outputs, both front-ends; 1,000 / 1,000 runs |
| B30: K-block 0's MMA uses K-block 1's scales (sf_id) | race + violation / clean | **fails** | bitwise equal (the mutant's wrong output is reproduced exactly) |

The PTX verdict for `good` matches r2_cute_small. As compiled, CuTeDSL's PTX has no
`tcgen05.wait::ld`, every lane issues the TMA store, and there are no tcgen05 fences around
the deallocation. ptxas compiles all of this safely, so the SASS is clean. B30 is a numerics
bug: execution is unchanged, and the exact-spec claim (`spec = "gemm_tn_mx"`) catches it.

B31 (fp8 accumulation without fp32 promotion) does not apply to this kernel. Block-scaled
tcgen05 MMAs accumulate in f32 in TMEM: the per-step truncation is part of the hardware
model, and no software promotion step exists that could be omitted. B31 needs a
DeepSeek-style kernel that accumulates into a narrow register accumulator and promotes it
periodically. That is not captured here. flashinfer's trtllm-gen dsFp8 gemm1 (docs/rung3b.md) does promote in
software: it scales each 128-K block's TMEM partial sum in f32 registers. But it promotes
once per scale block, and its MMA accumulates in f32 TMEM, so B31 does not fit there either.
B30 is caught there by the exact-reference numerics claim.

## Other implementations (docs/equiv-mma.md §5)

Each reads the same input bytes as its CuTeDSL reference; `ptxv equiv` compares them.

| fixture | implementation | kernel | vs CuTeDSL |
|---|---|---|---|
| `c_smm_mxf8` | torch._scaled_mm (cuBLASLt), MXF8 | nvjet_sm103_qqsss_128x128_128x6 (binary) | L0 up to the sign of zero |
| `c_triton_mxf8`, `_rev` | Triton tl.dot_scaled e4m3, K blocks of 128 / 32 reversed | tcgen05 mxf8f6f4 1X | L0 / L1, 306u x sum \|terms\| |
| `r3_cute_mxf4` | the CuTeDSL GEMM for MXF4 (e2m1, ue8m0 per 32) | 4-bit tensor maps, kind::mxf4nvf4 block32 | (reference) |
| `c_triton_mxf4`, `_rev` | Triton tl.dot_scaled e2m1, K blocks of 256 / 64 reversed | kind::mxf4 scale_vec::2X | L0 / L1, 10.04u x sum \|terms\| |
| `r3_cute_nvf4` | the CuTeDSL GEMM for NVFP4 (e2m1, ue4m3 per 16) | kind::mxf4nvf4 block16, one CTA | (reference) |
| `c_smm_nvf4` | torch._scaled_mm (cuBLASLt), NVFP4 | nvjet_sm103_oosss_128x128_256x7_1x2_2cta (binary, a 2-CTA pair) | L0 up to the sign of zero |

cuBLASLt's NVFP4 kernels at M = N = 128..256 are all `2cta`. Lifting them needed the pair
block-scaled MMA, the pair `UTCCP`, `LDTM.16dp128bit` and `LOP3.LUT.PAND` (rows above). Triton
3.5's tl.dot_scaled takes ue8m0 scales only, so it has no NVFP4 form. The global f32 scales
(torch.nn.functional.scaled_mm, BlockWise1x16 + TensorWise) do not reach the MMA: torch
multiplies GA * GB in an elementwise kernel (f32, round to nearest) and cuBLASLt applies the
product as alpha in its epilogue, out = fl(fl(GA GB) * acc) (GB300: 3,276,800 / 3,276,800
outputs; fl(fl(acc GA) GB) matches 64%; `tools/nvf4_global_scale_check.py`). The alpha kernel (`_bx`) reads alpha through a device
pointer and branches on alpha == 0, a data-dependent branch; it is not captured as a case.

## Not modeled

- MN-major FP6 / FP4 operands for the narrow kinds (8-bit MN-major is modeled: probes/mma_bs/mn);
  M = 64; sparsity.
- K = 96 other than kind::mxf4nvf4 scale_vec::4X; with N other than 64 / 128 / 256 (and other
  than 256 for cta_group::2); odd scale-factor ids.
- Tensor maps of the sub-byte types other than `16U4_ALIGN8B` (`elem4`), and device-built 4-bit
  maps.
- NaN / Inf operands or scales, subnormal f32 accumulators, f16 overflow. These are reported
  as outside the model.
- `tcgen05.cp` shapes other than 32x128b.warpx4, and its decompression forms.
- cvt forms not in the probe: f16 / bf16 `.rn` with `.relu` / `.satfinite`, and some SASS
  F2FP variants.
