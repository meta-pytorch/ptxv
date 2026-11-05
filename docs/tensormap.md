# CUtensorMap binary layout (GB300, driver 580 / CUDA 13.0)

The 128-byte `CUtensorMap` is opaque in the CUDA headers. This layout was recovered by
calling `cuTensorMapEncodeTiled` (plus a few `cuTensorMapEncodeIm2col` /
`cuTensorMapEncodeIm2colWide` calls) through `probes/tmap_encode/tmap_encode.c` and diffing
the output bytes. It is decoded by `crates/sim/src/tmap.rs` (`decode`).

Evidence levels:

- **measured**: single-field sweeps, then confirmed against a 3097-sample random sweep
  (`probes/tmap_encode/sweep.py`, seeds 1 and 2: all dtypes 0..12, rank 1..5, interleave
  none/16B/32B, all 7 swizzles, all L2 promotions, both OOB fills, dims up to 2^32, strides up
  to 2^40, element strides 1..8). Every field below that is "measured" matched all samples;
  no bit outside the listed fields was ever set. `samples.txt` (every 8th sample) is
  checked by the `encoder_sweep` unit test.
- **inferred**: deduced from SASS / library-built descriptors, not from the encoder.
- **unobserved**: never seen set; the decoder rejects it.

Little-endian throughout. "word X" = the u32 at byte offset X.

| bytes | bits | field | encoding | evidence |
|---|---|---|---|---|
| 0x00..0x07 | 0..56 | global address | raw byte address; 16-byte aligned (the encoder rejects anything else) | measured; addresses up to 2^57-16 accepted, 2^57 and above rejected by the encoder |
| 0x00..0x07 | 57..63 | - | always 0 | unobserved |
| word 0x08 | 0 | im2col mode | 1 for `cuTensorMapEncodeIm2col` and `...Im2colWide` (both), 0 tiled | measured (3 im2col samples) |
| word 0x08 | 1 | "library flag" | never set by the driver; set in every descriptor built by cuBLAS nvjet (r2_cublas_gemm) and CuTeDSL (r2_cute_*). Meaning unknown | inferred; see below |
| word 0x08 | 2..3 | - | 0 | unobserved |
| word 0x08 | 4..6 | rank - 1 | | measured |
| word 0x08 | 7..10 | element type | hardware code, NOT the `CUtensorMapDataType` order: u8 0, u16 1, u32 2, s32 3, u64 4, s64 5, f16 6, f32 7, f32_ftz 8, f64 9, bf16 10, 16U4_ALIGN8B 11, 16U4_ALIGN16B 12, 16U6_ALIGN16B 13 (= PTX `tensormap.replace .elemtype` numbering for 0..10); tf32 = 7 + bit 16, tf32_ftz = 8 + bit 16 | measured |
| word 0x08 | 11..12 | interleave | 0 none, 1 16B, 2 32B | measured |
| word 0x08 | 13..14 | swizzle span | 0 none, 1 32B, 2 64B, 3 128B (also 3 for every `128B_ATOM_*`) | measured |
| word 0x08 | 15 | OOB fill | 1 = `NAN_REQUEST_ZERO_FMA` | measured |
| word 0x08 | 16 | tf32 | with code 7/8 selects tf32 / tf32_ftz | measured |
| word 0x08 | 17..18 | L2 promotion | 0 none, 1 64B, 2 128B, 3 256B | measured |
| word 0x08 | 19..20 | swizzle atom | 0 16B (classic), 1 `128B_ATOM_32B`, 2 `128B_ATOM_32B_FLIP_8B`, 3 `128B_ATOM_64B` | measured |
| word 0x08 | 21 | "large" hint | set iff `prod(dims) * B >= 128 KiB`, B = element size (or 16 / 32 with interleave 16B / 32B), product wrapping mod 2^64 | measured (rule fits 3097/3097); meaning unknown. CUTLASS's device-side updates never refresh it, so the decoder only reports it |
| word 0x08 | 22..31 | - | 0 | unobserved |
| words 0x0c, 0x10, 0x14, 0x18 | all | stride of dim 1, 2, 3, 4 (low part) | `(stride_bytes >> 4) & 0xffffffff` | measured |
| word 0x1c | 4(i-1)..4(i-1)+3 | stride of dim i (high part), i = 1..4 | `stride_bytes >> 36` (strides are < 2^40 and multiples of 16) | measured |
| word 0x1c | 16..31 | - | 0 | unobserved |
| words 0x20..0x30 | all | dim i, i = 0..4 | `dim - 1` (u32; dims 1..2^32) | measured |
| word 0x34 | 3i..3i+2 | element stride i, i = 0..4 | `estride - 1` (1..8) | measured |
| word 0x34 | 15..23 | - | 0 | unobserved |
| byte 0x37 | | box dim 0 | `box - 1` (1..256) | measured |
| bytes 0x38..0x3b | | box dims 1..4 | `box - 1` each | measured |
| bytes 0x3c..0x3f | | - (tiled) | 0 for tiled maps; im2col maps put the pixel box corners / pixels per column around 0x38..0x3c | measured (tiled); im2col not decoded |
| 0x40..0x47 | | box bytes (derived) | `B * prod_i floor(box_i / estride_i)` (floor, even where the transfer itself rounds up; B as above; 0 for the sub-byte types). Im2col: channels * pixels * esize | measured |
| 0x48..0x4f | | swizzle repeat (derived) | 16 for no swizzle, else 8 * span (256 / 512 / 1024) | measured |
| 0x50..0x7f | | - | 0 | unobserved |

Slots of dims, strides, boxes and element strides at or beyond `rank` are 0 in driver
output. The decoder ignores them: CUTLASS's device-side update (below) writes non-zero junk
into the stride slots of dims 2..4 of its rank-2 maps, and cuBLAS leaves `0x10000000` in the
dim-3 stride slot of its rank-3 maps, and both kernels run correctly on hardware.

## Library-built (bit 1) descriptors

cuBLAS and CuTeDSL build descriptors themselves. Bytes 0..63 follow the table above (the
r2 fixtures' captures and the CuTeDSL `capture.py` matching against the driver confirm it
for address, rank, dtype, swizzle, L2, dims, strides, boxes), but bit 1 of word 0x08 is set and
bytes 64..127 are not the derived words: cuBLAS packs consecutive descriptors 64 bytes apart
(r2_cublas_gemm params 192/256 and 1216/1280), so bytes 64..127 of one map are the next map.
Since that kernel runs correctly, the hardware evidently needs only bytes 0..63 of a bit-1
descriptor (inferred). Whether it needs words 0x40/0x48 of a driver-built map is unknown;
the decoder checks them for consistency and errors if they are stale.

The extent-1 dim 2 of the cuBLAS maps has stride 0 in the descriptor (the r2 case's declared
strides 32768 / 131072 / 262144 for that dim were synthesized by capture.py).

## Not covered

- Interleaved layouts and the sub-byte types: encoding measured, semantics of the transfer
  not modeled; the decoder rejects them.
- Im2col / im2col-wide: only the mode bit and box-bytes rule were looked at; rejected.
- The meaning of word 0x08 bits 1 and 21 (see above).
- Driver versions other than 580 / CUDA 13.0, and GPUs other than GB300.

## Rung 3a: device-side updates (CUTLASS grouped GEMM)

The 2304-byte Params block of fixtures/r3_grouped_mm holds five driver-encoded maps, each
byte-identical to `cuTensorMapEncodeTiled` with address 0 and stride 0:

| param offset | c-bank | role | dims | box | swizzle | other |
|---|---|---|---|---|---|---|
| 128 | c[0x0][0x400] | A template (mainloop) | 1 x 1 | 64 x 128 | 128B | bf16, rank 2, L2 128B |
| 384 | c[0x0][0x500] | B template (mainloop) | 1 x 1 | 64 x 128 | 128B | same |
| 640, 896 | not referenced by this kernel | same bytes as 128 | | | | |
| 1664 | c[0x0][0xa00] | D template (epilogue TMA store) | 256 x 128 | 32 x 128 | 64B | bf16, rank 2, L2 128B |

The kernel (entry `..._GemmUniversal..._Sm100ArrayTmaUmmaWarpSpecialized...`) copies a
template into shared memory (STS.128 of the 16 c-bank qwords: A to smem 0x480, B to 0x500
in the producer warp at 0x0190; D at 0x400 of the epilogue warp's window at 0xaaa0), patches
the copy per group with plain STS (the same pattern appears in two code paths for A/B, around
0x3610 and 0x9450, and twice for the epilogue, around 0x9810 / 0x9b50 and 0xb040), publishes
the 128 bytes to a per-CTA slot of the global workspace, then fences:

Patch (byte offsets within the 128-byte descriptor; g = group, from W0 written by
`prepare_grouped_gemm_data`: problem shape (M, N, K) as 3 x i32 at c[0x0][0x390] + 12g,
pointer and stride arrays as i64 per group):

| offset | written value | instruction |
|---|---|---|
| 0x00 (u64) | `ptr[g]` (A: c[0x828], B: c[0x838], D: array at c[0x888]/[0x890]) | `STS.64` |
| 0x0c (u32) | `(stride_elems[g] * 2) >> 4` low 32 bits (A: c[0x830], B: c[0x840]; D: stride array if non-null, else N) | `STS` |
| 0x10, 0x14 (u32) | junk: the descriptor's own smem address >> 4 (stride slots of dims 2, 3; unused at rank 2) | `STS` |
| 0x18 (u32) | same junk (dim-4 stride slot) | `STS.64 [0x18]` (its high half is the word 0x1c read-modify-write) |
| 0x1c (u32) | nibble 0 = `(stride_bytes >> 36) & 0xf`; nibbles 1, 2, 3 cleared; read-modify-write per nibble (`LDS` + `LOP3 ... 0xb8` + `STS`) | `STS` |
| 0x20..0x2f | `STS.128` of `{dim0 - 1, dim1 - 1, 0, 0}`: A `{K-1, M-1}`, B `{K-1, N-1}`, D `{N-1, M-1}` | `STS.128` |
| 0x30 (u32) | 0 (dim 4 = 1) | `STS RZ` |

Nothing else is rewritten: rank, dtype, swizzle, box, element strides, L2, bit 21 and the
derived words 0x40/0x48 keep their template values (consistent with tensormap.replace only
touching global_address, global_dim and global_stride).

Publish (`tensormap.cp_fenceproxy.global.shared::cta...`): each of 32 lanes does
`LDS R, [desc + 4*lane]` and `ATOMG.E.EXCH.STRONG.GPU [gdesc + 4*lane], R` (A and B in one
loop at smem 0x480 / 0x500; global slots `c[0x820] + 0x80 * slot`, c[0x820] = param 1184 =
W0 + 272896), preceded by `UTMACMDFLUSH; DEPBAR.LE SB0, 0x0` on the elected lane.

Acquire before use (`fence.proxy.tensormap::generic.acquire.gpu`): `DEPBAR {5,4,3,2,1,0};
CCTL.E.C.LDCU.IV.DEEP [gdesc]; UTMACCTL.IV [gdesc]; UTMACMDFLUSH`, then
`UTMALDG.2D ... desc[...]` / `UTMASTG.2D` with the global descriptor address in a uniform
register pair.

For the sim this means: a TMA's descriptor is the 128 bytes at a global address, written by
ordinary (atomic) stores; decode it at UTMALDG/UTMASTG time with `tmap::decode`, after
modeling the generic-to-tensormap-proxy fence (CCTL...IV.DEEP + UTMACCTL.IV) as making those
stores visible to the TMA unit.
