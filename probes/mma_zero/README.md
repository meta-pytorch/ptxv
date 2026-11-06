# mma_zero: the sign of a zero MMA result (N-MMA, N-HMMA, N-MMA-F8)

Question: ptxv gave -0 where GB300 gave +0 for a bf16 MMA with C disabled whose
only nonzero product, -40762 * 2^-178, underflows f32. Does that hold on supported shapes, and
what is the hardware's rule for zero results?

Files:
- `zero.cu` -> `kernel.ptx` (`nvcc -ptx -arch=sm_103a -O3 -o kernel.ptx zero.cu`). Entries:
  - `tc_f16` / `tc_f8` / `tc_tf32`: one `tcgen05.mma.cta_group::1` (kind::f16 / f8f6f4 / tf32)
    per CTA, with M128 or M64, N16. C goes into TMEM with `tcgen05.st`, and `en` is
    enable_input_d.
  - `hm` / `hm0` / `hmh`: `mma.sync.m16n8k16`, bf16 (hmh: f16) -> f32. C comes from memory;
    in `hm0` it is the constant 0, which is `HMMA ... RZ` in SASS.
  - `tm` / `tm0`: `mma.sync.m16n8k8` tf32, C from memory or RZ.
- `run.py gen | run | report`:
  - `gen` writes the cases and inputs, `expected/<config>.{A,B,C}.bin` and
    `expected/configs.json`;
  - `run` runs each config once on GB300 (`ptxv probe`, `PTXV=<binary>`, `PTXV_GPUS=<idle gpu>`)
    and writes `expected/<config>.D.bin`;
  - `report` tabulates the signs.
- `crates/sim/tests/mma_zero.rs` replays every modeled config bit for bit: both front-ends for
  tcgen05, SASS for mma.sync (no PTX form).

Cases: each is one D element, with A row i and B row (or column) i of a tile. Products that a
case does not set are padded three ways: +0 x +0, +0 x -0 (all -0 products), and +0 x ±1
alternating. Each case runs with C = +0 and with C = -0 when C is enabled, and with C disabled
(tcgen05 `en` = 0, or the RZ variants). The families:
- **underflow**: the witness (raw bf16 0x16b2 x 0x96e5) and its negation. Single products
  2^-150, 2^-151, 2^-160, 2^-200, 2^-250 (x1 and x1.875, each sign, negated on either side).
  Cancellation residues ±2^-156, and two tiny products summing to ±2^-161.
- **tiny**: single products 2^-126 to 2^-149, and residues ±2^-132: normal or subnormal results.
- **zeros**: all products ±0 (±0 x ±1, all sign combinations; one -0 among +0, and the reverse).
- **cancel**: x y + (-x) y in both orders, for (1, 1), (0.5, 1.5), (2^-60, 2^-60) and
  (2^-100, 2^-20); and 1 + 1 - 2.
- **acc**: ±2^-160 with C = ±2^-126 or 1.

A case is skipped for a format that cannot represent it: fp8 and f16 products cannot
underflow f32. The off-diagonal elements of the tiles (cross products of different cases)
are also compared bitwise with the model.

## Result (GB300, sm_103a, CUDA 13.0, driver 580, GPU 3, 2026-11-05)

**Every zero result is +0.** This holds for every instruction, kind, shape and C mode below,
and for each zero class:

| exact result | C disabled | C = +0 | C = -0 | GB300 D |
|---|---|---|---|---|
| nonzero, \|sum\| < 2^-149, positive (one product, residue, two products) | +0 | +0 | +0 | +0 |
| nonzero, \|sum\| < 2^-149, negative (witness -40762 * 2^-178, ...) | +0 | +0 | +0 | +0 |
| all products +0 | +0 | +0 | +0 | +0 |
| all products -0 (any pad) | +0 | +0 | +0 | +0 |
| exact cancellation x y - x y (either order, any pad) | +0 | +0 | +0 | +0 |
| subnormal result (2^-127 .. 2^-149, residue 2^-132) | sign kept | sign kept | sign kept | ±subnormal, RZ |
| -2^-160 + C = 2^-126 | n/a | n/a | n/a | 2^-126 (the product truncates away) |

| config | instruction | D words | cases (zero) | model before fix: words != GB300 |
|---|---|---|---|---|
| tc_bf16_m128_en0 / en1 | tcgen05 kind::f16 bf16, M128 N16 | 65,536 / 67,584 | 510 / 516 (282) | 1,460 / 1,493 (all -0 vs +0) |
| tc_bf16_m64_en0 / en1 | tcgen05 kind::f16 bf16, M64 N16 | 65,536 / 67,584 | 510 / 516 (282) | 1,460 / 1,493 |
| tc_f16_m128_en0 / en1 | tcgen05 kind::f16 f16 | 6,144 | 42 (42) | (f16 inputs not modeled) |
| tc_tf32_m128_en0 / en1 | tcgen05 kind::tf32 | 65,536 / 67,584 | 510 / 516 (282) | (not modeled) |
| tc_e4m3_m128_en0 / en1 | tcgen05 kind::f8f6f4 e4m3, f32 D | 6,144 | 42 (42) | 504 / 2 |
| tc_e4m3f16d_m128_en0 / en1 | kind::f8f6f4 e4m3, f16 D | 6,144 | 42 (42) | 504 / 2 |
| tc_e2m1_m128_en0 / en1 | kind::f8f6f4 e2m1 | 4,096 | 30 (30) | 504 / 2 |
| hm_bf16_c / hm_bf16_rz | mma.sync m16n8k16 bf16, C regs / RZ | 8,320 / 4,096 | 516 / 255 (282 / 141) | 935 / 365 |
| hm_f16_c | mma.sync m16n8k16 f16 | 768 | 42 (42) | (not modeled) |
| tm_tf32_c / tm_tf32_rz | mma.sync m16n8k8 tf32, C regs / RZ | 8,320 / 4,096 | 516 / 255 (282 / 141) | 935 / 365 |

"cases (zero)" counts the diagonal cases, and in brackets those whose exact result is zero
or underflows. Before the fix, every differing word was ptxv -0 against GB300 +0, and all other
words were already bitwise equal (subnormal results included). After the fix, all configs
match bitwise. The witness is +0 on GB300 in every config that can represent it (tcgen05
bf16 M128 / M64, tf32, mma.sync bf16 / tf32), with each pad and C mode.

Not measured: the block-scaled kinds (scaled sums that underflow), and f16-D results that
underflow f16 (fp8 products cannot underflow it). The model applies the same rule to them,
through the same rounding code.
