# GB300 MMA overflow

This probe distinguishes saturation from infinity at the FP32 overflow boundary,
and checks whether cancellation happens before overflow. All operands and input
accumulators are finite normal values or signed zero.

GB300 returns signed infinity on overflow. The old Rust RZ conversion saturated
instead; Lean already matched the hardware. The fix changes the Rust MMA helper,
with no change to the Lean arithmetic or scalar IEEE rounding.

| Sum before alignment / rounding | Positive result on GB300 |
| --- | --- |
| `2^127 * 2` | `0x7f800000` (+infinity) |
| `2^127 + 2^127` | `0x7f800000` |
| `2^128 - 2^104` | `0x7f7fffff` (largest finite) |
| `2^128 - 2^103` | `0x7f7fffff` |
| `2^128 - 2^102` | `0x7f800000` (small term lost at alignment) |
| `2^128 - 2^127` | `0x7f000000` (`2^127`) |
| `2^254 - 2^254` | `0x00000000` (+zero) |

The negative cases are symmetric, with cancellation still producing +zero.
Before the fix, eight full PTX/SASS execution replays disagreed on 95,744 of
243,712 output words. All words agree after the fix. Lean matches all 112 BF16
case/configuration pairs; TF32 is covered by the full Rust execution replays.

Captured November 6, 2026, on SM103 with driver **580.126.20**, CUDA driver API
13030, and ptxas **13.1.115**. The original capture and source hashes are in
`expected/capture.json`. ptxv's numerical error bounds exclude overflow; this
fix corrects concrete execution, without widening those bounds.

It uses the unchanged kernels in `../mma_zero/kernel.ptx`:

| Instruction | Shape | Accumulator |
| --- | --- | --- |
| `tcgen05.mma.kind::f16`, BF16 inputs / FP32 output | M128 N16 K16 | Disabled and enabled |
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | M16 N8 K16 | Constant zero and registers |
| `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | M16 N8 K8 | Constant zero and registers |

The 15 cases run with both signs. Four cases need a nonzero accumulator and are
omitted when accumulation is disabled: 168 case/configuration pairs overall.
Every CTA computes a uniform tile, checked cell by cell. Three launches use
different output prefills; the output hashes must agree and the output guard
must remain intact. Identity and near-overflow finite cases serve as controls.

On an otherwise unused GB300, with CUDA 13 `ptxas` on `PATH`:

```sh
python3 probes/mma_overflow/probe.py > capture.json
```

The JSON contains the driver release, device UUID, CUDA driver API version,
compiler version, source hashes, compiled cubin, input bytes, and captured output
bytes. Binary payloads are compressed and hashed. To retain a capture printed as
`MMA_OVERFLOW <json>` by the test worker:

```sh
python3 probes/mma_overflow/unpack.py capture.log --output /tmp/new-mma-capture
```

`unpack.py` also accepts the standalone JSON. It verifies blob hashes, regenerates
and compares inputs, and verifies the summary against every captured output cell. The Rust regression uses the
captured cubin for SASS replay and the matching PTX source for PTX replay:

```sh
source env.sh
cargo test --locked --release -p ptxv-sim --test mma_overflow -- --nocapture
```

There are eight execution replays: both frontends for each `tcgen05` configuration,
SASS for each `mma.sync` configuration. These are measured GB300 examples, not an
exhaustive characterization of MMA, other GPU generations, or PTX's unspecified
rounding choices.
