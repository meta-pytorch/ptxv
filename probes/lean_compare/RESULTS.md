# Rust / Lean cross-check

This integration is based directly on ptxv
`88fe5ba67b1c2ea3a330d4dc3b361e2d9ac4e395`. The Rust simulator and its existing
probes are unchanged. The addition imports 625 Lean sources for GB300 and shared
arithmetic, adds a word-level checker, and links the original Rust functions
through a standalone bridge.

## Checks

| Check | Result |
| --- | --- |
| Ten native FP32 / FP64-high-word operations | 267,256 words per form agree |
| EX2 and TANH in FP16 and BF16 | All 65,536 inputs per form agree, including retained GB300 hashes |
| Seven PTX wrappers using pinned FloatLib | 267,256 words per form agree; 4,805,496 total scalar comparisons |
| Bounded synthetic BF16 MMA | 8,705 outputs agree, including signed-zero and underflow cases |
| Earlier retained BF16 MMA capture | 150,528 outputs agree |
| Existing Rust regressions | Five tests pass: four MUFU and MMA signed-zero; exhaustive FP32 replay remains opt-in |
| Lean core | Build, six original theorem audits, four checker witness audits, and five smoke checks pass |
| Optional overflow comparison | Two disagreements detected; runner returns nonzero |

The base Rust MMA model returns the largest finite value at overflow, while Lean
returns signed infinity. The optional `--check-overflow` run reports both signs;
this branch does not change Rust to resolve them. The simulator fix is maintained
separately on `rusch/mma-overflow-infinity`.

## Reproduction and scope

[README.md](README.md) gives the commands. [results/validation.json](results/validation.json)
records source hashes, commands, and outcomes for this independent branch.
The optional MMA holdout is external; its hashes and available device metadata
are in [results/mma.json](results/mma.json). No GPU data was recollected here.

Scalar inputs cover exponent and interpolation boundaries, BF16 embeddings,
seeded random words, FP64 high-word boundaries, and recorded special values.
Negative controls reject changed output bits, partial words, and PTX wrapper
spellings passed to a bare MUFU checker. FP32 TANH agreement is sampled across
its range; it is not an exhaustive equivalence proof.

The bridge uses its own Cargo lockfile and the original regression test sources.
This is not a full root-workspace test run. The signed-zero test covers 14
configurations through 24 PTX/SASS replays. `arc lint` is unavailable because this
repository has no configured lint engine.

GB300 arithmetic definitions are unchanged. The source manifest records the subset
and adaptations to imports, replay entry points, and comparison theorems. The core
was rebuilt without previous build outputs; all smoke checks use the SM103 checker.
Default Lake targets cover selected modules. Optional
FloatLib/Mathlib and generated-certificate modules are present as source but are
not all built; `lean_model/certificate-imports.json` lists missing generated
imports. The external FloatLib source pin is checked before using its existing
build. Lean theorems establish properties of their models; executable comparisons
do not prove the Rust simulator, compiler, GPU, or synchronization semantics.
