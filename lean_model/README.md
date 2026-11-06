# GB300 instruction models in Lean

Bit-level numerical models, Lean replay executables, and selected accuracy
proofs for NVIDIA GB300 (SM103). This directory is a standalone Lake package
beside ptxv's Rust workspace.

| Folder | Hardware |
| --- | --- |
| [Blackwell/](Blackwell/) | GB300, SM103 |
| [Shared/](Shared/) | Reusable arithmetic and common instruction models |

`Blackwell/` holds the GB300 models and replay programs. `Shared/` supplies
arithmetic, tables, and common instruction semantics used by those models.
Logical module names retain the `GB300Math` prefix.

## Build the core

Install Lean `v4.34.0`, as pinned in `lean-toolchain`, then run:

```sh
cd lean_model
lake build
lake env lean Audit.lean
lake build check_mufu_sm103 check_ptxv
python3 tests/smoke.py
```

The default build and six-theorem audit require no GPU, FloatLib, Mathlib, Buck,
or captured data. They do not compile every standalone certificate in this tree.
`source-manifest.json` identifies the imported source snapshot and file hashes.

## Compare with Rust and hardware

[probes/lean_compare/](../probes/lean_compare/README.md) contains the Rust bridge,
corpus generators, comparison runners, and result receipts. It reuses the scalar
fixtures already in ptxv and accepts an optional external MMA capture.
Hardware capture code and binary fixtures stay under `probes/`.

The `.lean` replay executables stay here because Lake builds them against the
model modules. `tests/smoke.py` checks their word interface and rejection paths.
The probe's [results](../probes/lean_compare/RESULTS.md) distinguish model
agreement, hardware evidence, and limits.

## Word interface

[Mufu](Shared/GB300Math/Mufu.lean) exposes word-to-word semantics for scalar math
units. [MufuSm103](Blackwell/GB300Math/MufuSm103.lean) adds GB300-specific forms.
The replay programs accept little-endian 32-bit input and output files:

```sh
.lake/build/bin/check_mufu_sm103 rcp inputs.u32 outputs.u32 4 4
```

The last two arguments are byte strides. Unsupported operations, malformed
files, and mismatches fail. Record the exact operation spelling from the SM103
module's `evaluate` function.
This protocol works with outputs captured from hardware or produced by ptxv.
The checker uses executable Lean code; it is a comparison, not a theorem about
the Rust simulator or GPU.

`check_ptxv OP INPUT OUTPUT` uses ptxv's operation names: `rcp.ftz` is the bare
unit, while `rcp` is a PTX wrapper and is rejected by this core-only checker.
`rcp64h` and `rsq64h` consume and return the high word, not a whole FP64 value.
`mma.bf16` consumes rows of 33 words: C, A[16], B[16]; each BF16 occupies the low
16 bits of a word. Its result is one FP32 word per row. This checks one arithmetic
step, with no claim about fragment mapping or scheduling.

The source collection covers math units, conversions, integer operations, predicates,
tensor arithmetic, and texture/fabric behaviors. SASS execution, scheduling, and
state exploration remain in ptxv. A bit-result model can supply the arithmetic
at a simulator transition; permitted implementation choices need a relation or
set of outcomes, rather than one arbitrarily selected result.

## Proof and validation scope

A Lean theorem establishes a property of the defined model. A replay compares
that model with supplied words. An accuracy theorem relates a model to an ideal
real function. None alone establishes bit-exact agreement with every GPU input.
Preserve hardware identity, compute capability, driver release, CUDA/compiler
versions, flags, input domain, and hashes alongside external captures.

Standalone sources may import FloatLib, Mathlib, or generated finite-domain
certificates. The FloatLib source pin is in `floatlib-lock.json`, with per-file hashes in
`floatlib-sources.json`; dependencies
and large certificate/capture bundles are not vendored here.
`certificate-imports.json` lists imports not supplied by this package or the
named external libraries. The default build is deliberately smaller than the
source inventory. An optional source's presence is not evidence it was rebuilt.

Historical hypotheses remain for comparison. In particular, the exact-sum
`Blackwell/FabricF16.lean` model was rejected by a holdout;
`Blackwell/FabricF16V2.lean` is the revised model. Do not treat every source as a
validated hardware model or assume an unlisted property has been proved.

The original scalar interpolation reconstruction is Bram Wasti's ptxv work.
The Lean port retains separate models and tests so simulator and proof results
can be cross-checked. BC6H and BC7 table sources retain their attribution and
license notes; see [NOTICE.md](NOTICE.md).
