# Instruction effect contracts

The extracted Rust entrypoints match actual `ptxv_ir::Op` fields. Their Lean contracts
compare results with independent register and bit-read semantics.

Supported memory instructions are `Ld`, `Ld256`, `St`, `St256`, `StAsync`, and all
`Atom` variants. The destination list is a may-write list: the instruction guard is
handled separately. Atomic replacements belong to data sources, including implicit
SASS CAS replacement registers. Store source counts follow actual payload assembly,
including width overrides of physical operand counts and distinct async thresholds.
`actual_reads_covered_spec` proves execution reads are included in the static list.

`read_window::narrow` selects the actual operand and consumed bit interval for
16-bit float conversions and 8/16-bit integer conversions. Other instructions use
the existing conservative definedness path.

`store_window::sources` selects actual store/exchange data registers and destination
byte intervals. `store_windows_spec` proves this producer agrees with the independent
layout, including physical word widths, virtual vector element widths, and the
frontend guard. `source_window_byte_in_payload` establishes byte locality.

The isolated IR archive uses this probe's pinned Cargo lockfile. Production simulator
builds use the workspace lockfile.

Run from the PTXV root with installed tools and cached Cargo dependencies:

```sh
python3 probes/instruction_effects/check.py \
  --tools /path/to/aeneas-tools.json --out /tmp/ptxv-instruction-check \
  --negative-controls
```

Use a fresh output directory. `--regenerate` replaces the two generated Lean files.
Both extraction pipelines publish the exact Aeneas bytes with the fixed license
comment from `lean_model/generated_lean.py`. The original files and separate raw
and published hashes remain in `raw-generated/`; final receipts include those
export records. The byte equality gate compares the published files, including
the deterministic comment.
The checker archives actual IR sources, runs their Rust tests, compares a fresh
extraction, builds the proofs, audits theorem axioms, and records source/tool hashes.
Negative controls must fail Rust regressions, compile through extraction, and fail
the corresponding source theorem.

`Instruction.memory_reads_spec`, `memory_writes_spec`, `data_reads_spec`, and
`control_reads_spec` preserve complete file/index observations. `narrow_read_window_spec`
preserves the source operand and exact consumed interval. Register ownership,
metadata transfer, and the state frame are separate composition obligations.

`register_owner` constructs warp owners and preserves full thread/file/index keys.
Its source proofs establish canonical aliasing and comparison-cache retention.
Predicate writes invalidate physical aliases; input-dependent guarded writes do so
even when skipped in the capture. The native cache collection and trace adapters
remain explicit boundaries. Uniform-predicate regressions use native IR; their
hardware source encoding is not established.
