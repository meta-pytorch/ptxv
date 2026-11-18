# Reproducing the strict-memory proof

All paths in `source-manifest.json` are relative to the PTXV repository or this
proof project. The default repository root is two directories above this project
(`lean_model/memory_safety`). Private checkouts must pass `--ptxv-root` explicitly.
Neither script downloads tools, packages, Rust crates, or build caches.

## Source freshness

```sh
python3 lean_model/memory_safety/check.py --source-only
```

This checks every extraction input, the actual extracted Rust closure, Cargo
inputs, native acceptance route, primitive model, backend patch, and both
generated Lean files against exact SHA-256 values. It also checks the complete
authored Lean module inventory. It does not claim that the proofs were compiled.

The manifest comes from genuine combined extraction of the final production
source, including the final simulator module declarations. `extraction-spec.json`
records the explicit source roots, translation flags, tool fingerprints and
primitive bindings. `qualified-extraction.json` records the qualified input,
LLBC and generated-code fingerprints. The freshness gate checks that this
qualification names every current extraction input, the current specification
and both generated files. Generated Lean bodies have no manual edits. The shared
`lean_model/generated_lean.py` publisher prepends one fixed license comment to
the exact Aeneas bytes. `extraction-spec.json` pins that transformation and its
preamble hash; the publisher itself is a sealed extraction input.

## Offline formal check

Use the pinned installed Lean toolchain and an existing Aeneas Lean backend with
all dependencies already cached:

```sh
python3 lean_model/memory_safety/check.py \
  --backend /path/to/aeneas/backends/lean \
  --lake /path/to/pinned-lean/bin/lake \
  --tools /path/to/local-tools.json \
  --out /tmp/ptxv-memory-safety-check
```

The output directory must be new and outside both source trees. Formal checking
first runs fresh Rust extraction against every current input and requires exact
generated-code equality. It then stages and builds every declared project module.
The audit replays all imported project constants through Lean's kernel in a
separate environment containing only external dependencies. This includes
generated definitions, private constants and inductive declarations. `--trust=0`
is also used, but does not replace explicit kernel replay.

Project axioms and dependencies beyond `propext`, `Classical.choice`, and
`Quot.sound` fail the audit. `endpoint-types.json` separately pins the
required theorem statements using binder-normalized, metadata-free elaborated
types and includes readable types for review. A same-name weaker theorem fails.
The sealer consumes this policy unchanged; it never derives new expected types.

`specification-inputs.json` separately pins the reviewed independent grammar,
execution and safety definitions. The sealer consumes that policy unchanged as
well. Changing a specification body to a weaker property therefore fails even
if ordinary source and proof seals are regenerated and the exported type name
stays the same. These reviewed policies are distinct from build receipts.

`backend-inputs.json` pins each cached package's sorted source/configuration and
compiled-environment inventory, including split private/server OLean files and
external path dependencies. It records stable package keys, file counts and
inventory digests. Added files and changed artifacts fail. The backend inventory
and all source hashes are checked before and after verification. A source-only
check validates the sealed inventory file; checking actual backend artifacts
requires the formal command.

The checker has two executable regression suites. The first tests its source,
inventory and trust-policy checks; the second compiles adversarial Lean modules:

```sh
python3 lean_model/memory_safety/test_check.py
python3 lean_model/memory_safety/test_kernel.py \
  --lean /path/to/pinned-lean/bin/lean \
  --out /tmp/ptxv-memory-safety-kernel-controls
```

The compiled controls establish that the audit rejects a weaker theorem type
and an imported malformed proof. They also demonstrate why axiom collection and
`--trust=0` alone do not replace explicit kernel replay. These use fixture modules;
the formal command above independently checks the actual memory-safety theorem.

`proof-manifest.json` seals every handwritten Lean module and names those final
theorems. The default check refuses an unsealed project. During proof development,
`--development` allows a provisional full build and audit, recorded explicitly as
`development-pass`; it is not a publication receipt. The proof seal is created
only after the final theorem and all source modules are frozen.

## Fresh Rust extraction

Create a local tools JSON file with these keys:

```json
{
  "charon": "/path/to/pinned/charon",
  "aeneas": "/path/to/patched/aeneas",
  "rust_bin": "/path/to/pinned/rust/bin",
  "cargo_home": "/path/to/populated/offline/cargo-home"
}
```

```sh
python3 lean_model/memory_safety/extract.py \
  --tools /path/to/local-tools.json \
  --out /tmp/ptxv-memory-safety-extraction
```

The script copies only the explicitly listed `extraction-inputs.json` files into
a new private workspace. Its `Cargo.lock` comes from the repository's committed
selection lockfile. Cargo runs with `--offline --locked`; Charon and Aeneas abort
on extraction errors. Exactly `Types.lean` and `Funs.lean` must be generated; an
external template fails the check. The publisher retains the unmodified tool
outputs in `raw-generated/`, records separate raw and published hashes, and
checks that removing exactly the fixed preamble recovers every original byte.
Both published files must match the checked-in generated files byte for byte.
The fresh-extraction checker and sealer independently enforce this relationship;
the qualified record retains the export policy and both sets of hashes. The
script retains commands, logs, input hashes and a receipt; it never replaces
checked-in code or expected hashes.

The pinned Aeneas binary includes `backend/aeneas-memory-primitives.patch`.
`backend/manifest.json` records original and patched source/binary fingerprints.
Apply the patch only to the matching backend source and build with the existing
offline OCaml environment. Its models include the empty string constructor,
exact UTF-8 string bytes, and an unused `Arc` representation in launch metadata. Acceptance separately
proves that NIC configuration is absent; no `Arc` operation is selected.
Printer margin changes affect formatting only. See `backend/README.md` for the
precise boundary.

When Rust changes, preserve the old manifest and run fresh extraction privately.
Review the new source closure and generated output, adapt and rebuild the proofs,
then update the source and proof manifests together. A source-only check does not
establish correctness of a regenerated or edited manifest.
