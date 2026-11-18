# Composition of instruction and memory effects

This package depends on `instruction_effects` and `memory_write`. Neither subsystem
imports this package or the other subsystem. Rust IR types are extracted once by
Instruction; MemoryWrite retains its generic register-key and payload interfaces.

## Contracts

- `resolved_key_alias_iff` proves the symbolic register projection preserves exactly
  the physical aliases of the extracted owner key, including virtual and physical
  register-file separation.
- `source_resolved_copy_refines_commit` composes extracted owner resolution with
  extracted metadata preparation and assignment, then proves exact agreement with
  the symbolic commit model.
- `source_owned_word_commit_preserves_dynamic` derives the four operand certificates
  from a sound symbolic state and composes actual key, input and word-commit calls.
- `source_guarded_retention_preserves_all_input_evidence` composes the actual
  destination metadata decision with physical-alias checks. Comparison evidence
  remains sound after either an active write or a possible input-dependent write,
  including a guard that was false in the capture.

## Boundaries

The word-store theorem describes one four-byte slot per resolved operand. Native
Machine capture, HashMap retention, thread collection and trace interpretation
remain adapters. Topology, representation and storage-capacity premises are
explicit. These proofs do not establish complete interpreter or explorer correctness.

The checker freshly extracts and verifies both source packages before staging a
portable composition build. It checks source/tool freshness and transitively audits
all composition theorem declarations using only `propext`, `Classical.choice` and
`Quot.sound`.

`Instruction.actual_memory_op_query_catches_every_path_write` connects the actual
selected memory-instruction destination producer through actual register-key
resolution, nested interning, graph publication and may-write queries. The thin
native iterator from registers to program keys is an explicit premise, alongside
instruction index bounds and the graph/path contract. This composition requires
no caller-supplied ideal destination list or unique register-ID assumption.

## Actual graph and cache production

`published_source_analysis_and_queries_return_true` derives successful graph
publication, successful analysis and true queries for selected memory-family
writes on raw static-CFG paths to a labeled barrier. Actual IR barrier seeds,
source reachability propagation and actual register-index construction supply
the producer results; successful analysis, queries, dense lookup, build and field
assignment are conclusions. Known PC/bit facts prove reachability termination;
processed input keys bound interner capacity and nested source-loop progress.

`ProgramSourceInputs` retains native CFG/reach/destination publication equations,
valid destination spans, target/return storage and finite key/capacity budgets.
Unsupported instructions, sound folds, complete returns and dynamic indirect
calls remain distinct obligations. This proves the subsystem contract for the
independent static relation; it does not establish whole-explorer completeness.
Missing compacted PCs retain conservative unknown behavior. The inverted-progress
control additionally builds and audits a concrete extracted self-loop witness:
no natural-number measure can strictly decrease from that unchanged loop state.

## Actual instruction rows and key publication

`kernel_source_publication_analysis_and_queries_return_true` derives selected
memory-family destination rows from actual instructions, actual canonical-key
loops and the actual core raw-row converter, then derives successful publication,
analysis and true path-write queries. Destination-row equality and successful
collector/converter/builder/analysis/query premises are conclusions. The immutable
program owner now constructs its private rows from its own kernel and exposes
read-only rows. SourceMemoryRows is derived for supported operations independently
of the unsupported callback's contents.

LegacyReturns applies only to unsupported instructions. Selected register spans,
actual returned-row key-count capacity, native wrapper correspondence, CFG/reach
publication, target/return storage and AnalysisFits remain explicit. The IR and
memory core remain independent; raw key triples cross their interface without a
second instruction schema. Native E2E measurements cover the temporary conversion
allocation. A compile-fail probe requires the intended E0596 mutation error.

## Source-owned CFG and reach construction

`native_kernel_publication_analysis_and_queries_return_true` derives the graph
through actual IR `graph_and_seeds`, actual core `graph_source::build`, destination
collection, key conversion and publication. Caller CFG/reach/destination equations
and successful intermediate results are conclusions. A private native builder owns
the kernel, tables and graph, exposes immutable early views, then moves them into
the immutable program. Successor and seed allocations and the existing auxiliary
arrival/id flags are preserved; those flag algorithms remain outside this theorem.

Native Arc/code-view correspondence, input target/return storage and validity,
correct dynamic target/return/fold interpretation, register spans, unsupported
callbacks, returned-key capacity and AnalysisFits remain explicit. No production
constructor accepts an arbitrary graph. Kernel-free default publication is empty;
nonempty raw fixtures are restricted to tests. This is a static path may-write
contract, not full WALKS or whole-explorer correctness. Source wiring controls,
actual parsed construction, allocation identity and E0596 immutability checks cover
the corresponding native boundaries.

## Certified folds and dynamic-path memory queries

`certified_native_queries_catch_dynamic_path_writes` derives published-CFG edge
coverage from the accepted guard certificate and actual merged/published rows.
Independent partial-state paths, including private-preserving peer steps, then
supply the write path and future barrier path to the actual graph/key/worklist/query
source theorem. Fold truth, static output paths, and successful analysis/query
calls are derived conclusions.

The local path relation uses the existing conservative control semantics with a
supplied predicate choice, local defined assignment/frame interpretation, and
actual code/return views. Its bridge to every native execution step remains
explicit: targets/returns and branch qualifiers, raw values plus undefined masks,
physical aliases/width, private peer/async preservation, front/parser validity,
constant-bank soundness, native grid/merge receipt correspondence, finite storage,
unsupported destination callbacks and AnalysisFits. This covers selected memory
families' destination registers; full WALKS and whole-explorer correctness remain
separate obligations.

The separate `CompositionWitness.lean` gate demonstrates that an unchecked wrong
fold drops a false guard's admitted fallthrough in the actual source classifier.
It is a kernel-checked necessity witness, rather than another source mutation.

### Atomic local admission

AtomicAdmissionInterop composes the actual typed IR classifier with the actual extracted
core admission/complete-pair scan. Four source theorems connect CAS, observable outputs
and ordinary scalar RED collisions to continuation or diagnostic obligations. Native
claim membership, location/owner/continuation projection and finding transport remain
adapter boundaries; disjoint-word external observers remain unresolved.
