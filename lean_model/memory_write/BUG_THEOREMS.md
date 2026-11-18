# Theorems for historical memory-effect bugs

These obligations compose the extracted Rust transitions with their consumers.
They use the existing three permitted Lean axioms and are included in the full
project build and axiom audit.

| Historical failure | Source-facing theorem | Rejected Rust mutation |
| --- | --- | --- |
| CAS64 retained the overwritten upper word | `rust_successful_cas_cell_commit` and its upper-word corollary derive history invalidation from actual execution, encoding, and byte writes | `atomic_width_truncated`, `copy_retains_destination_history` |
| Partial/unaligned writes retained an intersected word | `rust_unaligned_word_write_rejects_both_words` invalidates both intersected histories, including a history starting before the write | `exact_start_only` |
| Rewrites retained or recovered original half identities | `rust_unknown_write_blocks_initial_origin` rejects the identity consumer after actual origin replacement; `MarkerConsumerSafety.enabled_global_apply_consumer_safety` connects actual independent marker updates to fallback rejection | `rewrite_keeps_stale_half_origin`, `missing_written_marker`, `half_tracker_coupled_to_numeric` |
| A symbolic value captured as zero lost its dependence | `rust_symbolic_zero_cannot_equal_constant` derives taint from semantic variation; `rust_captured_zero_is_not_constant` supplies a sound concrete counterexample | `captured_zero_clears_taint` |
| Async copies paired issue-time bytes with different histories at completion | `rust_async_copy_preserves_issue_histories` preserves all captured metadata; `rust_async_copy_consumers_use_issue` derives source soundness from actual capture and proves destination consumers denote the issue memory | `snapshot_invents_zero_origin`, `snapshot_invents_defined_bytes`, `snapshot_clears_taint`, `snapshot_promotes_opaque_history` |
| A dependent guard was treated as a fixed captured execution | `rust_accepted_write_requires_independent_control` and its engine version derive both independent flags from success, including disabled writes | `dependent_guard_accepted`, `captured_false_dependent_guard_accepted` |
| Bulk writes left old destination histories | `rust_apply_write_consumer_safety` and the frozen-copy theorems replace destination histories with the issue snapshot, rather than old destination metadata | `copy_retains_destination_history`, `postword_history_dropped` |

## What the statements establish

The async theorems permit an arbitrary represented completion memory, including
source overwrites and overlapping destinations. They derive the payload
certificate from actual `checked.capture_span`; the caller does not supply a
`PayloadSound` premise. The consumer theorem covers known origins, complete
known-word histories, and input-independent constants across every admissible
input, with the same semantic interpretation at issue and completion.

The marker theorem derives membership from actual `engine.apply`. Numeric and
half tracking remain independent. A write touching any byte blocks initial
numeric fallback for that byte's aligned word; every written half byte blocks
its own initial origin fallback. Explicit valid current origins remain usable.

The CAS theorem composes extracted calls in a Lean commit wrapper. Its result
definedness is supplied by the producer; execution on bit patterns alone does
not establish definedness. The failed-CAS theorem preserves all byte planes
while allowing the existing numeric-only forget and marker effect. It also
derives the old-value return, predicate, and unchanged epoch.

## Production boundaries

These are the closed obligations of the extracted memory-effect subsystem.
The generic `Machine` atomic path uses the extracted execution, byte codec, and
footprint helpers. Its distributed map loops are covered by Machine regressions,
including a previously recorded rejection of the actual pre-fix CAS64 code.
The new commit wrapper does not constitute extraction of those loops.

The symbolic subsystem now extracts missing-history classification, frozen
operand transfers, namespace-qualified keys, checked node IDs, and control
construction. Its source proofs derive the existing memory payload and stable
execution contracts. Production numeric fallback, atomic/MMA pre-write capture,
predicate history keys, and full-width elementwise leaf bindings have matching
regressions. The complete production taint and expression-graph adapters are
not universally extracted.

The four-byte store constructor and engine composition are proved together;
[STORE_INTERFACE.md](STORE_INTERFACE.md) records the migrated production subset
and the capture/storage adapter boundary. Async queue ownership, bulk/CLC
dispatch, and general vector register invalidation remain outside extraction.

Floating arithmetic and scheduler obligations remain as described in
[ATOMIC_BOUNDARIES.md](ATOMIC_BOUNDARIES.md).
