/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.OwnerSymbolic

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false

namespace RegisterOwner.Symbolic

open MemoryWrite MemoryWrite.Symbolic

theorem source_copy_refines_commit {Input : Type} (sem : Semantics Input)
    (state : SourceState Input) (source destination predicate : RegisterKey) :
    sourceCopy sem state source destination predicate ⦃ result =>
      sourceView result = commit sem (sourceView state) destination
        (prepare (sourceView state) source) (predicateGuard (state.values predicate))
        (state.metadata predicate).tainted ⦄ := by
  unfold sourceCopy
  step with rust_prepare_spec (state.metadata source).captured (state.metadata source).history
    (state.metadata source).tainted (state.metadata source).defined as ⟨frozen, frozenExact⟩
  step with rust_assign_metadata_spec (predicateGuard (state.values predicate) sem.capture)
    (state.metadata predicate).tainted frozen (state.metadata destination) as ⟨assigned, assignedExact⟩
  change metadataView frozen = metadataView (state.metadata source) at frozenExact
  simp only [sourceView, commit, prepare]
  congr 1
  funext key
  by_cases selected : key = destination
  · simp [selected, assignedExact, frozenExact]
  · simp [selected]

theorem source_resolved_copy_refines_commit {Input : Type} (sem : Semantics Input)
    (state : SourceState Input) (thread : Usize) (owners : Slice U32)
    (source destination predicate : Reg) :
    resolvedCopy sem state thread owners source destination predicate ⦃ result =>
      sourceView result = commit sem (sourceView state)
        (resolvedKey thread.val (owners.val.map UScalar.val) destination)
        (prepare (sourceView state) (resolvedKey thread.val (owners.val.map UScalar.val) source))
        (predicateGuard (state.values (resolvedKey thread.val (owners.val.map UScalar.val) predicate)))
        (state.metadata (resolvedKey thread.val (owners.val.map UScalar.val) predicate)).tainted ⦄ := by
  unfold resolvedCopy
  step with rust_canonical_key thread owners source as ⟨sourceKey, sourceExact⟩
  step with rust_canonical_key thread owners destination as ⟨destinationKey, destinationExact⟩
  step with rust_canonical_key thread owners predicate as ⟨predicateKey, predicateExact⟩
  rw [resolved_result_exact _ _ _ _ sourceExact,
    resolved_result_exact _ _ _ _ destinationExact,
    resolved_result_exact _ _ _ _ predicateExact]
  exact source_copy_refines_commit sem state _ _ _

end RegisterOwner.Symbolic
