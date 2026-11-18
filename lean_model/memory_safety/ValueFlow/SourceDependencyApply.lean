/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceDependencyMemory
import ValueFlowDependencies.SourceSnapshot

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

abbrev DependencyState := value_flow_dependencies.State

def VirtualShadowEffect (before after : DependencyState) (actor : Usize)
    (destination : U16) (metadata : Metadata) : Prop :=
  ∃ registers : DependencyRegisters, ∃ index : Usize, index.val = destination.val ∧
    before.threads.val[actor.val]? = some registers ∧
    index.val < registers.virtuals.val.length ∧
    after = { before with threads := (before.threads.set actor
      { registers with virtuals := registers.virtuals.set index metadata }) }

def PredicateShadowEffect (before after : DependencyState) (actor : Usize)
    (destination : U16) (metadata : Metadata) : Prop :=
  ∃ registers : DependencyRegisters, ∃ index : Usize, index.val = destination.val ∧
    before.threads.val[actor.val]? = some registers ∧
    index.val < registers.predicates.val.length ∧
    after = { before with threads := (before.threads.set actor
      { registers with predicates := registers.predicates.set index metadata }) }

@[step]
theorem source_apply_transfer_virtual (program : execution_core.Program) (state : execution_core.State)
    (shadow : DependencyState) (actor : Usize) (action : V.Action) (destination : U16) (metadata : Metadata)
    (inside : actor.val < shadow.threads.val.length) :
    input_independent_execution.apply_transfer program state shadow actor action (.Virtual destination metadata)
      ⦃ result => ∀ after, result = some after → VirtualShadowEffect shadow after actor destination metadata ⦄ := by
  unfold input_independent_execution.apply_transfer
  rw [ValueFlowDependencies.Source.source_snapshot_state]
  simp only [bind_ok]
  step with UScalar.cast_inBounds_spec .Usize destination (by scalar_tac) as ⟨index, indexEq⟩
  step as ⟨registers, registersEq⟩
  split
  next outside => simp
  next available =>
    step as ⟨registersAgain, replaceRegisters, registersAgainEq, replaceRegistersEq⟩
    step with UScalar.cast_inBounds_spec .Usize destination (by scalar_tac) as ⟨indexAgain, indexAgainEq⟩
    have same : registersAgain = registers := by rw [registersAgainEq, registersEq]
    step as ⟨_, replace, _, replaceEq⟩
    simp only [WP.spec_ok, Option.some.injEq]
    intro after equal
    subst after
    refine ⟨registersAgain, indexAgain, indexAgainEq, ?_, ?_, ?_⟩
    · rw [List.getElem?_eq_getElem inside]
      exact congrArg some registersAgainEq.symm
    · scalar_tac
    · simp only [replaceEq, replaceRegistersEq]

@[step]
theorem source_apply_transfer_predicate (program : execution_core.Program) (state : execution_core.State)
    (shadow : DependencyState) (actor : Usize) (action : V.Action) (destination : U16) (metadata : Metadata)
    (inside : actor.val < shadow.threads.val.length) :
    input_independent_execution.apply_transfer program state shadow actor action (.Predicate destination metadata)
      ⦃ result => ∀ after, result = some after → PredicateShadowEffect shadow after actor destination metadata ⦄ := by
  unfold input_independent_execution.apply_transfer
  rw [ValueFlowDependencies.Source.source_snapshot_state]
  simp only [bind_ok]
  step with UScalar.cast_inBounds_spec .Usize destination (by scalar_tac) as ⟨index, indexEq⟩
  step as ⟨registers, registersEq⟩
  split
  next outside => simp
  next available =>
    step as ⟨registersAgain, replaceRegisters, registersAgainEq, replaceRegistersEq⟩
    step with UScalar.cast_inBounds_spec .Usize destination (by scalar_tac) as ⟨indexAgain, indexAgainEq⟩
    have same : registersAgain = registers := by rw [registersAgainEq, registersEq]
    step as ⟨_, replace, _, replaceEq⟩
    simp only [WP.spec_ok, Option.some.injEq]
    intro after equal
    subst after
    refine ⟨registersAgain, indexAgain, indexAgainEq, ?_, ?_, ?_⟩
    · rw [List.getElem?_eq_getElem inside]
      exact congrArg some registersAgainEq.symm
    · scalar_tac
    · simp only [replaceEq, replaceRegistersEq]

theorem source_apply_transfer_none (program : execution_core.Program) (state : execution_core.State)
    (shadow : DependencyState) (actor : Usize) (action : V.Action) :
    input_independent_execution.apply_transfer program state shadow actor action .None = .ok (some shadow) := by
  simp only [input_independent_execution.apply_transfer,
    ValueFlowDependencies.Source.source_snapshot_state, bind_ok]

def StoreShadowEffect (program : execution_core.Program) (state : execution_core.State)
    (before after : DependencyState) (pointer : V.Pointer) (bytes : U8) (metadata : Metadata) : Prop :=
  ∃ access original updated,
    execution_core.resolve_access program state pointer bytes true = .ok (.Ok access) ∧
    before.memory.val[access.allocation.val]? = some original ∧
    after.threads = before.threads ∧
    after.memory.val = before.memory.val.set access.allocation.val updated ∧
    MetadataFrame original updated access.offset.val access.width.val metadata

theorem source_apply_transfer_store_success (program : execution_core.Program) (state : execution_core.State)
    (shadow after : DependencyState) (actor : Usize) (bytes : U8) (pointer : V.Pointer) (value : U64)
    (metadata : Metadata)
    (computed : input_independent_execution.apply_transfer program state shadow actor
      (.Store bytes pointer value) (.Store metadata) = .ok (some after)) :
    StoreShadowEffect program state shadow after pointer bytes metadata := by
  unfold input_independent_execution.apply_transfer at computed
  rw [ValueFlowDependencies.Source.source_snapshot_state] at computed
  simp only [bind_ok] at computed
  obtain ⟨resolved, resolvedRun, remainder⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
  cases resolved with
  | Err error => simp at remainder
  | Ok access =>
      simp only [alloc.vec.Vec.deref_mut, lift, bind_ok] at remainder
      obtain ⟨written, writtenRun, storeRemainder⟩ :=
        MemorySafety.Source.source_bind_success _ _ _ remainder
      have meaning := source_dependency_store shadow.memory.slice access metadata
      rw [writtenRun] at meaning
      have storeMeaning : written.1 = true →
          ∃ original updated, shadow.memory.slice.val[access.allocation.val]? = some original ∧
            written.2 = shadow.memory.slice.set access.allocation updated ∧
            MetadataFrame original updated access.offset.val access.width.val metadata := by
        simpa only [WP.spec_ok] using meaning
      rcases written with ⟨accepted, output⟩
      cases accepted with
      | false => simp at storeRemainder
      | true =>
          obtain ⟨original, updated, originalAt, outputEq, frame⟩ := storeMeaning rfl
          change Result.ok (some ({ threads := shadow.threads, memory := ⟨output⟩ } : DependencyState)) =
            Result.ok (some after) at storeRemainder
          have afterEq : after = ({ threads := shadow.threads, memory := ⟨output⟩ } : DependencyState) := by
            simpa only [ok.injEq, Option.some.injEq] using storeRemainder.symm
          subst after
          refine ⟨access, original, updated, resolvedRun, originalAt, rfl, ?_, frame⟩
          change output.val = _
          simpa only [Slice.set_val_eq, alloc.vec.Vec.val] using congrArg Slice.val outputEq

end ValueFlow.Source
