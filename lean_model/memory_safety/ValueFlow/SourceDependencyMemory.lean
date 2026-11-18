/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceDependencies

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

abbrev MetadataVector := alloc.vec.Vec Metadata

def MetadataFrame (before after : MetadataVector) (offset count : Nat) (metadata : Metadata) : Prop :=
  after.val.length = before.val.length ∧
  (∀ index, index < offset ∨ offset + count ≤ index → after.val[index]? = before.val[index]?) ∧
  (∀ index, offset ≤ index → index < offset + count → after.val[index]? = some metadata)

private theorem metadata_frame_zero (before : MetadataVector) (offset : Nat) (metadata : Metadata) :
    MetadataFrame before before offset 0 metadata := by
  simp [MetadataFrame]
  omega

private theorem metadata_frame_set (before current : MetadataVector) (offset count : Nat)
    (index : Usize) (metadata : Metadata) (indexEq : index.val = offset + count)
    (inside : index.val < current.val.length) (framed : MetadataFrame before current offset count metadata) :
    MetadataFrame before (current.set index metadata) offset (count + 1) metadata := by
  obtain ⟨sameLength, outside, written⟩ := framed
  refine ⟨by simpa using sameLength, ?_, ?_⟩
  · intro cursor beyond
    have different : index.val ≠ cursor := by omega
    simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_ne different]
    exact outside cursor (by omega)
  · intro cursor lower upper
    by_cases same : index.val = cursor
    · subst cursor
      simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_self inside]
    · simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_ne same]
      exact written cursor lower (by omega)

@[step]
theorem source_dependency_store_loop (offset count : Usize) (metadata : Metadata)
    (before current : MetadataVector) (cursor : Usize)
    (progress : cursor.val ≤ count.val)
    (fits : offset.val + count.val ≤ before.val.length)
    (framed : MetadataFrame before current offset.val cursor.val metadata) :
    input_independent_execution.apply_store_loop offset count metadata current cursor
      ⦃ after => MetadataFrame before after offset.val count.val metadata ⦄ := by
  unfold input_independent_execution.apply_store_loop
  apply loop.spec_decr_nat
    (fun current : MetadataVector × Usize => count.val - current.2.val)
    (fun current : MetadataVector × Usize => current.2.val ≤ count.val ∧
      MetadataFrame before current.1 offset.val current.2.val metadata)
  · rintro ⟨current, cursor⟩ ⟨progress, framed⟩
    change cursor.val ≤ count.val at progress
    change MetadataFrame before current offset.val cursor.val metadata at framed
    unfold input_independent_execution.apply_store_loop.body
    by_cases inside : cursor.val < count.val
    · have sourceInside : cursor < count := by scalar_tac
      simp only [sourceInside, ite_true]
      step as ⟨selected, selectedEq⟩
      have selectedBound : selected.val < current.val.length := by
        have sameLength := framed.1
        scalar_tac
      step as ⟨_, replace, selectedRead, replaceEq⟩
      step as ⟨next, nextEq⟩
      simp only [replaceEq]
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      have updated := metadata_frame_set before current offset.val cursor.val selected metadata selectedEq selectedBound framed
      simpa only [nextEq] using updated
    · have sourceOutside : ¬cursor < count := by scalar_tac
      have complete : cursor.val = count.val := by omega
      simp only [sourceOutside, ite_false, WP.spec_ok]
      simpa only [complete] using framed
  · exact ⟨progress, framed⟩

@[step]
theorem source_dependency_store (memory : Slice MetadataVector)
    (access : execution_core.ResolvedAccess) (metadata : Metadata) :
    input_independent_execution.apply_store memory access metadata ⦃ result => result.1 = true →
      ∃ before after, memory.val[access.allocation.val]? = some before ∧
        result.2 = memory.set access.allocation after ∧
        MetadataFrame before after access.offset.val access.width.val metadata ⦄ := by
  unfold input_independent_execution.apply_store
  dsimp only
  split
  next outside => simp
  next inside =>
    step as ⟨before, replace, beforeEq, replaceEq⟩
    split
    next beyond => simp
    next offsetFits =>
      step as ⟨remaining, remainingEq⟩
      split
      next tooWide => simp
      next widthFits =>
        have fits : access.offset.val + access.width.val ≤ before.val.length := by scalar_tac
        step with source_dependency_store_loop access.offset access.width metadata before before 0#usize
          (by scalar_tac) fits (metadata_frame_zero before access.offset.val metadata)
          as ⟨after, afterFrame⟩
        simp only [WP.spec_ok, true_implies]
        refine ⟨before, after, ?_, by simp only [replaceEq], afterFrame⟩
        rw [List.getElem?_eq_getElem (by scalar_tac)]
        exact congrArg some beforeEq.symm

@[step]
theorem source_dependency_read_loop (offset count : Usize) (bytes result : MetadataVector) (cursor : Usize)
    (progress : cursor.val ≤ count.val) (fits : offset.val + count.val ≤ bytes.val.length)
    (copied : result.val = (bytes.val.drop offset.val).take cursor.val) :
    input_independent_execution.loaded_metadata_loop offset count bytes result cursor
      ⦃ after => after.val = (bytes.val.drop offset.val).take count.val ⦄ := by
  unfold input_independent_execution.loaded_metadata_loop
  apply loop.spec_decr_nat
    (fun current : MetadataVector × Usize => count.val - current.2.val)
    (fun current : MetadataVector × Usize => current.2.val ≤ count.val ∧
      current.1.val = (bytes.val.drop offset.val).take current.2.val)
  · rintro ⟨result, cursor⟩ ⟨progress, copied⟩
    change cursor.val ≤ count.val at progress
    change result.val = (bytes.val.drop offset.val).take cursor.val at copied
    unfold input_independent_execution.loaded_metadata_loop.body
    by_cases inside : cursor.val < count.val
    · have sourceInside : cursor < count := by scalar_tac
      simp only [sourceInside, ite_true]
      step as ⟨selected, selectedEq⟩
      have selectedBound : selected.val < bytes.val.length := by scalar_tac
      step as ⟨entry, entryEq⟩
      have copiedLength : result.val.length = cursor.val := by
        simp only [copied, List.length_take, List.length_drop]
        omega
      step with alloc.vec.Vec.push_spec result _
        (by have := alloc.vec.Vec.len_ineq bytes; scalar_tac) as ⟨updated, appended⟩
      step as ⟨next, nextEq⟩
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      have droppedInside : cursor.val < (bytes.val.drop offset.val).length := by
        simp only [List.length_drop]
        omega
      rw [appended, nextEq, List.take_succ_eq_append_getElem droppedInside, copied]
      apply congrArg (fun value => (bytes.val.drop offset.val).take cursor.val ++ [value])
      simpa only [List.getElem_drop, selectedEq] using entryEq
    · have sourceOutside : ¬cursor < count := by scalar_tac
      have complete : cursor.val = count.val := by omega
      simp only [sourceOutside, ite_false, WP.spec_ok]
      simpa only [complete] using copied
  · exact ⟨progress, copied⟩

@[step]
theorem source_dependency_read_memory (memory : Slice MetadataVector) (access : execution_core.ResolvedAccess) :
    input_independent_execution.loaded_metadata memory access ⦃ result => ∀ loaded,
      result = some loaded → ∃ bytes, memory.val[access.allocation.val]? = some bytes ∧
        access.offset.val + access.width.val ≤ bytes.val.length ∧
        loaded.val = (bytes.val.drop access.offset.val).take access.width.val ⦄ := by
  unfold input_independent_execution.loaded_metadata
  dsimp only
  split
  next outside => simp
  next inside =>
    step as ⟨bytes, bytesAt⟩
    split
    next beyond => simp
    next offsetFits =>
      step as ⟨remaining, remainingEq⟩
      split
      next tooWide => simp
      next widthFits =>
        have fits : access.offset.val + access.width.val ≤ bytes.val.length := by scalar_tac
        step with source_dependency_read_loop access.offset access.width bytes (alloc.vec.Vec.new _) 0#usize
          (by scalar_tac) fits (by simp) as ⟨loaded, loadedEq⟩
        simp only [WP.spec_ok, Option.some.injEq]
        intro output same
        subst output
        refine ⟨bytes, ?_, fits, loadedEq⟩
        rw [List.getElem?_eq_getElem (by scalar_tac)]
        exact congrArg some bytesAt.symm

@[step]
theorem source_dependency_loaded_defined (loaded : Slice Metadata) (bytes : U8) :
    value_flow_dependencies.loaded_metadata loaded bytes ⦃ result => ∀ metadata,
      result = .Ok metadata → ∀ value ∈ loaded.val, value.defined = true ⦄ := by
  unfold value_flow_dependencies.loaded_metadata
  step with UScalar.cast_inBounds_spec .Usize bytes (by scalar_tac) as ⟨width, widthEq⟩
  split
  next wrongShape => simp
  next shape =>
    step
    split
    next empty => simp
    next nonempty =>
      step as ⟨initial, initialEq⟩
      subst initial
      step as ⟨result, resultDefined, resultTaint⟩
      split
      next defined =>
        simp only [WP.spec_ok, core.result.Result.Ok.injEq]
        intro metadata equal
        subst metadata
        intro value member
        obtain ⟨index, inside, found⟩ := List.mem_iff_getElem.mp member
        exact resultDefined defined index value inside
          (by rw [List.getElem?_eq_getElem inside]; exact congrArg some found)
      next undefined => simp

theorem metadata_dependency_set (values : List Metadata) (index : Nat) (metadata : Metadata)
    (inside : index < values.length) (defined : metadata.defined = true) (query : Nat) :
    metadataDependency (values.set index metadata) query =
      if query = index then some metadata.tainted else metadataDependency values query := by
  by_cases same : query = index
  · subst query
    simp only [metadataDependency, List.getElem?_set_self inside, Option.bind_eq_bind,
      Option.bind_some, defined, ite_true]
  · have different : index ≠ query := Ne.symm same
    simp only [metadataDependency, List.getElem?_set_ne different, same, ite_false]

theorem dependencies_model_virtual_set (registers : DependencyRegisters) (index : Usize) (metadata : Metadata)
    (inside : index.val < registers.virtuals.val.length) (defined : metadata.defined = true) :
    dependenciesModel { registers with virtuals := registers.virtuals.set index metadata } =
      { dependenciesModel registers with virtual := (fun query =>
        if query = index.val then some metadata.tainted else (dependenciesModel registers).virtual query) } := by
  apply congrArg₂ Dependencies.mk
  · funext query
    simpa only [alloc.vec.Vec.set_val_eq, dependenciesModel] using
      metadata_dependency_set registers.virtuals.val index.val metadata inside defined query
  · rfl

theorem dependencies_model_predicate_set (registers : DependencyRegisters) (index : Usize) (metadata : Metadata)
    (inside : index.val < registers.predicates.val.length) (defined : metadata.defined = true) :
    dependenciesModel { registers with predicates := registers.predicates.set index metadata } =
      { dependenciesModel registers with predicate := (fun query =>
        if query = index.val then some metadata.tainted else (dependenciesModel registers).predicate query) } := by
  apply congrArg₂ Dependencies.mk
  · rfl
  · funext query
    simpa only [alloc.vec.Vec.set_val_eq, dependenciesModel] using
      metadata_dependency_set registers.predicates.val index.val metadata inside defined query

theorem source_loaded_memory_metadata (memory : Slice MetadataVector) (access : execution_core.ResolvedAccess)
    (loaded : MetadataVector) (bytes : U8) (metadata : Metadata)
    (read : input_independent_execution.loaded_metadata memory access = .ok (some loaded))
    (joined : value_flow_dependencies.loaded_metadata loaded.slice bytes = .ok (.Ok metadata))
    (untainted : metadata.tainted = false) :
    ∃ original, memory.val[access.allocation.val]? = some original ∧
      access.offset.val + access.width.val ≤ original.val.length ∧
      ∀ index entry, access.offset.val ≤ index → index < access.offset.val + access.width.val →
        original.val[index]? = some entry → entry.defined = true ∧ entry.tainted = false := by
  have readSpec := source_dependency_read_memory memory access
  rw [read] at readSpec
  have readMeaning : ∃ original, memory.val[access.allocation.val]? = some original ∧
      access.offset.val + access.width.val ≤ original.val.length ∧
      loaded.val = (original.val.drop access.offset.val).take access.width.val := by
    have meaning : ∀ output, some loaded = some output → ∃ original,
        memory.val[access.allocation.val]? = some original ∧
        access.offset.val + access.width.val ≤ original.val.length ∧
        output.val = (original.val.drop access.offset.val).take access.width.val := by
      simpa only [WP.spec_ok, alloc.vec.Vec.val] using readSpec
    exact meaning loaded rfl
  obtain ⟨original, originalAt, fits, loadedEq⟩ := readMeaning
  have definedSpec := source_dependency_loaded_defined loaded.slice bytes
  rw [joined] at definedSpec
  have allDefined : ∀ value ∈ loaded.val, value.defined = true := by
    have meaning : ∀ output, (core.result.Result.Ok metadata : core.result.Result Metadata value_flow_dependencies.DependencyError) = .Ok output →
        ∀ value ∈ loaded.val, value.defined = true := by simpa only [WP.spec_ok, alloc.vec.Vec.val] using definedSpec
    exact meaning metadata rfl
  have taintedSpec := source_dependency_loaded loaded.slice bytes
  rw [joined] at taintedSpec
  have allUntainted : ∀ value ∈ loaded.val, value.tainted = false := by
    have meaning : ∀ output, (core.result.Result.Ok metadata : core.result.Result Metadata value_flow_dependencies.DependencyError) = .Ok output →
        output.defined = true ∧ (output.tainted = false → ∀ value ∈ loaded.val, value.tainted = false) := by
      simpa only [WP.spec_ok, alloc.vec.Vec.val] using taintedSpec
    exact (meaning metadata rfl).2 untainted
  refine ⟨original, originalAt, fits, ?_⟩
  intro index entry lower upper found
  have relativeBound : index - access.offset.val < access.width.val := by omega
  have originalIndex : access.offset.val + (index - access.offset.val) = index := by omega
  have loadedAt : loaded.val[index - access.offset.val]? = some entry := by
    simpa only [loadedEq, List.getElem?_take, relativeBound, ite_true, List.getElem?_drop, originalIndex] using found
  have member : entry ∈ loaded.val := List.mem_of_getElem? loadedAt
  exact ⟨allDefined entry member, allUntainted entry member⟩

theorem source_transfer_load_computation (instruction : V.GuardedInstruction)
    (destination : U16) (bits bytes : U8) (signed : Bool) (pointer : V.Pointer)
    (registers : DependencyRegisters) (loaded : Slice Metadata) (transferred : DependencyTransfer)
    (computed : value_flow_dependencies.transfer instruction
      (.Load destination bits bytes signed pointer) registers loaded = .ok (.Ok transferred)) :
    ∃ metadata, value_flow_dependencies.loaded_metadata loaded bytes = .ok (.Ok metadata) ∧
      transferred = .Virtual destination metadata := by
  unfold value_flow_dependencies.transfer at computed
  obtain ⟨guardResult, guardRun, remainder⟩ :=
    MemorySafety.Source.source_bind_success _ _ _ computed
  cases guardResult with
  | Err error => simp at remainder
  | Ok ignored =>
      cases sourceInstruction : instruction.instruction <;> simp only [sourceInstruction] at remainder
      all_goals try simp at remainder
      obtain ⟨addressResult, addressRun, remainder⟩ :=
        MemorySafety.Source.source_bind_success _ _ _ remainder
      cases addressResult with
      | Err error => simp at remainder
      | Ok ignored =>
          obtain ⟨metadataResult, metadataRun, remainder⟩ :=
            MemorySafety.Source.source_bind_success _ _ _ remainder
          cases metadataResult with
          | Err error => simp at remainder
          | Ok metadata =>
              refine ⟨metadata, metadataRun, ?_⟩
              simpa only [ok.injEq, core.result.Result.Ok.injEq] using remainder.symm

end ValueFlow.Source
