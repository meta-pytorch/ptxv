/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.InputBanks
import ValueFlow.SourceWriteVirtual
import ValueFlow.SourceWritePredicate

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

open Storage.MachineRegistersVirtualMap

def virtualDependencies (before : Dependencies) (index : Nat) (tainted : Bool) : Dependencies :=
  { before with virtual := fun query => if query = index then some tainted else before.virtual query }

def predicateDependencies (before : Dependencies) (index : Nat) (tainted : Bool) : Dependencies :=
  { before with predicate := fun query => if query = index then some tainted else before.predicate query }

def renderedWord (bits : U64) (value : V.Value) : Word :=
  { bits := bits.val, undefined := 0, pointer := (valuePointer value).map pointerModel }

private theorem rendered_words_related (left right : V.Value) (leftBits rightBits : U64)
    (leftLayouts rightLayouts : Slice value_flow.AllocationLayout) (tainted : Bool)
    (leftRendered : value_flow.render_value left leftLayouts = .ok (.Ok leftBits))
    (rightRendered : value_flow.render_value right rightLayouts = .ok (.Ok rightBits))
    (related : ValuesRelated tainted (valueModel left) (valueModel right)) :
    if tainted then (renderedWord leftBits left).pointer = none ∧
      (renderedWord rightBits right).pointer = none
    else WordRelated (renderedWord leftBits left) (renderedWord rightBits right) := by
  cases tainted with
  | false =>
    simp only [ValuesRelated, Bool.false_eq_true, ite_false] at related ⊢
    cases left with
    | Scalar leftValue =>
      cases right with
      | Scalar rightValue =>
        have leftEq : leftValue = leftBits := by simpa [value_flow.render_value] using leftRendered
        have rightEq : rightValue = rightBits := by simpa [value_flow.render_value] using rightRendered
        have bitsEq : leftBits.val = rightBits.val := by
          simpa only [valueModel, Value.scalar.injEq, leftEq, rightEq] using related
        exact ⟨rfl, rfl, fun _ => bitsEq⟩
      | Pointer pointer => cases related
    | Pointer leftPointer =>
      cases right with
      | Scalar scalar => cases related
      | Pointer rightPointer =>
        have pointerEq : pointerModel leftPointer = pointerModel rightPointer := by
          simpa only [valueModel, Value.pointer.injEq] using related
        exact ⟨rfl, congrArg some pointerEq, by simp [renderedWord, valuePointer]⟩
  | true =>
    simp only [ValuesRelated, ite_true] at related ⊢
    obtain ⟨a, b, leftEq, rightEq⟩ := related
    cases left <;> cases right <;> simp_all [valueModel, renderedWord, valuePointer]

theorem source_value_writes_preserve_inputs (dependencies : Dependencies)
    (left right leftAfter rightAfter : V.Registers) (index : U16)
    (leftValue rightValue : V.Value) (leftLayouts rightLayouts : Slice value_flow.AllocationLayout)
    (tainted : Bool) (related : InputBanksRelated dependencies (bankModel left) (bankModel right))
    (leftCanonical : Canonical left.raw.vund.val) (rightCanonical : Canonical right.raw.vund.val)
    (leftRun : value_flow.write_value left index leftValue leftLayouts = .ok (.Ok (), leftAfter))
    (rightRun : value_flow.write_value right index rightValue rightLayouts = .ok (.Ok (), rightAfter))
    (values : ValuesRelated tainted (valueModel leftValue) (valueModel rightValue)) :
    InputBanksRelated (virtualDependencies dependencies index.val tainted)
      (bankModel leftAfter) (bankModel rightAfter) := by
  obtain ⟨leftBits, leftRendered, leftEffect⟩ :=
    source_write_value_success left leftAfter index leftValue leftLayouts leftCanonical leftRun
  obtain ⟨rightBits, rightRendered, rightEffect⟩ :=
    source_write_value_success right rightAfter index rightValue rightLayouts rightCanonical rightRun
  have wordRelated := rendered_words_related leftValue rightValue leftBits rightBits leftLayouts rightLayouts
    tainted leftRendered rightRendered values
  constructor
  · intro query metadata tracked
    by_cases selected : query = index.val
    · subst query
      have sameTaint : metadata = tainted := by
        simpa only [virtualDependencies, ite_true, Option.some.injEq] using tracked.symm
      subst metadata
      exact ⟨renderedWord leftBits leftValue, renderedWord rightBits rightValue,
        leftEffect.1, rightEffect.1, rfl, rfl, wordRelated⟩
    · have oldTracked : dependencies.virtual query = some metadata := by
        simpa only [virtualDependencies, selected, ite_false] using tracked
      obtain ⟨l, r, leftWord, rightWord, ld, rd, words⟩ := related.virtuals query metadata oldTracked
      refine ⟨l, r, ?_, ?_, ld, rd, words⟩
      · simpa only [bankModel, leftEffect.2.1 query selected] using leftWord
      · simpa only [bankModel, rightEffect.2.1 query selected] using rightWord
  · intro query metadata tracked
    obtain ⟨l, r, leftPredicate, rightPredicate, same⟩ := related.predicates query metadata tracked
    refine ⟨l, r, ?_, ?_, same⟩
    · simpa only [bankModel, leftEffect.2.2.1 query] using leftPredicate
    · simpa only [bankModel, rightEffect.2.2.1 query] using rightPredicate

theorem source_predicate_writes_preserve_inputs (dependencies : Dependencies)
    (left right leftAfter rightAfter : V.Registers) (index : U16)
    (leftValue rightValue tainted : Bool)
    (related : InputBanksRelated dependencies (bankModel left) (bankModel right))
    (leftRun : value_flow.write_predicate left index leftValue = .ok (.Ok (), leftAfter))
    (rightRun : value_flow.write_predicate right index rightValue = .ok (.Ok (), rightAfter))
    (values : tainted = false → leftValue = rightValue) :
    InputBanksRelated (predicateDependencies dependencies index.val tainted)
      (bankModel leftAfter) (bankModel rightAfter) := by
  have leftEffect := source_write_predicate_success left leftAfter index leftValue leftRun
  have rightEffect := source_write_predicate_success right rightAfter index rightValue rightRun
  constructor
  · intro query metadata tracked
    obtain ⟨l, r, leftWord, rightWord, ld, rd, words⟩ := related.virtuals query metadata tracked
    refine ⟨l, r, ?_, ?_, ld, rd, words⟩
    · simpa only [bankModel, leftEffect.2 query] using leftWord
    · simpa only [bankModel, rightEffect.2 query] using rightWord
  · intro query metadata tracked
    by_cases selected : query = index.val
    · subst query
      have sameTaint : metadata = tainted := by
        simpa only [predicateDependencies, ite_true, Option.some.injEq] using tracked.symm
      subst metadata
      refine ⟨leftValue, rightValue, ?_, ?_, values⟩
      · simpa only [bankModel, ite_true] using leftEffect.1 index.val
      · simpa only [bankModel, ite_true] using rightEffect.1 index.val
    · have oldTracked : dependencies.predicate query = some metadata := by
        simpa only [predicateDependencies, selected, ite_false] using tracked
      obtain ⟨l, r, leftPredicate, rightPredicate, same⟩ := related.predicates query metadata oldTracked
      refine ⟨l, r, ?_, ?_, same⟩
      · simpa only [bankModel, leftEffect.1 query, selected, ite_false] using leftPredicate
      · simpa only [bankModel, rightEffect.1 query, selected, ite_false] using rightPredicate

end ValueFlow.Source
