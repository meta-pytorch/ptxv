/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceReads
import ValueFlow.SourceBankBits

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

open Storage.MachineRegisters Storage.MachineRegistersDefinedness

def PredicateWriteEffect (before after : V.Registers) (index : U16) (value : Bool) : Prop :=
  (∀ query, predicateModel after query =
    if query = index.val then some value else predicateModel before query) ∧
  (∀ query, wordModel after query = wordModel before query)

 theorem source_write_predicate_effect (registers : V.Registers) (index : U16) (value : Bool) :
    value_flow.write_predicate registers index value ⦃ result =>
      result.1 = .Ok () → PredicateWriteEffect registers result.2 index value ⦄ := by
  unfold value_flow.write_predicate
  step with source_writable_predicate registers index as ⟨writable, shape⟩
  cases writable with
  | false => simp
  | true =>
    have bounds : index.val / 64 < registers.raw.preds.val.length ∧
        index.val / 64 < registers.raw.pdef.val.length := by simpa using shape.symm
    simp only [ite_true]
    step with source_write_predicate registers.raw index value true bounds.1 bounds.2 as
      ⟨raw, words, written, undefined, predicates, definitions⟩
    have predLength : raw.preds.val.length = registers.raw.preds.val.length := by
      rw [predicates]; simp
    have defLength : raw.pdef.val.length = registers.raw.pdef.val.length := by
      rw [definitions]; simp
    intro _
    constructor
    · intro query
      rw [predicateModel_bits, predicateModel_bits]
      simp only [predLength, defLength]
      by_cases selected : query = index.val
      · subst query
        simp only [bounds, and_self, ite_true]
        have predBit : bitAt raw.preds.val index.val = value := by
          rw [predicates]
          simp [bitAt, bounds.1, bit_word_selected]
        have defBit : bitAt raw.pdef.val index.val = true := by
          rw [definitions]
          simp [bitAt, bounds.2, bit_word_selected]
        rw [predBit, defBit]
        rfl
      · have predFrame := bit_update_frame registers.raw.preds.val index value bounds.1 query selected
        have defFrame := bit_update_frame registers.raw.pdef.val index true bounds.2 query selected
        simp only [←predicates] at predFrame
        simp only [←definitions] at defFrame
        simp only [selected, ite_false, predFrame, defFrame]
    · intro query
      simp only [wordModel, words, written, undefined]

 theorem source_write_predicate_success (before after : V.Registers) (index : U16) (value : Bool)
    (run : value_flow.write_predicate before index value = .ok (.Ok (), after)) :
    PredicateWriteEffect before after index value := by
  have spec := source_write_predicate_effect before index value
  rw [run] at spec
  have implication : True → PredicateWriteEffect before after index value := by
    simpa only [WP.spec_ok] using spec
  exact implication True.intro

end ValueFlow.Source
