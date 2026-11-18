/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MaskedLoadProducer

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
namespace Instruction

def assignLoadWord {RegisterKey : Type} [DecidableEq RegisterKey]
    (project : Key → RegisterKey) (state : RegisterKey → BitVec 32) (action : LoadObservation) :
    RegisterKey → BitVec 32 :=
  fun register => if register = project action.destination then action.value else state register

theorem load_fold_preserves_frame {RegisterKey : Type} [DecidableEq RegisterKey]
    (project : Key → RegisterKey) (state : RegisterKey → BitVec 32)
    (actions : List LoadObservation) (register : RegisterKey)
    (hframe : register ∉ actions.map (project ∘ LoadObservation.destination)) :
    (actions.foldl (assignLoadWord project) state) register = state register := by
  induction actions generalizing state with
  | nil => rfl
  | cons action actions ih =>
    have hnot : register ≠ project action.destination := by
      intro heq
      apply hframe
      simp [heq]
    have hrest : register ∉ actions.map (project ∘ LoadObservation.destination) := by
      intro hm
      apply hframe
      exact List.mem_cons.mpr (Or.inr hm)
    rw [List.foldl_cons, ih _ hrest]
    simp [assignLoadWord, hnot]

@[step] theorem actual_masked_load_preserves_frame {Writer RegisterKey : Type}
    [DecidableEq RegisterKey] (writerInst : masked_load.RegisterWriter Writer)
    (operation : Op) (data : Array U8 32#usize) (writer : Writer)
    (observe : Writer → RegisterKey → BitVec 32) (valid : Writer → Prop)
    (project : Key → RegisterKey) (hvalid : valid writer)
    (hbounded : Bounded ((maskedLoadObservations operation (loadedMemoryWord data)).map
      LoadObservation.destination))
    (hwrite : LoadWriterContract writerInst observe valid (assignLoadWord project)
      (maskedLoadObservations operation (loadedMemoryWord data)))
    (register : RegisterKey)
    (hframe : register ∉ (maskedLoadObservations operation (loadedMemoryWord data)).map
      (project ∘ LoadObservation.destination)) :
    masked_load.execute writerInst operation data writer ⦃ updated =>
      valid updated ∧ observe updated register = observe writer register ⦄ := by
  step with actual_masked_load_execute_spec writerInst operation data writer observe valid
    (assignLoadWord project) hvalid hbounded hwrite as ⟨updated, hupdatedValid, hupdated⟩
  refine ⟨hupdatedValid, ?_⟩
  rw [hupdated]
  exact load_fold_preserves_frame project (observe writer) _ register hframe

end Instruction
