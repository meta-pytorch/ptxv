/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Construction

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

private theorem option_none {α : Type} (value : Option α) :
    core.option.Option.is_none value = true ↔ value = none := by
  cases value <;> simp [core.option.Option.is_none]

private theorem empty_vector {α : Type} (value : alloc.vec.Vec α) :
    alloc.vec.Vec.len value = 0#usize ↔ value.val = [] := by
  constructor
  · intro empty
    cases h : value.val with
    | nil => rfl
    | cons first rest => scalar_tac
  · intro empty
    scalar_tac

private theorem accept_positive (condition : Prop) [Decidable condition] (next : Result Bool)
    (computed : (if condition then next else .ok false) = .ok true) :
    condition ∧ next = .ok true := by
  by_cases h : condition
  · exact ⟨h, by simpa only [h, ite_true] using computed⟩
  · simp only [h, ite_false, ok.injEq, Bool.false_eq_true] at computed

private theorem accept_negative (condition : Prop) [Decidable condition] (next : Result Bool)
    (computed : (if condition then .ok false else next) = .ok true) :
    ¬condition ∧ next = .ok true := by
  by_cases h : condition
  · simp only [h, ite_true, ok.injEq, Bool.false_eq_true] at computed
  · exact ⟨h, by simpa only [h, ite_false] using computed⟩

theorem source_manifest_supported (manifest : ptxv_case.Manifest)
    (computed : strict_memory_frontend_launch.manifest_supported manifest = .ok true) :
    SupportedManifest manifest := by
  unfold strict_memory_frontend_launch.manifest_supported at computed
  dsimp only at computed
  obtain ⟨gate0, checked1⟩ := accept_positive _ _ computed
  obtain ⟨gate1, checked2⟩ := accept_positive _ _ checked1
  obtain ⟨gate2, checked3⟩ := accept_positive _ _ checked2
  obtain ⟨gate3, checked4⟩ := accept_positive _ _ checked3
  obtain ⟨gate4, checked5⟩ := accept_positive _ _ checked4
  obtain ⟨gate5, checked6⟩ := accept_positive _ _ checked5
  obtain ⟨gate6, checked7⟩ := accept_positive _ _ checked6
  obtain ⟨gate7, checked8⟩ := accept_positive _ _ checked7
  obtain ⟨gate8, checked9⟩ := accept_positive _ _ checked8
  obtain ⟨gate9, checked10⟩ := accept_negative _ _ checked9
  obtain ⟨gate10, checked11⟩ := accept_positive _ _ checked10
  obtain ⟨gate11, checked12⟩ := accept_negative _ _ checked11
  obtain ⟨gate12, checked13⟩ := accept_negative _ _ checked12
  obtain ⟨gate13, checked14⟩ := accept_negative _ _ checked13
  obtain ⟨gate14, checked15⟩ := accept_negative _ _ checked14
  obtain ⟨gate15, checked16⟩ := accept_positive _ _ checked15
  obtain ⟨gate16, checked17⟩ := accept_negative _ _ checked16
  obtain ⟨gate17, checked18⟩ := accept_negative _ _ checked17
  obtain ⟨gate18, checked19⟩ := accept_negative _ _ checked18
  obtain ⟨gate19, checked20⟩ := accept_negative _ _ checked19
  obtain ⟨gate20, checked21⟩ := accept_positive _ _ checked20
  obtain ⟨gate21, checked22⟩ := accept_positive _ _ checked21
  obtain ⟨gate22, checked23⟩ := accept_positive _ _ checked22
  obtain ⟨gate23, checked24⟩ := accept_positive _ _ checked23
  obtain ⟨gate24, checked25⟩ := accept_positive _ _ checked24
  obtain ⟨gate25, checked26⟩ := accept_positive _ _ checked25
  obtain ⟨gate26, checked27⟩ := accept_positive _ _ checked26
  obtain ⟨gate27, checked28⟩ := accept_positive _ _ checked27
  obtain ⟨gate28, checked29⟩ := accept_positive _ _ checked28
  obtain ⟨gate29, checked30⟩ := accept_positive _ _ checked29
  simp only [ok.injEq, decide_eq_true_eq, empty_vector] at checked30
  simp only [option_none, empty_vector] at gate0 gate1 gate2 gate3 gate4 gate5 gate6 gate7 gate8 gate9 gate10 gate11 gate12 gate13 gate14 gate15 gate16 gate17 gate18 gate19 gate20 gate21 gate22 gate23 gate24 gate25 gate26 gate27 gate28 gate29
  simp only [SupportedManifest]
  repeat' apply And.intro
  all_goals first | assumption | scalar_tac

end PtxFrontendProgram
