/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceApplication

open Aeneas Aeneas.Std Result ControlFlow
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

theorem source_loop_success {α β : Type} (body : α → Result (ControlFlow α β))
    (measure : α → Nat) (invariant : α → Prop) (post : β → Prop)
    (bodySound : ∀ before flow, invariant before → body before = .ok flow →
      match flow with
      | .cont after => invariant after ∧ measure after < measure before
      | .done output => post output)
    (before : α) (output : β) (valid : invariant before)
    (computed : loop body before = .ok output) : post output := by
  generalize measured : measure before = count
  induction count using Nat.strong_induction_on generalizing before with
  | h count ih =>
    rw [loop.eq_def] at computed
    obtain ⟨flow,first,remaining⟩ := source_bind_success _ _ _ computed
    have sound := bodySound before flow valid first
    cases flow with
    | done result =>
      have equal : result = output := by simpa only [ok.injEq] using remaining
      simpa only [equal] using sound
    | cont after =>
      exact ih (measure after) (by omega) after sound.1 remaining rfl

theorem source_loop_map {α β γ : Type} (leftBody : α → Result (ControlFlow α β))
    (rightBody : α → Result (ControlFlow α γ)) (measure : α → Nat) (mapping : β → γ)
    (step : ∀ before flow,leftBody before=.ok flow →
      rightBody before=.ok (match flow with | .cont after=>.cont after | .done output=>.done (mapping output)) ∧
      (∀ after,flow=.cont after → measure after<measure before))
    (before : α) (output : β) (computed : loop leftBody before=.ok output) :
    loop rightBody before=.ok (mapping output) := by
  generalize measured : measure before=count
  induction count using Nat.strong_induction_on generalizing before with
  | h count ih =>
    rw [loop.eq_def] at computed ⊢
    obtain ⟨flow,first,remaining⟩:=source_bind_success _ _ _ computed
    obtain ⟨rightFirst,decreases⟩:=step before flow first
    rw [rightFirst]
    cases flow with
    | done result =>
      have equal : result=output := by simpa only [ok.injEq] using remaining
      simpa only [bind_ok,equal]
    | cont after =>
      simp only [bind_ok]
      exact ih (measure after) (by have bound:=decreases after rfl;omega) after remaining rfl

end PtxFrontendProgram
