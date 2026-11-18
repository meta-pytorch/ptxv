/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result
open Lean.Order

set_option autoImplicit false

namespace MemoryWrite

universe u v

theorem bind_success {α : Type u} {β : Type u} (computation : Result α)
    (continuation : α → Result β) (result : β)
    (success : Aeneas.Std.bind computation continuation = .ok result) :
    ∃ intermediate, computation = .ok intermediate ∧ continuation intermediate = .ok result := by
  cases computation with
  | ret intermediate => exact ⟨intermediate, rfl, by simpa only [bind_ok] using success⟩
  | div => simp only [bind_div, div_not_ok] at success
  | vis effect next => simp only [bind_vis, vis_not_ok] at success

section
unseal Result

theorem success_post_admissible {α : Type u} (post : α → Prop) :
    admissible (fun computation : Result α => ∀ result, computation = .ok result → post result) := by
  intro chain isChain allHold result returned
  have member := Aeneas.Data.Coinductive.ITree.csup_ret_mem isChain returned
  exact allHold _ member result rfl

end

inductive LoopExecution {α : Type u} {β : Type u}
    (body : α → Result (ControlFlow α β)) : α → β → Prop
  | done {state : α} {result : β} : body state = .ok (.done result) →
      LoopExecution body state result
  | cont {state next : α} {result : β} : body state = .ok (.cont next) →
      LoopExecution body next result → LoopExecution body state result

theorem successful_loop_has_execution {α : Type u} {β : Type u}
    (body : α → Result (ControlFlow α β)) (state : α) (result : β)
    (success : loop body state = .ok result) : LoopExecution body state result := by
  have allStates : ∀ state result, loop body state = .ok result → LoopExecution body state result := by
    delta loop
    apply fix_induct _ (fun recursive : α → Result β => ∀ state result,
      recursive state = .ok result → LoopExecution body state result)
    · apply admissible_pi
      intro state
      exact admissible_apply (fun (_ : α) (computation : Result β) =>
        ∀ result, computation = .ok result → LoopExecution body state result) state
        (success_post_admissible (LoopExecution body state))
    · intro recursive ih state result success
      obtain ⟨flow, bodySuccess, nextSuccess⟩ := bind_success (body state) _ result success
      cases flow with
      | done value =>
          have equal : value = result := Result.ok_injective nextSuccess
          subst value
          exact .done bodySuccess
      | cont next => exact .cont bodySuccess (ih next result nextSuccess)
  exact allStates state result success

theorem LoopExecution.preserves {α : Type u} {β : Type u}
    {body : α → Result (ControlFlow α β)} {state : α} {result : β}
    (execution : LoopExecution body state result) (invariant : α → Prop) (post : β → Prop)
    (initial : invariant state)
    (step : ∀ state, invariant state → ∀ flow, body state = .ok flow →
      match flow with
      | .done result => post result
      | .cont next => invariant next) : post result := by
  induction execution with
  | done success => exact step _ initial _ success
  | cont success _ ih => exact ih (step _ initial _ success)

end MemoryWrite
