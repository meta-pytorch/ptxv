/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlowDependencies.SourceTypes
import MemorySafety.SourceCertificate

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlowDependencies.Source

def comparisonBody {α : Type} [DecidableEq α] (left right : List α)
    (equal : Bool) (index : Usize) : Result (ControlFlow (Bool × Usize) Bool) := do
  if index.val < left.length then
    let next ← index + 1#usize
    .ok (.cont (equal && decide (left[index.val]? = right[index.val]?), next))
  else .ok (.done equal)

def comparisonLoop {α : Type} [DecidableEq α] (left right : List α)
    (equal : Bool) (index : Usize) : Result Bool :=
  loop (fun state => comparisonBody left right state.1 state.2) (equal, index)

theorem comparison_loop_exact {α : Type} [DecidableEq α] (left right : List α)
    (bounded : left.length < 2 ^ UScalarTy.Usize.numBits) (sameLength : left.length = right.length)
    (equal : Bool) (index : Usize) :
    comparisonLoop left right equal index =
      .ok (equal && decide (left.drop index.val = right.drop index.val)) := by
  generalize fuelEq : left.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing equal index with
  | h fuel ih =>
    have unroll : comparisonLoop left right equal index = (do
        let output ← comparisonBody left right equal index
        match output with
        | .cont pair => comparisonLoop left right pair.1 pair.2
        | .done value => .ok value) := by
      unfold comparisonLoop
      rw [loop.eq_def]
      congr 1
      funext output
      cases output with
      | cont pair => cases pair; rfl
      | done value => rfl
    rw [unroll]
    by_cases inside : index.val < left.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by omega
      let next : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : next.val = index.val + 1 := by simp [next]
      have increment := MemorySafety.Source.usize_increment index bound
      simp only [comparisonBody, inside, ite_true, increment, bind_ok]
      change comparisonLoop left right
          (equal && decide (left[index.val]? = right[index.val]?)) next = _
      rw [ih (left.length - next.val) (by omega) _ next rfl]
      simp only [nextVal, ok.injEq]
      cases equal with
      | false => simp
      | true =>
        simp only [Bool.true_and]
        apply Bool.eq_iff_iff.mpr
        have rightInside : index.val < right.length := by omega
        simp only [Bool.and_eq_true, decide_eq_true_eq,
          List.getElem?_eq_getElem inside, List.getElem?_eq_getElem rightInside, Option.some.injEq]
        rw [List.drop_eq_getElem_cons inside, List.drop_eq_getElem_cons rightInside]
        simp only [List.cons.injEq]
    · have rightOutside : right.length ≤ index.val := by omega
      simp [comparisonBody, inside, List.drop_eq_nil_of_le (show left.length ≤ index.val by omega),
        List.drop_eq_nil_of_le rightOutside]

theorem comparison_loop_from_body {α : Type} [DecidableEq α] (left right : List α)
    (bounded : left.length < 2 ^ UScalarTy.Usize.numBits) (sameLength : left.length = right.length)
    (body : Bool × Usize → Result (ControlFlow (Bool × Usize) Bool))
    (bodies : ∀ equal index, body (equal, index) = comparisonBody left right equal index)
    (equal : Bool) (index : Usize) :
    loop body (equal, index) = .ok (equal && decide (left.drop index.val = right.drop index.val)) := by
  have same : body = (fun state => comparisonBody left right state.1 state.2) := by
    funext state
    cases state with
    | mk equal index => exact bodies equal index
  rw [same]
  exact comparison_loop_exact left right bounded sameLength equal index

end ValueFlowDependencies.Source
