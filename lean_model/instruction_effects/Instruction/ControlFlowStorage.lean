/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ControlFlowGraphRefinement

open Aeneas Aeneas.Std PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

namespace Instruction.ControlFlow

def TargetStorage (operation : Op) : Prop :=
  match operation with
  | .Brx _ _ targets => targets.val.length < Usize.max
  | _ => True

theorem return_targets_length (context : Context)
    (count : context.count < Usize.max)
    (returns : ∀ row, context.returns = some row → row.length < Usize.max) :
    (returnTargets context).length < Usize.max := by
  unfold returnTargets
  cases found : context.returns with
  | none => simpa only [found, Option.getD_none, List.length_range] using count
  | some row => simpa only [found, Option.getD_some] using returns row found

theorem successors_fit_storage (operation : Op) (guard : Pred) (context : Context)
    (count : context.count < Usize.max) (storage : TargetStorage operation)
    (returns : ∀ row, context.returns = some row → row.length < Usize.max) :
    (successors operation guard context).length ≤ Usize.max := by
  have nextLength := next_length context
  have returnLength := return_targets_length context count returns
  have scalarSpace : 2 ≤ Usize.max := by scalar_tac
  cases operation <;> simp only [successors, TargetStorage] at storage ⊢
  all_goals repeat' split
  all_goals try contradiction
  all_goals try simp only [List.length_cons, List.length_append, List.length_map, List.length_nil]
  all_goals omega

theorem graph_rows_fit_storage (code : List Inst) (cut : U8) (fold : List (Option Bool))
    (returns : List (alloc.vec.Vec Usize)) (count : code.length < Usize.max)
    (targets : ∀ instruction ∈ code, TargetStorage instruction.op)
    (returnStorage : ∀ row ∈ returns, row.val.length < Usize.max) :
    ∀ pc < code.length, (graphRows code cut fold returns)[pc]!.length ≤ Usize.max := by
  intro pc inside
  rw [graph_row_at code cut fold returns pc inside]
  apply successors_fit_storage _ _ _ count (targets _ (List.getElem_mem inside))
  intro row found
  change (returns[pc]?.map (fun values => values.val.map UScalar.val)) = some row at found
  cases actual : returns[pc]? with
  | none => simp only [actual, Option.map_none, reduceCtorEq] at found
  | some values =>
      simp only [actual, Option.map_some, Option.some.injEq] at found
      rw [←found, List.length_map]
      exact returnStorage values (List.mem_of_getElem? actual)

end Instruction.ControlFlow
