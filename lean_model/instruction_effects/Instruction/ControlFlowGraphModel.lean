/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ControlFlowRefinement

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false

namespace Instruction.ControlFlow

def rowContext (count : Nat) (cut : U8) (fold : List (Option Bool))
    (returns : List (alloc.vec.Vec Usize)) (pc : Nat) : Context :=
  ⟨pc, count, cut, fold[pc]?.getD none, returns[pc]?.map (fun row => row.val.map UScalar.val)⟩

def graphRows (code : List Inst) (cut : U8) (fold : List (Option Bool))
    (returns : List (alloc.vec.Vec Usize)) : List (List Nat) :=
  code.mapIdx (fun pc instruction =>
    successors instruction.op instruction.guard (rowContext code.length cut fold returns pc))

@[simp]
theorem graph_rows_length (code : List Inst) (cut : U8) (fold : List (Option Bool))
    (returns : List (alloc.vec.Vec Usize)) : (graphRows code cut fold returns).length = code.length := by
  simp only [graphRows, List.length_mapIdx]

theorem graph_row_at (code : List Inst) (cut : U8) (fold : List (Option Bool))
    (returns : List (alloc.vec.Vec Usize)) (pc : Nat) (inside : pc < code.length) :
    (graphRows code cut fold returns)[pc]! =
      successors code[pc].op code[pc].guard (rowContext code.length cut fold returns pc) := by
  have outputBound : pc < (graphRows code cut fold returns).length := by simpa only [graph_rows_length] using inside
  rw [←List.Inhabited_getElem_eq_getElem! _ _ outputBound]
  exact List.getElem_mapIdx

end Instruction.ControlFlow
