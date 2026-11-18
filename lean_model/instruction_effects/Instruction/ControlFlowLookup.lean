/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ControlFlowGraphModel

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace Instruction.ControlFlow

@[step]
theorem rust_fold_at (fold : Slice (Option Bool)) (pc : Usize) :
    (if pc < Slice.len fold then Slice.index_usize fold pc else .ok none)
      ⦃ result => result = fold.val[pc.val]?.getD none ⦄ := by
  by_cases inside : pc.val < fold.val.length
  · have test : pc < Slice.len fold := by scalar_tac
    simp only [test, ite_true]
    step with Slice.index_usize_spec fold pc inside as ⟨decision, decisionEq⟩
    simp only [List.getElem?_eq_getElem inside, Option.getD_some, decisionEq]
  · have test : ¬pc < Slice.len fold := by scalar_tac
    simp only [test, ite_false, WP.spec_ok, List.getElem?_eq_none (by omega : fold.val.length ≤ pc.val),
      Option.getD_none]

@[step]
theorem rust_returns_at (returns : Slice (alloc.vec.Vec Usize)) (pc : Usize) :
    (if pc < Slice.len returns then
      Std.bind (Slice.index_usize returns pc) (fun row => .ok (some (alloc.vec.Vec.deref row)))
      else .ok none) ⦃ result => result = returns.val[pc.val]?.map alloc.vec.Vec.deref ⦄ := by
  by_cases inside : pc.val < returns.val.length
  · have test : pc < Slice.len returns := by scalar_tac
    simp only [test, ite_true]
    step with Slice.index_usize_spec returns pc inside as ⟨row, rowEq⟩
    simp only [rowEq, List.getElem?_eq_getElem inside, Option.map_some]
  · have test : ¬pc < Slice.len returns := by scalar_tac
    simp only [test, ite_false, WP.spec_ok, List.getElem?_eq_none (by omega : returns.val.length ≤ pc.val),
      Option.map_none]

end Instruction.ControlFlow
