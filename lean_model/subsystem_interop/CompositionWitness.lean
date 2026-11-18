/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.CertifiedNativeQuery
import Lean.Util.CollectAxioms

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.GuardCertificate
set_option autoImplicit false
namespace Instruction.ControlFlow

private def falseGuard : Pred := ⟨some ⟨.P, 0#u16⟩, false⟩
private def wrongFold : control_flow.Context :=
  ⟨0#usize, 3#usize, 0#u8, some true, none⟩

/-- An unchecked false fold can erase an actual edge in the real source classifier. -/
theorem wrong_fold_source_drops_false_guard_fallthrough :
    control_flow.successors (.Bra 2#usize false false false none none) falseGuard wrongFold
      ⦃ row => 1 ∉ row.val.map UScalar.val ∧
        1 ∈ semanticEdges (.Bra 2#usize false false false none none) falseGuard false
          (contextView wrongFold) ⦄ := by
  have source := rust_successors (.Bra 2#usize false false false none none) falseGuard wrongFold
    (by simp [successors, contextView, wrongFold, falseGuard, always]; scalar_tac)
  apply WP.spec_mono source
  intro row meaning
  have meaning' : row.val.map UScalar.val = [2] := by
    simpa [RowPost, successors, contextView, wrongFold, falseGuard, always] using meaning
  constructor
  · simp [meaning']
  · simp [semanticEdges, effectiveGuard, successors, contextView, wrongFold, falseGuard, always, next]

end Instruction.ControlFlow

open Lean Elab Command
run_cmd do
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let assumptions ← collectAxioms ``Instruction.ControlFlow.wrong_fold_source_drops_false_guard_fallthrough
  for assumption in assumptions do
    unless assumption ∈ allowed do
      throwError "Unexpected witness assumption {assumption}"
  logInfo m!"Composition witness audited: {assumptions}"
