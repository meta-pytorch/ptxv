/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.InstructionPositions

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false
set_option backward.isDefEq.respectTransparency false

namespace PtxFrontendProgram

theorem source_label_matches (label : Label) (token : SourceToken) (pc : Usize)
    (computed : strict_memory_frontend_parse.label_matches label token pc=.ok true) :
    label.«name»=token ∧ label.target.val=pc.val := by
  unfold strict_memory_frontend_parse.label_matches at computed
  split at computed
  · rename_i startEq
    split at computed
    · rename_i endEq
      split at computed
      · rename_i wordEq
        simp only [lift,bind_ok,ok.injEq,decide_eq_true_eq] at computed
        have castValue : (UScalar.cast .Usize label.target).val=label.target.val := by
          simp only [UScalar.cast_val_eq]
          have targetBound := label.target.hBounds
          rw [Nat.mod_eq_of_lt (by scalar_tac)]
        have targetEq : label.target.val=pc.val := by rw [computed] at castValue;exact castValue.symm
        refine ⟨?_,targetEq⟩
        rcases label with ⟨⟨leftStart,leftEnd,leftWord⟩,target⟩
        rcases token with ⟨rightStart,rightEnd,rightWord⟩
        simp only at startEq endEq wordEq
        cases startEq
        cases endEq
        cases wordEq
        rfl
      · simp at computed
    · simp at computed
  · simp at computed

end PtxFrontendProgram
