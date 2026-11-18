/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.InstructionHeads
import FrontendProgram.InstructionUnique

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192

namespace PtxFrontendProgram

theorem code_syntax_unique (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram)
    (meaning : ModuleMeaning source entry program)
    (tokens : List SourceToken) (start pc leftFinish rightFinish : Nat)
    (leftCode rightCode : List GuardedInstruction) (leftLabels rightLabels : List Label)
    (first : CodeSyntax source program.scope tokens start pc leftCode leftLabels leftFinish)
    (second : CodeSyntax source program.scope tokens start pc rightCode rightLabels rightFinish) :
    leftFinish=rightFinish ∧ leftLabels=rightLabels ∧ leftCode=rightCode := by
  induction first generalizing rightCode rightLabels rightFinish with
  | @empty position pc firstClose =>
    cases second with
    | empty => exact ⟨rfl,rfl,rfl⟩
    | label target selected valid colon targetEq tail =>
      exact False.elim (identifier_at_not_text source tokens position _ "}" 125#u8 (by decide) (by norm_num [InitialCharacter])
        selected valid firstClose)
    | instruction head tail =>
      exact False.elim (instruction_start_not_text source program.scope tokens position _ _ "}" 125#u8
        (by decide) (by decide) (by decide) head firstClose)
  | @label start pc code labels finish labelToken target selected valid colon targetEq tail ih =>
    cases second with
    | empty rightClose =>
      exact False.elim (identifier_at_not_text source tokens start labelToken "}" 125#u8 (by decide) (by norm_num [InitialCharacter])
        selected valid rightClose)
    | label otherTarget otherSelected otherValid otherColon otherTargetEq otherTail =>
      have sameName:=Option.some.inj (selected.symm.trans otherSelected)
      have sameTarget : target=otherTarget:=UScalar.eq_of_val_eq (targetEq.trans otherTargetEq.symm)
      subst otherTarget
      cases sameName
      obtain ⟨sameFinish,sameLabels,sameLength⟩:=ih _ _ _ otherTail
      exact ⟨sameFinish,congrArg (List.cons ⟨labelToken,target⟩) sameLabels,sameLength⟩
    | instruction head otherTail =>
      exact False.elim (instruction_head_not_label source entry program meaning tokens start _ _ labelToken
        selected valid colon head)
  | @instruction start pc next code labels finish instruction head tail ih =>
    cases second with
    | empty rightClose =>
      exact False.elim (instruction_start_not_text source program.scope tokens start _ _ "}" 125#u8
        (by decide) (by decide) (by decide) head rightClose)
    | label otherTarget otherSelected otherValid otherColon otherTargetEq otherTail =>
      exact False.elim (instruction_head_not_label source entry program meaning tokens start _ _ _
        otherSelected otherValid otherColon head)
    | instruction otherHead otherTail =>
      obtain ⟨sameNext,sameInstruction⟩:=module_instruction_syntax_unique source entry program meaning tokens start
        next _ instruction _ head otherHead
      cases sameNext
      cases sameInstruction
      obtain ⟨sameFinish,sameLabels,sameCode⟩:=ih _ _ _ otherTail
      exact ⟨sameFinish,sameLabels,congrArg (List.cons instruction) sameCode⟩

end PtxFrontendProgram
