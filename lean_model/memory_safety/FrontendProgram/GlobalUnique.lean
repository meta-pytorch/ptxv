/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SyntaxFacts

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192

namespace PtxFrontendProgram

private theorem initializer_nonempty (source : List U8) (tokens : List SourceToken)
    (width start finish : Nat) (values : List Nat)
    (meaning : InitializerSyntax source tokens width start values finish) : values ≠ [] := by
  cases meaning <;> simp

theorem initializer_syntax_unique (source : List U8) (tokens : List SourceToken)
    (width start leftFinish rightFinish : Nat) (left right : List Nat)
    (first : InitializerSyntax source tokens width start left leftFinish)
    (second : InitializerSyntax source tokens width start right rightFinish)
    (length : left.length = right.length) : left = right ∧ leftFinish = rightFinish := by
  induction first generalizing right rightFinish with
  | single firstNumber firstBound =>
    cases second with
    | single secondNumber secondBound =>
      have same := number_at_unique source tokens _ _ _ firstNumber secondNumber
      subst_vars
      exact ⟨rfl, rfl⟩
    | cons _ _ _ tail =>
      cases tail <;> simp_all
  | cons firstNumber firstBound comma tail ih =>
    cases second with
    | single _ _ =>
      cases tail <;> simp_all
    | cons secondNumber secondBound otherComma otherTail =>
      have same := number_at_unique source tokens _ _ _ firstNumber secondNumber
      obtain ⟨sameTail, sameFinish⟩ := ih _ _ otherTail (by simpa only [List.length_cons, Nat.add_right_cancel_iff] using length)
      subst_vars
      exact ⟨rfl, rfl⟩

theorem global_element_type_unique (source : List U8) (token : SourceToken) (left right : Nat)
    (first : GlobalElementType source token left)
    (second : GlobalElementType source token right) : left = right := by
  obtain ⟨leftText, leftListed, leftSpelling⟩ := first
  obtain ⟨rightText, rightListed, rightSpelling⟩ := second
  apply table_denotation_unique _ (by decide) leftText rightText left right leftListed rightListed
  exact token_text_unique source token _ _ leftSpelling rightSpelling

theorem global_syntax_unique (source : List U8) (tokens : List SourceToken)
    (start leftFinish rightFinish : Nat) (left right : Global)
    (first : GlobalSyntax source tokens start leftFinish left)
    (second : GlobalSyntax source tokens start rightFinish right) :
    left = right ∧ leftFinish = rightFinish := by
  cases first with
  | @global middle _ width alignment count values globalName typeToken _
      globalToken alignToken alignmentNumber alignmentPower wideEnough typeSelected elementType
      nameSelected identifier bracket countNumber countPositive closingBracket equal opening initializers countLength
      closing semicolon leftEnd leftName leftAlignment leftBytes =>
    cases second with
    | @global otherMiddle _ otherWidth otherAlignment otherCount otherValues otherName otherTypeToken _
        otherGlobalToken otherAlignToken otherAlignmentNumber otherAlignmentPower otherWideEnough otherTypeSelected otherElementType
        otherNameSelected otherIdentifier otherBracket otherCountNumber otherCountPositive otherClose otherEqual otherOpen otherInitializers otherCountLength
        otherClosing otherSemicolon rightEnd rightName rightAlignment rightBytes =>
      have sameType : typeToken = otherTypeToken := Option.some.inj (typeSelected.symm.trans otherTypeSelected)
      subst otherTypeToken
      have sameWidth := global_element_type_unique source typeToken width otherWidth elementType otherElementType
      subst otherWidth
      have sameAlignment := number_at_unique source tokens _ alignment otherAlignment alignmentNumber otherAlignmentNumber
      have sameCount := number_at_unique source tokens _ count otherCount countNumber otherCountNumber
      have sameName : globalName = otherName := Option.some.inj (nameSelected.symm.trans otherNameSelected)
      have sameLength : values.length = otherValues.length := by omega
      obtain ⟨sameValues, sameMiddle⟩ := initializer_syntax_unique source tokens width _ middle otherMiddle values otherValues
        initializers otherInitializers sameLength
      have nameEq : left.name = right.name := leftName.trans (sameName.trans rightName.symm)
      have alignmentEq : left.alignment = right.alignment := UScalar.eq_of_val_eq (leftAlignment.trans (sameAlignment.trans rightAlignment.symm))
      have bytesEq : left.bytes = right.bytes := by
        simpa only [alloc.vec.Vec.eq_iff] using (show left.bytes.val = right.bytes.val by rw [leftBytes, rightBytes, sameValues])
      constructor
      · cases left
        cases right
        simp_all
      · omega

theorem entry_syntax_unique (source entry : List U8) (tokens : List SourceToken)
    (start leftFinish rightFinish : Nat) (leftName rightName : SourceToken)
    (first : EntrySyntax source entry tokens start leftFinish leftName)
    (second : EntrySyntax source entry tokens start rightFinish rightName) :
    leftFinish = rightFinish ∧ leftName = rightName := by
  induction first generalizing rightFinish rightName with
  | entry keyword selected identifier named opening closingBracket body =>
    cases second with
    | entry _ otherSelected => exact ⟨rfl, Option.some.inj (selected.symm.trans otherSelected)⟩
    | visible otherKeyword =>
      have impossible := token_at_spelling_unique source tokens _ ".entry" ".visible" keyword otherKeyword
      exact False.elim ((by decide : utf8 ".entry" ≠ utf8 ".visible") impossible)
  | visible keyword tail ih =>
    cases second with
    | entry otherKeyword =>
      have impossible := token_at_spelling_unique source tokens _ ".visible" ".entry" keyword otherKeyword
      exact False.elim ((by decide : utf8 ".visible" ≠ utf8 ".entry") impossible)
    | visible _ otherTail => exact ih _ _ otherTail

private theorem entry_not_global (source entry : List U8) (tokens : List SourceToken)
    (start finish : Nat) (globalName : SourceToken) (meaning : EntrySyntax source entry tokens start finish globalName)
    (global : TokenAt source tokens start ".global") : False := by
  cases meaning with
  | entry keyword =>
    have impossible := token_at_spelling_unique source tokens _ ".entry" ".global" keyword global
    exact (by decide : utf8 ".entry" ≠ utf8 ".global") impossible
  | visible keyword =>
    have impossible := token_at_spelling_unique source tokens _ ".visible" ".global" keyword global
    exact (by decide : utf8 ".visible" ≠ utf8 ".global") impossible

theorem globals_before_entry_unique (source entry : List U8) (tokens : List SourceToken)
    (start leftFinish rightFinish leftEntryEnd rightEntryEnd : Nat)
    (leftGlobals rightGlobals : List Global) (leftName rightName : SourceToken)
    (firstGlobals : GlobalsSyntax source tokens start leftGlobals leftFinish)
    (secondGlobals : GlobalsSyntax source tokens start rightGlobals rightFinish)
    (firstEntry : EntrySyntax source entry tokens leftFinish leftEntryEnd leftName)
    (secondEntry : EntrySyntax source entry tokens rightFinish rightEntryEnd rightName) :
    leftGlobals = rightGlobals ∧ leftFinish = rightFinish := by
  induction firstGlobals generalizing rightGlobals rightFinish with
  | empty =>
    cases secondGlobals with
    | empty => exact ⟨rfl, rfl⟩
    | cons head tail =>
      cases head with
      | global _ keyword => exact False.elim (entry_not_global source entry tokens _ _ _ firstEntry keyword)
  | cons head tail ih =>
    cases secondGlobals with
    | empty =>
      cases head with
      | global _ keyword => exact False.elim (entry_not_global source entry tokens _ _ _ secondEntry keyword)
    | cons otherHead otherTail =>
      obtain ⟨sameHead, sameMiddle⟩ := global_syntax_unique source tokens _ _ _ _ _ head otherHead
      subst_vars
      obtain ⟨sameTail, sameFinish⟩ := ih _ _ otherTail firstEntry secondEntry
      exact ⟨congrArg _ sameTail, sameFinish⟩

end PtxFrontendProgram
