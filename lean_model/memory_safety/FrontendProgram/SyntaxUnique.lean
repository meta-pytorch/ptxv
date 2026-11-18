/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.MeaningUnique

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192

namespace PtxFrontendProgram

theorem token_at_spelling_unique (source : List U8) (tokens : List SourceToken) (position : Nat)
    (left right : String) (first : TokenAt source tokens position left)
    (second : TokenAt source tokens position right) : utf8 left=utf8 right := by
  obtain ⟨leftToken,leftSelected,leftSpelled⟩:=first
  obtain ⟨rightToken,rightSelected,rightSpelled⟩:=second
  have sameToken:=Option.some.inj (leftSelected.symm.trans rightSelected)
  subst rightToken
  exact token_text_unique source leftToken _ _ leftSpelled rightSpelled

theorem module_operand_not_minus (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (token : SourceToken) (bits : Nat) (operand : Operand)
    (interpreted : OperandMeaning source program.scope token bits operand)
    (spelled : TokenSpelling source token "-") : False := by
  have minus : source[token.start.val]?=some 45#u8 := by
    have point:=spelled.2.2.2 0 (by decide)
    have actual : (utf8 "-")[0]?=some 45#u8 := by decide
    simpa only [Nat.add_zero] using point.trans actual
  cases interpreted with
  | immediate value number =>
    obtain ⟨byte,selected,lower,upper⟩:=numeral_first source token.start.val token.end.val value.val number
    have same : byte=45#u8 := Option.some.inj (selected.symm.trans minus)
    subst byte
    norm_num at lower
  | register index reference =>
    have percent:=module_register_first source entry program meaning token bits false index reference
    have impossible : 37#u8=45#u8 := Option.some.inj (percent.symm.trans minus)
    exact (by decide : (37#u8)≠45#u8) impossible
  | global index _ reference =>
    obtain ⟨byte,selected,valid⟩:=module_global_first source entry program meaning token index reference
    have same : byte=45#u8 := Option.some.inj (selected.symm.trans minus)
    subst byte
    rcases valid with h|h|h|h <;> norm_num at h
  | special kind text _ known spelled =>
    have percent:=special_first source token text kind known spelled
    have impossible : 37#u8=45#u8 := Option.some.inj (percent.symm.trans minus)
    exact (by decide : (37#u8)≠45#u8) impossible

theorem module_operand_syntax_unique (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (tokens : List SourceToken) (bits start leftFinish rightFinish : Nat) (left right : Operand)
    (first : OperandSyntax source program.scope tokens bits start leftFinish left)
    (second : OperandSyntax source program.scope tokens bits start rightFinish right) :
    leftFinish=rightFinish ∧ left=right := by
  cases first with
  | single leftSelected leftMeaning =>
    cases second with
    | single rightSelected rightMeaning =>
      have sameToken:=Option.some.inj (leftSelected.symm.trans rightSelected)
      subst_vars
      exact ⟨rfl,module_operand_meaning_unique source entry program meaning _ bits _ _ leftMeaning rightMeaning⟩
    | negative encoded minus =>
      obtain ⟨token,selected,spelled⟩:=minus
      have sameToken:=Option.some.inj (leftSelected.symm.trans selected)
      subst_vars
      exact False.elim (module_operand_not_minus source entry program meaning _ bits _ leftMeaning spelled)
  | negative encoded minus number width bound value =>
    cases second with
    | single rightSelected rightMeaning =>
      obtain ⟨token,selected,spelled⟩:=minus
      have sameToken:=Option.some.inj (selected.symm.trans rightSelected)
      subst_vars
      exact False.elim (module_operand_not_minus source entry program meaning _ bits _ rightMeaning spelled)
    | negative other _ otherNumber _ _ otherValue =>
      obtain ⟨leftToken,leftSelected,_,leftNumeral⟩:=number
      obtain ⟨rightToken,rightSelected,_,rightNumeral⟩:=otherNumber
      have sameToken:=Option.some.inj (leftSelected.symm.trans rightSelected)
      subst rightToken
      have sameNumber:=unsigned_numeral_unique source leftToken.start.val leftToken.end.val _ _ leftNumeral rightNumeral
      have same : encoded=other := by
        apply UScalar.eq_of_val_eq
        rw [value,otherValue,sameNumber]
      subst other
      exact ⟨rfl,rfl⟩

theorem offset_syntax_unique (source : List U8) (tokens : List SourceToken) (start leftFinish rightFinish : Nat)
    (left right : Int) (first : OffsetSyntax source tokens start leftFinish left)
    (second : OffsetSyntax source tokens start rightFinish right) (finishEqual : leftFinish=rightFinish) : left=right := by
  cases first with
  | absent =>
    cases second with
    | absent => rfl
    | positive => omega
  | positive plus number bound =>
    cases second with
    | absent => omega
    | positive _ otherNumber =>
      obtain ⟨leftToken,leftSelected,_,leftNumeral⟩:=number
      obtain ⟨rightToken,rightSelected,_,rightNumeral⟩:=otherNumber
      have sameToken:=Option.some.inj (leftSelected.symm.trans rightSelected)
      subst rightToken
      have sameNumber:=unsigned_numeral_unique source leftToken.start.val leftToken.end.val _ _ leftNumeral rightNumeral
      exact congrArg Int.ofNat sameNumber


theorem offset_closed_unique (source : List U8) (tokens : List SourceToken) (start leftFinish rightFinish : Nat)
    (left right : Int) (first : OffsetSyntax source tokens start leftFinish left)
    (second : OffsetSyntax source tokens start rightFinish right)
    (firstClosed : TokenAt source tokens leftFinish "]") (secondClosed : TokenAt source tokens rightFinish "]") :
    leftFinish=rightFinish ∧ left=right := by
  cases first with
  | absent =>
    cases second with
    | absent => exact ⟨rfl,rfl⟩
    | positive plus =>
      have impossible:=token_at_spelling_unique source tokens start "]" "+" firstClosed plus
      exact False.elim ((by decide : utf8 "]"≠utf8 "+") impossible)
  | positive plus number bound =>
    cases second with
    | absent =>
      have impossible:=token_at_spelling_unique source tokens start "+" "]" plus secondClosed
      exact False.elim ((by decide : utf8 "+"≠utf8 "]") impossible)
    | positive otherPlus otherNumber otherBound =>
      exact ⟨rfl,offset_syntax_unique source tokens start _ _ _ _
        (.positive plus number bound) (.positive otherPlus otherNumber otherBound) rfl⟩

theorem module_address_syntax_unique (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (tokens : List SourceToken) (start leftFinish rightFinish : Nat)
    (left right : ptxv_memory_write.value_flow.Address)
    (first : AddressSyntax source program.scope tokens start leftFinish left)
    (second : AddressSyntax source program.scope tokens start rightFinish right) :
    leftFinish=rightFinish ∧ left=right := by
  cases first with
  | @address leftMiddle leftToken leftBase leftIndex leftOffset _ leftEncoded leftOpen leftSelected leftMeaning leftTail leftValue leftClose leftEnd =>
    cases second with
    | @address rightMiddle rightToken rightBase rightIndex rightOffset _ rightEncoded rightOpen rightSelected rightMeaning rightTail rightValue rightClose rightEnd =>
      have sameToken:=Option.some.inj (leftSelected.symm.trans rightSelected)
      subst rightToken
      have sameBase:=module_operand_meaning_unique source entry program meaning leftToken 64 leftBase rightBase leftMeaning rightMeaning
      subst rightBase
      cases leftTail with
      | offset leftOffsetMeaning =>
        cases rightTail with
        | offset rightOffsetMeaning =>
          obtain ⟨sameMiddle,sameOffset⟩:=offset_closed_unique source tokens (start+2) leftMiddle rightMiddle _ _
            leftOffsetMeaning rightOffsetMeaning leftClose rightClose
          have sameEncoded : leftEncoded=rightEncoded := by
            apply IScalar.eq_of_val_eq
            rw [leftValue,rightValue,sameOffset]
          subst rightEncoded
          exact ⟨by omega,rfl⟩

end PtxFrontendProgram
