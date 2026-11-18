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


def DeclarationShape (left right : Declaration) : Prop :=
  left.name=right.name ∧ left.count=right.count ∧ left.bits=right.bits ∧
    left.predicate=right.predicate ∧ left.indexed=right.indexed

theorem register_type_meaning_unique (source : List U8) (token : SourceToken)
    (leftBits rightBits : Nat) (leftPredicate rightPredicate : Bool)
    (first : RegisterTypeMeaning source token leftBits leftPredicate)
    (second : RegisterTypeMeaning source token rightBits rightPredicate) :
    leftBits=rightBits ∧ leftPredicate=rightPredicate := by
  obtain ⟨leftText,leftKnown,leftSpelled⟩:=first
  obtain ⟨rightText,rightKnown,rightSpelled⟩:=second
  have sameText:=token_text_unique source token leftText rightText leftSpelled rightSpelled
  simp only [registerTypes,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at leftKnown rightKnown
  rcases leftKnown with h|h|h|h|h|h|h <;> rcases rightKnown with g|g|g|g|g|g|g <;>
    rcases h with ⟨rfl,rfl,rfl⟩ <;> rcases g with ⟨rfl,rfl,rfl⟩ <;> simp_all

theorem declaration_syntax_shape_unique (source : List U8) (tokens : List SourceToken)
    (start leftFinish rightFinish : Nat) (left right : Declaration)
    (first : DeclarationSyntax source tokens start leftFinish left)
    (second : DeclarationSyntax source tokens start rightFinish right) :
    leftFinish=rightFinish ∧ DeclarationShape left right := by
  cases first with
  | single declaration reg leftTypeAt leftType leftNameAt leftNameValid leftName leftCount leftIndexed leftEnd =>
    cases second with
    | single other otherReg rightTypeAt rightType rightNameAt rightNameValid rightName rightCount rightIndexed rightEnd =>
      have sameType:=Option.some.inj (leftTypeAt.symm.trans rightTypeAt)
      cases sameType
      have typeUnique:=register_type_meaning_unique source _ _ _ _ _ leftType rightType
      have sameName:=Option.some.inj (leftNameAt.symm.trans rightNameAt)
      refine ⟨rfl,?_,?_,?_,typeUnique.2,?_⟩
      · exact leftName.trans (sameName.trans rightName.symm)
      · exact UScalar.eq_of_val_eq (leftCount.trans rightCount.symm)
      · exact UScalar.eq_of_val_eq typeUnique.1
      · exact leftIndexed.trans rightIndexed.symm
    | indexed other otherReg rightTypeAt rightType rightNameAt rightNameValid rightName rightOpen rightNumber rightPositive rightCount rightIndexed rightClose rightEnd =>
      have impossible:=token_at_spelling_unique source tokens (start+3) ";" "<" leftEnd rightOpen
      exact False.elim ((by decide : utf8 ";"≠utf8 "<") impossible)
  | indexed declaration reg leftTypeAt leftType leftNameAt leftNameValid leftName leftOpen leftNumber leftPositive leftCount leftIndexed leftClose leftEnd =>
    cases second with
    | single other otherReg rightTypeAt rightType rightNameAt rightNameValid rightName rightCount rightIndexed rightEnd =>
      have impossible:=token_at_spelling_unique source tokens (start+3) "<" ";" leftOpen rightEnd
      exact False.elim ((by decide : utf8 "<"≠utf8 ";") impossible)
    | indexed other otherReg rightTypeAt rightType rightNameAt rightNameValid rightName rightOpen rightNumber rightPositive rightCount rightIndexed rightClose rightEnd =>
      have sameType:=Option.some.inj (leftTypeAt.symm.trans rightTypeAt)
      cases sameType
      have typeUnique:=register_type_meaning_unique source _ _ _ _ _ leftType rightType
      have sameName:=Option.some.inj (leftNameAt.symm.trans rightNameAt)
      have sameCount:=number_at_unique source tokens (start+4) _ _ leftNumber rightNumber
      refine ⟨rfl,?_,?_,?_,typeUnique.2,?_⟩
      · exact leftName.trans (sameName.trans rightName.symm)
      · exact UScalar.eq_of_val_eq (leftCount.trans (sameCount.trans rightCount.symm))
      · exact UScalar.eq_of_val_eq typeUnique.1
      · exact leftIndexed.trans rightIndexed.symm


theorem instruction_body_starts_opcode (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (instruction : Instruction)
    (meaning : InstructionBody source scope tokens start finish instruction) :
    ∃opcode,OpcodeAt source tokens start opcode := by
  cases meaning <;> exact ⟨_,by assumption⟩

private theorem opcode_not_reg (source : List U8) (tokens : List SourceToken) (start : Nat)
    (opcode : Opcode) (meaning : OpcodeAt source tokens start opcode)
    (reg : TokenAt source tokens start ".reg") : False := by
  obtain ⟨token,byte,selected,first,lower,upper⟩:=opcode_at_first source tokens start opcode meaning
  obtain ⟨other,otherSelected,spelled⟩:=reg
  have sameToken:=Option.some.inj (selected.symm.trans otherSelected)
  subst other
  have dot:=spelling_first source token ".reg" 46#u8 spelled (by decide)
  have sameByte:=Option.some.inj (first.symm.trans dot)
  subst byte
  norm_num at lower

theorem instruction_syntax_not_reg (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (instruction : GuardedInstruction)
    (meaning : InstructionSyntax source scope tokens start finish instruction)
    (reg : TokenAt source tokens start ".reg") : False := by
  cases meaning with
  | instruction guard body guardSyntax bodySyntax restriction last finish =>
    cases guardSyntax with
    | unguarded =>
      obtain ⟨opcode,meaning⟩:=instruction_body_starts_opcode source scope tokens start _ body bodySyntax
      exact opcode_not_reg source tokens start opcode meaning reg
    | positive register atSign =>
      have impossible:=token_at_spelling_unique source tokens start "@" ".reg" atSign reg
      exact (by decide : utf8 "@"≠utf8 ".reg") impossible
    | negative register atSign =>
      have impossible:=token_at_spelling_unique source tokens start "@" ".reg" atSign reg
      exact (by decide : utf8 "@"≠utf8 ".reg") impossible

theorem code_syntax_not_reg (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start pc finish : Nat) (code : List GuardedInstruction) (labels : List Label)
    (meaning : CodeSyntax source scope tokens start pc code labels finish)
    (reg : TokenAt source tokens start ".reg") : False := by
  cases meaning with
  | empty close =>
    have impossible:=token_at_spelling_unique source tokens start "}" ".reg" close reg
    exact (by decide : utf8 "}"≠utf8 ".reg") impossible
  | label target selected valid colon pc tail =>
    obtain ⟨token,otherSelected,spelled⟩:=reg
    have sameToken:=Option.some.inj (selected.symm.trans otherSelected)
    subst token
    obtain ⟨byte,first,validFirst⟩:=identifier_false_first source _ valid
    have dot:=spelling_first source _ ".reg" 46#u8 spelled (by decide)
    have sameByte:=Option.some.inj (first.symm.trans dot)
    subst byte
    rcases validFirst with h|h|h|h <;> norm_num at h
  | instruction head tail => exact instruction_syntax_not_reg source scope tokens start _ _ head reg

theorem declaration_syntax_starts_reg (source : List U8) (tokens : List SourceToken)
    (start finish : Nat) (declaration : Declaration)
    (meaning : DeclarationSyntax source tokens start finish declaration) : TokenAt source tokens start ".reg" := by
  cases meaning <;> assumption

theorem declarations_syntax_shapes_unique (source : List U8) (tokens : List SourceToken)
    (leftScope rightScope : Scope) (start leftFinish rightFinish leftCodeFinish rightCodeFinish : Nat)
    (left right : List Declaration) (leftCode rightCode : List GuardedInstruction) (leftLabels rightLabels : List Label)
    (first : DeclarationsSyntax source tokens start left leftFinish)
    (second : DeclarationsSyntax source tokens start right rightFinish)
    (firstCode : CodeSyntax source leftScope tokens leftFinish 0 leftCode leftLabels leftCodeFinish)
    (secondCode : CodeSyntax source rightScope tokens rightFinish 0 rightCode rightLabels rightCodeFinish) :
    leftFinish=rightFinish ∧ List.Forall₂ DeclarationShape left right := by
  induction first generalizing right rightFinish with
  | empty position =>
    cases second with
    | empty => exact ⟨rfl,.nil⟩
    | cons head tail =>
      exact False.elim (code_syntax_not_reg source leftScope tokens position 0 leftCodeFinish leftCode leftLabels
        firstCode (declaration_syntax_starts_reg source tokens position _ _ head))
  | @cons start middle finish declaration declarations head tail ih =>
    cases second with
    | empty =>
      exact False.elim (code_syntax_not_reg source rightScope tokens start 0 rightCodeFinish rightCode rightLabels
        secondCode (declaration_syntax_starts_reg source tokens start middle declaration head))
    | @cons _ otherMiddle otherFinish other others otherHead otherTail =>
      obtain ⟨sameMiddle,sameHead⟩:=declaration_syntax_shape_unique source tokens start middle otherMiddle declaration other head otherHead
      subst otherMiddle
      obtain ⟨sameFinish,sameTail⟩:=ih _ others otherTail firstCode secondCode
      exact ⟨sameFinish,.cons sameHead sameTail⟩


theorem declaration_shapes_bank_size (left right : List Declaration) (predicate : Bool)
    (shapes : List.Forall₂ DeclarationShape left right) : bankSize left predicate=bankSize right predicate := by
  induction shapes with
  | nil => rfl
  | @cons first second left right shape tail ih =>
    have sameCount:=congrArg (fun value : U16=>value.val) shape.2.1
    have samePredicate:=shape.2.2.2.1
    simp only [bankSize,List.filter_cons] at *
    rw [samePredicate]
    split <;> simp_all

theorem declaration_shapes_take (left right : List Declaration)
    (shapes : List.Forall₂ DeclarationShape left right) (count : Nat) :
    List.Forall₂ DeclarationShape (left.take count) (right.take count) := by
  induction shapes generalizing count with
  | nil => simp
  | cons head tail ih =>
    cases count with
    | zero => simp
    | succ count => simpa only [List.take_succ_cons] using List.Forall₂.cons head (ih count)

theorem declaration_shapes_selected (left right : List Declaration)
    (shapes : List.Forall₂ DeclarationShape left right) (index : Nat) (first second : Declaration)
    (firstSelected : left[index]?=some first) (secondSelected : right[index]?=some second) :
    DeclarationShape first second := by
  induction shapes generalizing index with
  | nil => simp at firstSelected
  | cons head tail ih =>
    cases index with
    | zero =>
      simp only [List.getElem?_cons_zero,Option.some.injEq] at firstSelected secondSelected
      subst_vars
      exact head
    | succ index =>
      simp only [List.getElem?_cons_succ] at firstSelected secondSelected
      exact ih index firstSelected secondSelected

theorem declaration_shape_first_unique (left right : Declaration) (shape : DeclarationShape left right)
    (sameFirst : left.first=right.first) : left=right := by
  cases left
  cases right
  simp_all [DeclarationShape]

theorem declaration_layout_shapes_unique (leftScope rightScope : Scope)
    (leftLayout : DeclarationLayout leftScope) (rightLayout : DeclarationLayout rightScope)
    (shapes : List.Forall₂ DeclarationShape leftScope.declarations.val rightScope.declarations.val) :
    leftScope.declarations=rightScope.declarations ∧ leftScope.virtual_count=rightScope.virtual_count ∧
      leftScope.predicate_count=rightScope.predicate_count := by
  have lengths:=shapes.length_eq
  have fields : leftScope.declarations.val=rightScope.declarations.val := by
    apply List.ext_getElem?
    intro index
    cases firstSelected : leftScope.declarations.val[index]? with
    | none =>
      have outside:=(List.getElem?_eq_none_iff.mp firstSelected)
      have otherNone : rightScope.declarations.val[index]?=none:=List.getElem?_eq_none_iff.mpr (by omega)
      exact otherNone.symm
    | some first =>
      have bound:=(List.getElem?_eq_some_iff.mp firstSelected).1
      have rightBound:index<rightScope.declarations.val.length:=by omega
      have secondSelected:=List.getElem?_eq_getElem rightBound
      let second:=rightScope.declarations.val[index]
      have shape:=declaration_shapes_selected _ _ shapes index first second firstSelected secondSelected
      have firstEq : first.first=second.first := by
        apply UScalar.eq_of_val_eq
        rw [leftLayout.2.2 index first firstSelected,rightLayout.2.2 index second secondSelected,shape.2.2.2.1]
        exact declaration_shapes_bank_size _ _ second.predicate (declaration_shapes_take _ _ shapes index)
      have equal : first=second := declaration_shape_first_unique first second shape firstEq
      rw [equal]
      exact secondSelected.symm
  refine ⟨?_,?_,?_⟩
  · exact (alloc.vec.Vec.eq_iff _ _).mpr fields
  · apply UScalar.eq_of_val_eq
    exact leftLayout.1.trans ((declaration_shapes_bank_size _ _ false shapes).trans rightLayout.1.symm)
  · apply UScalar.eq_of_val_eq
    exact leftLayout.2.1.trans ((declaration_shapes_bank_size _ _ true shapes).trans rightLayout.2.1.symm)

theorem declarations_then_code_unique (source : List U8) (tokens : List SourceToken)
    (leftScope rightScope : Scope) (start leftFinish rightFinish leftCodeFinish rightCodeFinish : Nat)
    (leftCode rightCode : List GuardedInstruction) (leftLabels rightLabels : List Label)
    (first : DeclarationsSyntax source tokens start leftScope.declarations.val leftFinish)
    (second : DeclarationsSyntax source tokens start rightScope.declarations.val rightFinish)
    (firstCode : CodeSyntax source leftScope tokens leftFinish 0 leftCode leftLabels leftCodeFinish)
    (secondCode : CodeSyntax source rightScope tokens rightFinish 0 rightCode rightLabels rightCodeFinish)
    (leftLayout : DeclarationLayout leftScope) (rightLayout : DeclarationLayout rightScope) :
    leftFinish=rightFinish ∧ leftScope.declarations=rightScope.declarations ∧
      leftScope.virtual_count=rightScope.virtual_count ∧ leftScope.predicate_count=rightScope.predicate_count := by
  obtain ⟨sameFinish,shapes⟩:=declarations_syntax_shapes_unique source tokens leftScope rightScope start leftFinish rightFinish
    leftCodeFinish rightCodeFinish _ _ leftCode rightCode leftLabels rightLabels first second firstCode secondCode
  exact ⟨sameFinish,declaration_layout_shapes_unique leftScope rightScope leftLayout rightLayout shapes⟩

end PtxFrontendProgram
