/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SyntaxUnique
import Frontend.Unique

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192

namespace PtxFrontendProgram

theorem number_at_unique (source : List U8) (tokens : List SourceToken) (position left right : Nat)
    (first : NumberAt source tokens position left) (second : NumberAt source tokens position right) : left=right := by
  obtain ⟨leftToken,leftSelected,_,leftNumeral⟩:=first
  obtain ⟨rightToken,rightSelected,_,rightNumeral⟩:=second
  have sameToken:=Option.some.inj (leftSelected.symm.trans rightSelected)
  subst rightToken
  exact unsigned_numeral_unique source leftToken.start.val leftToken.end.val left right leftNumeral rightNumeral

theorem source_token_model_injective (left right : SourceToken) (same : tokenModel left=tokenModel right) : left=right := by
  have startEq : left.start=right.start := UScalar.eq_of_val_eq (congrArg PtxFrontend.Token.start same)
  have finishEq : left.end=right.end := UScalar.eq_of_val_eq (congrArg PtxFrontend.Token.finish same)
  have wordEq : left.word=right.word:=congrArg PtxFrontend.Token.word same
  cases left
  cases right
  simp_all

theorem source_tokens_lexed_unique (source : List U8) (left right : List SourceToken)
    (first : PtxFrontend.Lexed source (left.map tokenModel))
    (second : PtxFrontend.Lexed source (right.map tokenModel)) : left=right := by
  have same:=PtxFrontend.lexed_unique source _ _ first second
  have injective : ∀ (left right : List SourceToken),left.map tokenModel=right.map tokenModel → left=right := by
    intro left
    induction left with
    | nil => intro right equal;cases right <;> simp_all
    | cons head tail ih =>
      intro right equal
      cases right with
      | nil => simp at equal
      | cons other rest =>
        obtain ⟨sameHead,sameTail⟩:=List.cons.inj equal
        have sameToken:=source_token_model_injective head other sameHead
        subst other
        exact congrArg (List.cons head) (ih rest sameTail)
  exact injective left right same

theorem code_syntax_label_identifiers (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start pc finish : Nat) (code : List GuardedInstruction) (labels : List Label)
    (meaning : CodeSyntax source scope tokens start pc code labels finish) :
    ∀ label∈labels,Identifier source label.name false := by
  induction meaning with
  | empty => simp
  | label target selected identifier colon targetPc tail ih =>
    intro label member
    rcases List.mem_cons.mp member with same|later
    · subst label;exact identifier
    · exact ih label later
  | instruction head tail ih => exact ih

theorem module_scope_identifiers (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program) :
    (∀ declaration∈program.scope.declarations.val,Identifier source declaration.name true) ∧
    (∀ global∈program.scope.globals.val,Identifier source global.name false) ∧
    (∀ label∈program.scope.labels.val,Identifier source label.name false) := by
  obtain ⟨tokens,globalsEnd,declarationsStart,instructionsStart,finish,
    _,_,globals,_,declarations,code,_,_,_,_,_,_,_⟩:=meaning
  have properties:=declarations_syntax_properties source tokens declarationsStart instructionsStart
    program.scope.declarations.val declarations
  refine ⟨fun declaration member=>(properties.2 declaration member).1,
    globals_syntax_identifiers source tokens 6 globalsEnd program.scope.globals.val globals,?_⟩
  exact code_syntax_label_identifiers source program.scope tokens instructionsStart 0 finish
    program.code.val program.scope.labels.val code

theorem spelling_first (source : List U8) (token : SourceToken) (text : String) (byte : U8)
    (spelled : TokenSpelling source token text) (selected : (utf8 text)[0]?=some byte) :
    source[token.start.val]?=some byte := by
  have bound:0<(utf8 text).length:=(List.getElem?_eq_some_iff.mp selected).1
  simpa only [Nat.add_zero] using (spelled.2.2.2 0 bound).trans selected

theorem identifier_false_first (source : List U8) (token : SourceToken)
    (valid : Identifier source token false) :
    ∃byte,source[token.start.val]?=some byte ∧ InitialCharacter byte := by
  simpa only [Bool.false_eq_true,ite_false,Nat.add_zero] using valid.2.2.2.2.2.1

theorem opcode_names_first : ∀ item∈opcodeNames,
    ∃ byte,(utf8 item.1)[0]?=some byte ∧ 97≤byte.val ∧ byte.val≤122 := by decide

theorem opcode_at_first (source : List U8) (tokens : List SourceToken) (position : Nat) (opcode : Opcode)
    (meaning : OpcodeAt source tokens position opcode) :
    ∃token byte,tokens[position]?=some token ∧ source[token.start.val]?=some byte ∧ 97≤byte.val ∧ byte.val≤122 := by
  obtain ⟨text,member,token,selected,spelled⟩:=meaning
  obtain ⟨byte,first,lower,upper⟩:=opcode_names_first (text,opcode) member
  exact ⟨token,byte,selected,spelling_first source token text byte spelled first,lower,upper⟩

end PtxFrontendProgram
