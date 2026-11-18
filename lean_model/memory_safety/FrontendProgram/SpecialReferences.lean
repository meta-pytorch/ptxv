/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SyntaxProperties

open Aeneas Aeneas.Std PtxvMemorySafety

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false

namespace PtxFrontendProgram

private theorem text_bytes (source : List U8) (token : SourceToken) (bytes : List U8)
    (text : TokenText source token.start.val token.end.val bytes) :
    nameBytes source token=bytes := by
  apply List.ext_getElem?
  intro index
  by_cases inside : index<bytes.length
  · simpa [nameBytes,List.getElem?_take,List.getElem?_drop,text.2.2.1,inside,Nat.add_comm]
      using text.2.2.2 index inside
  · simp [nameBytes,text.2.2.1,inside]

private theorem bytes_text (source : List U8) (token : SourceToken) (bytes : List U8)
    (ordered : token.start.val≤token.end.val) (bounded : token.end.val≤source.length)
    (equal : nameBytes source token=bytes) :
    TokenText source token.start.val token.end.val bytes := by
  have available : token.end.val-token.start.val≤source.length-token.start.val := by omega
  have width : token.end.val-token.start.val=bytes.length := by
    have same:=congrArg List.length equal
    simpa only [nameBytes,List.length_take,List.length_drop,Nat.min_eq_left available] using same
  refine ⟨ordered,bounded,width,?_⟩
  intro index inside
  have same:=congrArg (fun values=>values[index]?) equal
  simpa [nameBytes,List.getElem?_take,List.getElem?_drop,width,inside,Nat.add_comm] using same

private theorem spelling_name_equal (source : List U8) (left right : SourceToken)
    (text : String) (equal : NameEqual source left right)
    (spelled : TokenSpelling source right text) : TokenSpelling source left text := by
  apply bytes_text source left (utf8 text) (Nat.le_of_lt equal.1) equal.2.1
  exact equal.2.2.2.2.trans (text_bytes source right (utf8 text) spelled)

private theorem decimal_point (source : List U8) (start finish accumulator result : Nat)
    (digits : DigitSequence source 10 start finish accumulator result) :
    ∀ index, start ≤ index → index < finish →
      ∃ byte,source[index]?=some byte ∧ 48≤byte.val ∧ byte.val≤57 := by
  induction digits with
  | empty position value => intro index lower upper;omega
  | @next start finish accumulator digit result byte selected valid tail ih =>
    intro index lower upper
    by_cases same : index=start
    · subst index
      refine ⟨byte,selected,?_⟩
      rcases valid with ⟨bound,alternatives⟩
      rcases alternatives with decimal|hex|hex <;> omega
    · exact ih index (by omega) upper

private theorem special_last (text : String)
    (kind : ptxv_memory_write.value_flow.SpecialRegister)
    (known : (text,kind)∈specialNames) :
    0<(utf8 text).length ∧
      ∃ byte,(utf8 text)[(utf8 text).length-1]?=some byte ∧ 57<byte.val := by
  simp only [specialNames,List.mem_cons,Prod.mk.injEq,List.not_mem_nil,or_false] at known
  rcases known with ⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|
    ⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩
  all_goals first
    | exact ⟨by decide,120#u8,by decide,by decide⟩
    | exact ⟨by decide,121#u8,by decide,by decide⟩
    | exact ⟨by decide,122#u8,by decide,by decide⟩
    | exact ⟨by decide,100#u8,by decide,by decide⟩

private theorem special_shape (text : String)
    (kind : ptxv_memory_write.value_flow.SpecialRegister)
    (known : (text,kind)∈specialNames) :
    text="%laneid" ∨ ∃ position,2≤position ∧ (utf8 text)[position]?=some 46#u8 := by
  simp only [specialNames,List.mem_cons,Prod.mk.injEq,List.not_mem_nil,or_false] at known
  rcases known with ⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|
    ⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩
  all_goals first
    | exact .inl rfl
    | exact .inr ⟨4,by decide,by decide⟩
    | exact .inr ⟨6,by decide,by decide⟩
    | exact .inr ⟨5,by decide,by decide⟩
    | exact .inr ⟨7,by decide,by decide⟩

private theorem identifier_no_dot (source : List U8) (token : SourceToken)
    (text : String) (position : Nat) (identifier : Identifier source token true)
    (spelled : TokenSpelling source token text) (lower : 2≤position)
    (dot : (utf8 text)[position]?=some 46#u8) : False := by
  have inside : position<(utf8 text).length := (List.getElem?_eq_some_iff.mp dot).1
  have pointAt:=spelled.2.2.2 position inside
  rw [dot] at pointAt
  have width:=spelled.2.2.1
  obtain ⟨byte,selected,following⟩:=identifier.2.2.2.2.2.2 (token.start.val+position)
    (by change token.start.val+1+1≤token.start.val+position;omega) (by omega)
  have equal : byte=46#u8 := Option.some.inj (selected.symm.trans pointAt)
  subst byte
  simp only [FollowingCharacter] at following
  norm_num [InitialCharacter] at following

private theorem named_special_not_indexed (source : List U8) (declaration : Declaration)
    (token : SourceToken) (text : String) (kind : ptxv_memory_write.value_flow.SpecialRegister)
    (number : Nat) (indexed : declaration.indexed=true)
    (known : (text,kind)∈specialNames) (spelled : TokenSpelling source token text)
    (named : RegisterName source declaration token number) : False := by
  simp only [RegisterName,indexed,ite_true] at named
  obtain ⟨_,_,nonempty,_,_,digits,_⟩:=named
  obtain ⟨last,lastAt,decimal⟩:=decimal_point source _ _ _ _ digits (token.end.val-1)
    (by omega) (by omega)
  obtain ⟨positive,lastByte,lastByteAt,nondecimal⟩:=special_last text kind known
  have width:=spelled.2.2.1
  have point:=spelled.2.2.2 ((utf8 text).length-1) (by omega)
  rw [lastByteAt] at point
  have indexEq : token.start.val+((utf8 text).length-1)=token.end.val-1 := by
    have ordered:=spelled.1
    omega
  rw [indexEq] at point
  have equal : last=lastByte := Option.some.inj (lastAt.symm.trans point)
  subst lastByte
  omega

theorem special_spelling_not_register (source : List U8) (scope : Scope)
    (identifiers : ∀ declaration∈scope.declarations.val,Identifier source declaration.name true)
    (separate : RegisterNamesSeparate source scope) (token : SourceToken)
    (text : String) (kind : ptxv_memory_write.value_flow.SpecialRegister)
    (known : (text,kind)∈specialNames) (spelled : TokenSpelling source token text)
    (bits : Nat) (predicate : Bool) (index : U16) :
    ¬RegisterReference source scope token bits predicate index := by
  intro reference
  obtain ⟨declaration,member,_,_,number,_,named,_⟩:=reference
  cases indexed : declaration.indexed with
  | true => exact named_special_not_indexed source declaration token text kind number indexed known spelled named
  | false =>
    simp only [RegisterName,indexed] at named
    have declaredSpelling:=spelling_name_equal source declaration.name token text named.1 spelled
    rcases special_shape text kind known with lane|⟨position,lower,dot⟩
    · subst text
      exact separate declaration member "%laneid" (by simp) declaredSpelling
    · exact identifier_no_dot source declaration.name text position (identifiers declaration member)
        declaredSpelling lower dot

theorem module_special_not_register (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (token : SourceToken) (text : String) (kind : ptxv_memory_write.value_flow.SpecialRegister)
    (known : (text,kind)∈specialNames) (spelled : TokenSpelling source token text)
    (bits : Nat) (predicate : Bool) (index : U16) :
    ¬RegisterReference source program.scope token bits predicate index := by
  obtain ⟨tokens,globalsEnd,declarationsStart,instructionsStart,finish,
    _,_,_,_,declarations,_,_,_,_,_,_,_,symbols⟩:=meaning
  have identifiers:=declarations_syntax_properties source tokens declarationsStart instructionsStart
    program.scope.declarations.val declarations
  exact special_spelling_not_register source program.scope (fun declaration member=>(identifiers.2 declaration member).1)
    symbols.2.1 token text kind known spelled bits predicate index

end PtxFrontendProgram
