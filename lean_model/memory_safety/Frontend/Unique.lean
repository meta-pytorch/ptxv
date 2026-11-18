/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Frontend.Semantics

open Aeneas Aeneas.Std
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192

namespace PtxFrontend

theorem span_stopped_unique (source : List U8) (start left right : Nat) (predicate : U8 → Prop)
    (first : Span source start left predicate) (second : Span source start right predicate)
    (firstStopped : Stopped source left predicate) (secondStopped : Stopped source right predicate) : left=right := by
  have firstLower:=first.1
  have firstUpper:=first.2.1
  have secondLower:=second.1
  have secondUpper:=second.2.1
  rcases lt_trichotomy left right with before|same|after
  · obtain ⟨byte,selected,valid⟩:=second.2.2 left first.1 before
    rcases firstStopped with full|⟨other,otherSelected,invalid⟩
    · omega
    · have equal : byte=other := Option.some.inj (selected.symm.trans otherSelected)
      subst other
      exact False.elim (invalid valid)
  · exact same
  · obtain ⟨byte,selected,valid⟩:=first.2.2 right second.1 after
    rcases secondStopped with full|⟨other,otherSelected,invalid⟩
    · omega
    · have equal : byte=other := Option.some.inj (selected.symm.trans otherSelected)
      subst other
      exact False.elim (invalid valid)

theorem word_span_unique (source : List U8) (start left right : Nat)
    (first : WordSpan source start left) (second : WordSpan source start right) : left=right := by
  exact span_stopped_unique source start left right WordByte first.2.1 second.2.1 first.2.2 second.2.2

theorem line_body_unique (source : List U8) (start left right : Nat)
    (first : LineBody source start left) (second : LineBody source start right) : left=right := by
  apply span_stopped_unique source start left right (fun byte=>byte.val≠10) first.1 second.1
  · rcases first.2 with full|⟨byte,selected,newline⟩
    · exact Or.inl full
    · exact Or.inr ⟨byte,selected,fun other=>other newline⟩
  · rcases second.2 with full|⟨byte,selected,newline⟩
    · exact Or.inl full
    · exact Or.inr ⟨byte,selected,fun other=>other newline⟩

theorem block_body_unique (source : List U8) (start left right : Nat)
    (first : BlockBody source start left) (second : BlockBody source start right) : left=right := by
  have firstLower:=first.1
  have secondLower:=second.1
  rcases lt_trichotomy left right with before|same|after
  · exact False.elim (second.2.2.2 (left-2) (by omega) (by omega) first.2.2.1)
  · exact same
  · exact False.elim (first.2.2.2 (right-2) (by omega) (by omega) second.2.2.1)

theorem word_not_punctuation (byte : U8) (word : WordByte byte) (punctuation : Punctuation byte) : False := by
  simp only [Punctuation,List.mem_cons,List.not_mem_nil,or_false] at punctuation
  rcases word with lower|upper|digit|symbols
  · rcases punctuation with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h <;> omega
  · rcases punctuation with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h <;> omega
  · rcases punctuation with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h <;> omega
  · simp only [List.mem_cons,List.not_mem_nil,or_false] at symbols
    rcases symbols with h|h|h|h <;> rcases punctuation with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h <;> omega

theorem token_meaning_unique (source : List U8) (left right : Token)
    (sameStart : left.start=right.start) (first : TokenMeaning source left)
    (second : TokenMeaning source right) : left=right := by
  have sameWord : left.word=right.word := by
    cases leftWord:left.word <;> cases rightWord:right.word
    · rfl
    · simp only [TokenMeaning,leftWord,rightWord,ite_true] at first second
      obtain ⟨byte,selected,punct⟩:=first.2.2
      obtain ⟨other,otherSelected,word⟩:=second.2.1.2.2 right.start (by omega) second.1
      have same : byte=other:=Option.some.inj (selected.symm.trans (sameStart ▸ otherSelected))
      subst other
      exact False.elim (word_not_punctuation byte word punct)
    · simp only [TokenMeaning,leftWord,rightWord,ite_true] at first second
      obtain ⟨byte,selected,word⟩:=first.2.1.2.2 left.start (by omega) first.1
      obtain ⟨other,otherSelected,punct⟩:=second.2.2
      have same : byte=other:=Option.some.inj (selected.symm.trans (sameStart ▸ otherSelected))
      subst other
      exact False.elim (word_not_punctuation byte word punct)
    · rfl
  have sameFinish : left.finish=right.finish := by
    cases word:left.word
    · have otherWord : right.word=false:=sameWord.symm.trans word
      simp only [TokenMeaning,word,otherWord] at first second
      calc
        left.finish=left.start+1:=first.1
        _=right.start+1:=congrArg (fun n:Nat=>n+1) sameStart
        _=right.finish:=second.1.symm
    · have otherWord : right.word=true:=sameWord.symm.trans word
      simp only [TokenMeaning,word,otherWord,ite_true] at first second
      apply word_span_unique source left.start left.finish right.finish first
      simpa only [sameStart] using second
  cases left
  cases right
  simp_all


theorem character_not_trivia (byte : U8) (kind : WordByte byte ∨ Punctuation byte) :
    ¬Space byte ∧ byte.val≠47 := by
  simp only [WordByte,Punctuation,Space,List.mem_cons,List.not_mem_nil,or_false] at kind ⊢
  omega

theorem token_first_not_trivia (source : List U8) (token : Token) (meaning : TokenMeaning source token) :
    ∃ byte,source[token.start]?=some byte ∧ ¬Space byte ∧ byte.val≠47 := by
  cases word:token.word
  · simp only [TokenMeaning,word] at meaning
    obtain ⟨byte,selected,kind⟩:=meaning.2.2
    exact ⟨byte,selected,character_not_trivia byte (Or.inr kind)⟩
  · simp only [TokenMeaning,word,ite_true] at meaning
    obtain ⟨byte,selected,kind⟩:=meaning.2.1.2.2 token.start (by omega) meaning.1
    exact ⟨byte,selected,character_not_trivia byte (Or.inl kind)⟩

theorem trivia_ordered (source : List U8) (start finish : Nat) (meaning : Trivia source start finish) : start≤finish := by
  induction meaning with
  | empty => omega
  | space selected space next tail ih => omega
  | line first second body tail ih => have lower:=body.1.1;omega
  | block first second body tail ih => have lower:=body.1;omega

theorem token_bounds (source : List U8) (token : Token) (meaning : TokenMeaning source token) :
    token.start<token.finish ∧ token.finish≤source.length := by
  cases word:token.word
  · simp only [TokenMeaning,word] at meaning
    exact ⟨by have first:=meaning.1;omega,meaning.2.1⟩
  · simp only [TokenMeaning,word,ite_true] at meaning
    exact ⟨meaning.1,meaning.2.1.2.1⟩

theorem space_not_slash (source : List U8) (start : Nat) (byte : U8)
    (selected : source[start]?=some byte) (space : Space byte)
    (slash : source[start]?=some 47#u8) : False := by
  have same : byte=47#u8:=Option.some.inj (selected.symm.trans slash)
  subst byte
  norm_num [Space] at space

theorem token_not_space (source : List U8) (token : Token) (byte : U8)
    (meaning : TokenMeaning source token) (selected : source[token.start]?=some byte)
    (space : Space byte) : False := by
  obtain ⟨other,otherSelected,notSpace,_⟩:=token_first_not_trivia source token meaning
  have same : byte=other:=Option.some.inj (selected.symm.trans otherSelected)
  subst other
  exact notSpace space

theorem token_not_slash (source : List U8) (token : Token)
    (meaning : TokenMeaning source token) (slash : source[token.start]?=some 47#u8) : False := by
  obtain ⟨byte,selected,_,notSlash⟩:=token_first_not_trivia source token meaning
  have same : byte=47#u8:=Option.some.inj (selected.symm.trans slash)
  subst byte
  exact notSlash rfl

theorem trivia_token_stops_before (source : List U8) (start left : Nat) (first : Trivia source start left) :
    ∀ (right : Nat) (token : Token),Trivia source start right → token.start=right → TokenMeaning source token → left≤right := by
  induction first with
  | empty position =>
    intro right token second selected meaning
    exact trivia_ordered source position right second
  | @space start next finish byte selected space nextEq tail ih =>
    intro right token second tokenStart meaning
    cases second with
    | empty =>
      exact False.elim (token_not_space source token byte meaning (by simpa only [tokenStart] using selected) space)
    | @space _ otherNext _ otherByte otherSelected otherSpace otherNextEq otherTail =>
      have sameNext : otherNext=next:=by omega
      subst otherNext
      exact ih right token (by simpa only [nextEq] using otherTail) tokenStart meaning
    | line slash => exact False.elim (space_not_slash source start byte selected space slash)
    | block slash => exact False.elim (space_not_slash source start byte selected space slash)
  | @line start body finish slash secondSlash bodyMeaning tail ih =>
    intro right token second tokenStart meaning
    cases second with
    | empty => exact False.elim (token_not_slash source token meaning (by simpa only [tokenStart] using slash))
    | space selected space => exact False.elim (space_not_slash source start _ selected space slash)
    | @line _ otherBody _ _ _ otherBodyMeaning otherTail =>
      have sameBody:=line_body_unique source (start+2) body otherBody bodyMeaning otherBodyMeaning
      subst otherBody
      exact ih right token otherTail tokenStart meaning
    | block _ star =>
      have same : 47#u8=42#u8:=Option.some.inj (secondSlash.symm.trans star)
      exact False.elim ((by decide : (47#u8)≠42#u8) same)
  | @block start body finish slash star bodyMeaning tail ih =>
    intro right token second tokenStart meaning
    cases second with
    | empty => exact False.elim (token_not_slash source token meaning (by simpa only [tokenStart] using slash))
    | space selected space => exact False.elim (space_not_slash source start _ selected space slash)
    | line _ secondSlash =>
      have same : 42#u8=47#u8:=Option.some.inj (star.symm.trans secondSlash)
      exact False.elim ((by decide : (42#u8)≠47#u8) same)
    | @block _ otherBody _ _ _ otherBodyMeaning otherTail =>
      have sameBody:=block_body_unique source (start+2) body otherBody bodyMeaning otherBodyMeaning
      subst otherBody
      exact ih right token otherTail tokenStart meaning


theorem trivia_token_unique (source : List U8) (start : Nat) (left right : Token)
    (firstTrivia : Trivia source start left.start) (secondTrivia : Trivia source start right.start)
    (first : TokenMeaning source left) (second : TokenMeaning source right) : left=right := by
  have before:=trivia_token_stops_before source start left.start firstTrivia right.start right secondTrivia rfl second
  have after:=trivia_token_stops_before source start right.start secondTrivia left.start left firstTrivia rfl first
  exact token_meaning_unique source left right (by omega) first second

theorem token_sequence_unique (source : List U8) (start : Nat) (left right : List Token)
    (leftFinish rightFinish : Nat) (first : TokenSequence source start left leftFinish)
    (second : TokenSequence source start right rightFinish)
    (firstTrailing : Trivia source leftFinish source.length) (secondTrailing : Trivia source rightFinish source.length) : left=right := by
  induction first generalizing right rightFinish with
  | empty position =>
    cases second with
    | empty => rfl
    | @next _ token tokens finish trivia meaning tail =>
      have before:=trivia_token_stops_before source position source.length firstTrailing token.start token trivia rfl meaning
      have bounds:=token_bounds source token meaning
      omega
  | @next start token tokens finish trivia meaning tail ih =>
    cases second with
    | empty =>
      have before:=trivia_token_stops_before source start source.length secondTrailing token.start token trivia rfl meaning
      have bounds:=token_bounds source token meaning
      omega
    | @next _ otherToken otherTokens otherFinish otherTrivia otherMeaning otherTail =>
      have same:=trivia_token_unique source start token otherToken trivia otherTrivia meaning otherMeaning
      subst otherToken
      have sameTail:=ih otherTokens _ otherTail firstTrailing secondTrailing
      exact congrArg (List.cons token) sameTail

theorem lexed_unique (source : List U8) (left right : List Token)
    (first : Lexed source left) (second : Lexed source right) : left=right := by
  obtain ⟨leftFinish,first,firstTrailing⟩:=first
  obtain ⟨rightFinish,second,secondTrailing⟩:=second
  exact token_sequence_unique source 0 left right leftFinish rightFinish first second firstTrailing secondTrailing

end PtxFrontend
