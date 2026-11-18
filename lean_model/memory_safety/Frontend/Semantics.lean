/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Aeneas

open Aeneas Aeneas.Std

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace PtxFrontend

def Space (byte : U8) : Prop := byte.val ∈ [9, 10, 13, 32]

def WordByte (byte : U8) : Prop :=
  (97 ≤ byte.val ∧ byte.val ≤ 122) ∨
  (65 ≤ byte.val ∧ byte.val ≤ 90) ∨
  (48 ≤ byte.val ∧ byte.val ≤ 57) ∨ byte.val ∈ [95, 46, 36, 37]

def Punctuation (byte : U8) : Prop :=
  byte.val ∈ [40, 41, 123, 125, 91, 93, 59, 44, 61, 43, 45, 60, 62, 58, 64, 33]

instance (byte : U8) : Decidable (Space byte) := by unfold Space; infer_instance
instance (byte : U8) : Decidable (WordByte byte) := by unfold WordByte; infer_instance
instance (byte : U8) : Decidable (Punctuation byte) := by unfold Punctuation; infer_instance

def Span (source : List U8) (start finish : Nat) (predicate : U8 → Prop) : Prop :=
  start ≤ finish ∧ finish ≤ source.length ∧ ∀ index, start ≤ index → index < finish →
    ∃ byte, source[index]? = some byte ∧ predicate byte

def Stopped (source : List U8) (finish : Nat) (predicate : U8 → Prop) : Prop :=
  finish = source.length ∨ ∃ byte, source[finish]? = some byte ∧ ¬predicate byte

def WordSpan (source : List U8) (start finish : Nat) : Prop :=
  start < finish ∧ Span source start finish WordByte ∧ Stopped source finish WordByte

def LineBody (source : List U8) (start finish : Nat) : Prop :=
  Span source start finish (fun byte => byte.val ≠ 10) ∧
    (finish = source.length ∨ ∃ byte, source[finish]? = some byte ∧ byte.val = 10)

def CloserAt (source : List U8) (index : Nat) : Prop :=
  ∃ first second, source[index]? = some first ∧ source[index + 1]? = some second ∧
    first.val = 42 ∧ second.val = 47

def BlockBody (source : List U8) (start finish : Nat) : Prop :=
  start + 2 ≤ finish ∧ finish ≤ source.length ∧ CloserAt source (finish - 2) ∧
    ∀ index, start ≤ index → index < finish - 2 → ¬CloserAt source index

inductive Trivia (source : List U8) : Nat → Nat → Prop
  | empty (position) : Trivia source position position
  | space {start next finish byte} : source[start]? = some byte → Space byte →
      next = start + 1 → Trivia source next finish → Trivia source start finish
  | line {start body finish} : source[start]? = some 47#u8 → source[start + 1]? = some 47#u8 →
      LineBody source (start + 2) body → Trivia source body finish → Trivia source start finish
  | block {start body finish} : source[start]? = some 47#u8 → source[start + 1]? = some 42#u8 →
      BlockBody source (start + 2) body → Trivia source body finish → Trivia source start finish

structure Token where
  start : Nat
  finish : Nat
  word : Bool
  deriving DecidableEq

def TokenMeaning (source : List U8) (token : Token) : Prop :=
  if token.word then WordSpan source token.start token.finish
  else token.finish = token.start + 1 ∧ token.finish ≤ source.length ∧
    ∃ byte, source[token.start]? = some byte ∧ Punctuation byte

inductive TokenSequence (source : List U8) : Nat → List Token → Nat → Prop
  | empty (position) : TokenSequence source position [] position
  | next {start token tokens finish} : Trivia source start token.start → TokenMeaning source token →
      TokenSequence source token.finish tokens finish → TokenSequence source start (token :: tokens) finish

def Lexed (source : List U8) (tokens : List Token) : Prop :=
  ∃ finish, TokenSequence source 0 tokens finish ∧ Trivia source finish source.length

end PtxFrontend
