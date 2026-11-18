/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Text

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false

namespace PtxFrontendProgram

def registerTypes : List (List U8 × Nat × Bool) :=
  [([46#u8, 112#u8, 114#u8, 101#u8, 100#u8], 1, true),
   ([46#u8, 98#u8, 51#u8, 50#u8], 32, false), ([46#u8, 117#u8, 51#u8, 50#u8], 32, false),
   ([46#u8, 115#u8, 51#u8, 50#u8], 32, false), ([46#u8, 98#u8, 54#u8, 52#u8], 64, false),
   ([46#u8, 117#u8, 54#u8, 52#u8], 64, false), ([46#u8, 115#u8, 54#u8, 52#u8], 64, false)]

def RegisterTypeMeaning (source : List U8) (token : strict_memory_frontend.Token)
    (bits : Nat) (predicate : Bool) : Prop :=
  ∃ text, (text, bits, predicate) ∈ registerTypes ∧
    TokenText source token.start.val token.end.val text

open Lean Elab Tactic Meta in
elab "peel_frontend_bool " hypothesis:ident : tactic => withMainContext do
  let declaration ← (← getFVarId hypothesis).getDecl
  let equality ← instantiateMVars declaration.type
  unless equality.isAppOfArity ``Eq 3 do
    throwError "not a source equality"
  let arguments := equality.getAppArgs
  let mut left := arguments[1]!.headBeta
  while left.isLet do
    left := left.letBody!.instantiate1 left.letValue! |>.headBeta
  unless left.getAppFn.isConstOf ``Aeneas.Std.bind do
    throwError "not a source bind"
  evalTactic (← `(tactic| (
    obtain ⟨value, decoded, nextComputed⟩ := source_bind_success _ _ _ $hypothesis
    clear $hypothesis
    have $hypothesis := nextComputed
    clear nextComputed
    cases value)))

theorem source_register_type (source : Slice U8) (token : strict_memory_frontend.Token)
    (bits : U8) (predicate : Bool)
    (computed : strict_memory_frontend_parse.register_type source token = .ok (some (bits, predicate))) :
    RegisterTypeMeaning source.val token bits.val predicate := by
  unfold strict_memory_frontend_parse.register_type at computed
  repeat' first
    | fail_if_no_progress simp only [lift, bind_ok, ite_true, Bool.false_eq_true, ite_false] at computed
    | peel_frontend_bool computed
  all_goals simp only [ok.injEq, Option.some.injEq, Prod.mk.injEq, reduceCtorEq] at computed
  all_goals rcases computed with ⟨rfl, rfl⟩
  all_goals rename_i matched
  all_goals
    have meaning := source_token_equals source _ token matched
    exact ⟨_, by simp [registerTypes, Array.to_slice, Array.make], meaning⟩

end PtxFrontendProgram
