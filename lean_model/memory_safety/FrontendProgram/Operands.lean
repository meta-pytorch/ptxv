/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.RegisterLookup
import FrontendProgram.GlobalReferences
import FrontendProgram.OpcodeTables

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false
set_option backward.isDefEq.respectTransparency false

namespace PtxFrontendProgram

theorem source_operand (source : Slice U8) (scope : Scope) (token : SourceToken) (bits : U8) (operand : Operand)
    (counts : PositiveCounts scope)
    (computed : strict_memory_frontend_parse.operand source scope token bits=.ok (some operand)) :
    OperandMeaning source.val scope token bits.val operand := by
  by_cases bits32 : bits=32#u8
  · subst bits
    unfold strict_memory_frontend_parse.operand at computed
    simp only [bne_self_eq_false,Bool.false_eq_true,ite_false] at computed
    obtain ⟨parsed,numberComputed,computed⟩ := source_bind_success _ _ _ computed
    cases parsed with
    | some value =>
      simp only [ite_true] at computed
      split at computed
      · simp at computed
      · simp only [ok.injEq,Option.some.injEq] at computed
        cases computed
        exact OperandMeaning.immediate value (source_parse_unsigned source token value numberComputed)
          (source_unsigned_word source token value numberComputed) (Or.inl rfl) (by scalar_tac)
    | none =>
      obtain ⟨reg,registerComputed,computed⟩ := source_bind_success _ _ _ computed
      cases reg with
      | some index =>
        simp only [ok.injEq,Option.some.injEq] at computed
        cases computed
        exact OperandMeaning.register index (source_lookup_register source scope token 32#u8 false index counts registerComputed)
      | none =>
        simp only [ite_true] at computed
        obtain ⟨special,specialComputed,computed⟩ := source_bind_success _ _ _ computed
        cases special with
        | none => simp at computed
        | some register =>
          simp only [ok.injEq,Option.some.injEq] at computed
          cases computed
          obtain ⟨text,listed,spelling⟩ := source_special source token register specialComputed
          exact OperandMeaning.special register text rfl listed spelling
  · by_cases bits64 : bits=64#u8
    · subst bits
      unfold strict_memory_frontend_parse.operand at computed
      have not32 : (64#u8 !=32#u8)=true := by decide
      simp only [not32,bne_self_eq_false,Bool.false_eq_true,ite_true,ite_false] at computed
      obtain ⟨parsed,numberComputed,computed⟩ := source_bind_success _ _ _ computed
      cases parsed with
      | some value =>
        have different : (64#u8:U8)≠32#u8 := by decide
        simp only [different,ite_false,ok.injEq,Option.some.injEq] at computed
        cases computed
        exact OperandMeaning.immediate value (source_parse_unsigned source token value numberComputed)
          (source_unsigned_word source token value numberComputed) (Or.inr rfl) (by scalar_tac)
      | none =>
        obtain ⟨reg,registerComputed,computed⟩ := source_bind_success _ _ _ computed
        cases reg with
        | some index =>
          simp only [ok.injEq,Option.some.injEq] at computed
          cases computed
          exact OperandMeaning.register index (source_lookup_register source scope token 64#u8 false index counts registerComputed)
        | none =>
          have different : (64#u8:U8)≠32#u8 := by decide
          simp only [different,ite_false,ite_true] at computed
          obtain ⟨global,globalComputed,computed⟩ := source_bind_success _ _ _ computed
          cases global with
          | none => simp at computed
          | some index =>
            simp only [ok.injEq,Option.some.injEq] at computed
            cases computed
            exact OperandMeaning.global index rfl (source_lookup_global source scope token index globalComputed)
    · have not32 : (bits !=32#u8)=true := by scalar_tac
      have not64 : (bits !=64#u8)=true := by scalar_tac
      simp only [strict_memory_frontend_parse.operand,not32,not64,ite_true] at computed
      simp at computed

end PtxFrontendProgram
