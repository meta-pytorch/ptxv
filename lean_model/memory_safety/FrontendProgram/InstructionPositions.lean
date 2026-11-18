/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Instructions

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
namespace PtxFrontendProgram

theorem operand_syntax_advance (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (bits start finish : Nat) (operand : Operand) (meaning : OperandSyntax source scope tokens bits start finish operand) :
    start<finish := by cases meaning <;> omega

theorem offset_syntax_monotone (source : List U8) (tokens : List SourceToken)
    (start finish : Nat) (offset : Int) (meaning : OffsetSyntax source tokens start finish offset) :
    start≤finish := by cases meaning <;> omega

theorem address_tail_monotone (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (index : Operand) (offset : Int) (meaning : AddressTail source scope tokens start finish index offset) :
    start≤finish := by
  cases meaning with
  | offset offsetSyntax => exact offset_syntax_monotone _ _ _ _ _ offsetSyntax

theorem address_syntax_advance (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (address : ptxv_memory_write.value_flow.Address)
    (meaning : AddressSyntax source scope tokens start finish address) : start<finish := by
  cases meaning with
  | address encoded _ _ _ tail _ _ after =>
    have := address_tail_monotone _ _ _ _ _ _ _ tail
    omega

theorem guard_syntax_monotone (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (guard : ptxv_memory_write.value_flow.Guard)
    (meaning : GuardSyntax source scope tokens start finish guard) : start≤finish := by
  cases meaning <;> omega

theorem instruction_body_advance (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (instruction : Instruction)
    (meaning : InstructionBody source scope tokens start finish instruction) : start<finish := by
  cases meaning with
  | exit _ => omega
  | nop _ => omega
  | branch _ _ _ _ => omega
  | move bits destination operand _ _ _ operandSyntax =>
    have := operand_syntax_advance _ _ _ _ _ _ _ operandSyntax
    omega
  | binary bits destination operation left right _ _ _ leftSyntax _ rightSyntax =>
    have := operand_syntax_advance _ _ _ _ _ _ _ leftSyntax
    have := operand_syntax_advance _ _ _ _ _ _ _ rightSyntax
    omega
  | wide destination left right _ _ _ leftSyntax _ rightSyntax =>
    have := operand_syntax_advance _ _ _ _ _ _ _ leftSyntax
    have := operand_syntax_advance _ _ _ _ _ _ _ rightSyntax
    omega
  | convert fromBits toBits destination operand _ _ _ operandSyntax =>
    have := operand_syntax_advance _ _ _ _ _ _ _ operandSyntax
    omega
  | compare bits destination comparison left right _ _ _ leftSyntax _ rightSyntax =>
    have := operand_syntax_advance _ _ _ _ _ _ _ leftSyntax
    have := operand_syntax_advance _ _ _ _ _ _ _ rightSyntax
    omega
  | load bytes bits destination signed address _ _ _ _ _ addressSyntax =>
    have := address_syntax_advance _ _ _ _ _ _ addressSyntax
    omega
  | store bytes address operand _ addressSyntax _ operandSyntax =>
    have := address_syntax_advance _ _ _ _ _ _ addressSyntax
    have := operand_syntax_advance _ _ _ _ _ _ _ operandSyntax
    omega

theorem instruction_syntax_bounds (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (instruction : GuardedInstruction)
    (meaning : InstructionSyntax source scope tokens start finish instruction) : start<finish ∧ finish≤tokens.length := by
  cases meaning with
  | instruction guard body guardSyntax bodySyntax _operands semicolon after =>
    have lower := guard_syntax_monotone _ _ _ _ _ _ guardSyntax
    have advance := instruction_body_advance _ _ _ _ _ _ bodySyntax
    obtain ⟨token,found,_⟩ := semicolon
    have upper := (List.getElem?_eq_some_iff.mp found).1
    omega

end PtxFrontendProgram
