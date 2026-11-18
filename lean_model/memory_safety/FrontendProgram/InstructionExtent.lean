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

theorem operand_syntax_extent (source entry : List U8)
    (leftProgram rightProgram : strict_memory_frontend_parse.SourceProgram)
    (leftMeaning : ModuleMeaning source entry leftProgram) (rightMeaning : ModuleMeaning source entry rightProgram)
    (tokens : List SourceToken) (bits start leftFinish rightFinish : Nat) (left right : Operand)
    (first : OperandSyntax source leftProgram.scope tokens bits start leftFinish left)
    (second : OperandSyntax source rightProgram.scope tokens bits start rightFinish right) : leftFinish=rightFinish := by
  cases first with
  | single leftSelected leftOperand =>
    cases second with
    | single => rfl
    | negative encoded minus =>
      obtain ⟨token,selected,spelled⟩:=minus
      have sameToken:=Option.some.inj (leftSelected.symm.trans selected)
      subst_vars
      exact False.elim (module_operand_not_minus source entry leftProgram leftMeaning _ bits _ leftOperand spelled)
  | negative encoded minus =>
    cases second with
    | single rightSelected rightOperand =>
      obtain ⟨token,selected,spelled⟩:=minus
      have sameToken:=Option.some.inj (selected.symm.trans rightSelected)
      subst_vars
      exact False.elim (module_operand_not_minus source entry rightProgram rightMeaning _ bits _ rightOperand spelled)
    | negative => rfl

theorem address_syntax_extent (source : List U8) (leftScope rightScope : Scope)
    (tokens : List SourceToken) (start leftFinish rightFinish : Nat)
    (left right : ptxv_memory_write.value_flow.Address)
    (first : AddressSyntax source leftScope tokens start leftFinish left)
    (second : AddressSyntax source rightScope tokens start rightFinish right) : leftFinish=rightFinish := by
  cases first with
  | @address leftMiddle leftToken leftBase leftIndex leftOffset _ leftEncoded leftOpen leftSelected leftMeaning leftTail leftValue leftClose leftEnd =>
    cases second with
    | @address rightMiddle rightToken rightBase rightIndex rightOffset _ rightEncoded rightOpen rightSelected rightMeaning rightTail rightValue rightClose rightEnd =>
      cases leftTail with
      | offset leftOffsetMeaning =>
        cases rightTail with
        | offset rightOffsetMeaning =>
          obtain ⟨sameMiddle,_⟩:=offset_closed_unique source tokens (start+2) leftMiddle rightMiddle _ _
            leftOffsetMeaning rightOffsetMeaning leftClose rightClose
          omega

def extentOpcode : Instruction → Opcode
  | .Move bits _ _ => .Move bits
  | .Binary bits _ operation _ _ => .Binary bits operation
  | .MultiplyWide _ _ _ => .MultiplyWide
  | .ConvertUnsigned _ _ fromBits toBits => .Convert fromBits toBits
  | .Compare bits _ comparison _ _ => .Compare bits comparison
  | .Branch _ => .Branch
  | .Load _ _ bytes signed _ => .Load bytes signed
  | .Store bytes _ _ => .Store bytes
  | .Nop => .Nop
  | .Exit => .Exit

theorem instruction_extent_opcode (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (body : Instruction) (meaning : InstructionBody source scope tokens start finish body) :
    OpcodeAt source tokens start (extentOpcode body) := by
  cases meaning <;> simp only [extentOpcode] <;> assumption

theorem two_operands_extent (source entry : List U8)
    (leftProgram rightProgram : strict_memory_frontend_parse.SourceProgram)
    (leftMeaning : ModuleMeaning source entry leftProgram) (rightMeaning : ModuleMeaning source entry rightProgram)
    (tokens : List SourceToken) (bits start leftMiddle rightMiddle leftFinish rightFinish : Nat)
    (leftA leftB rightA rightB : Operand)
    (firstA : OperandSyntax source leftProgram.scope tokens bits start leftMiddle leftA)
    (secondA : OperandSyntax source rightProgram.scope tokens bits start rightMiddle rightA)
    (firstB : OperandSyntax source leftProgram.scope tokens bits (leftMiddle+1) leftFinish leftB)
    (secondB : OperandSyntax source rightProgram.scope tokens bits (rightMiddle+1) rightFinish rightB) : leftFinish=rightFinish := by
  have middleEq:=operand_syntax_extent source entry leftProgram rightProgram leftMeaning rightMeaning
    tokens bits start leftMiddle rightMiddle leftA rightA firstA secondA
  subst rightMiddle
  exact operand_syntax_extent source entry leftProgram rightProgram leftMeaning rightMeaning
    tokens bits (leftMiddle+1) leftFinish rightFinish leftB rightB firstB secondB

theorem address_operand_extent (source entry : List U8)
    (leftProgram rightProgram : strict_memory_frontend_parse.SourceProgram)
    (leftMeaning : ModuleMeaning source entry leftProgram) (rightMeaning : ModuleMeaning source entry rightProgram)
    (tokens : List SourceToken) (bits start leftMiddle rightMiddle leftFinish rightFinish : Nat)
    (leftAddress rightAddress : ptxv_memory_write.value_flow.Address) (left right : Operand)
    (firstA : AddressSyntax source leftProgram.scope tokens start leftMiddle leftAddress)
    (secondA : AddressSyntax source rightProgram.scope tokens start rightMiddle rightAddress)
    (firstB : OperandSyntax source leftProgram.scope tokens bits (leftMiddle+1) leftFinish left)
    (secondB : OperandSyntax source rightProgram.scope tokens bits (rightMiddle+1) rightFinish right) : leftFinish=rightFinish := by
  have middleEq:=address_syntax_extent source leftProgram.scope rightProgram.scope tokens start leftMiddle rightMiddle
    leftAddress rightAddress firstA secondA
  subst rightMiddle
  exact operand_syntax_extent source entry leftProgram rightProgram leftMeaning rightMeaning
    tokens bits (leftMiddle+1) leftFinish rightFinish left right firstB secondB

theorem instruction_body_extent (source entry : List U8)
    (leftProgram rightProgram : strict_memory_frontend_parse.SourceProgram)
    (leftMeaning : ModuleMeaning source entry leftProgram) (rightMeaning : ModuleMeaning source entry rightProgram)
    (tokens : List SourceToken) (start leftFinish rightFinish : Nat) (left right : Instruction)
    (first : InstructionBody source leftProgram.scope tokens start leftFinish left)
    (second : InstructionBody source rightProgram.scope tokens start rightFinish right) : leftFinish=rightFinish := by
  have sameOpcode:=opcode_at_unique source tokens start _ _
    (instruction_extent_opcode source leftProgram.scope tokens start leftFinish left first)
    (instruction_extent_opcode source rightProgram.scope tokens start rightFinish right second)
  cases first <;> cases second <;> simp only [extentOpcode] at sameOpcode <;> cases sameOpcode
  all_goals first
    | rfl
    | solve
      | apply operand_syntax_extent source entry leftProgram rightProgram leftMeaning rightMeaning tokens _ (start+3)
        all_goals assumption
    | solve
      | apply address_syntax_extent source leftProgram.scope rightProgram.scope tokens (start+3)
        all_goals assumption
    | solve
      | apply two_operands_extent source entry leftProgram rightProgram leftMeaning rightMeaning tokens _ (start+3)
        all_goals assumption
    | solve
      | apply address_operand_extent source entry leftProgram rightProgram leftMeaning rightMeaning tokens _ (start+1)
        all_goals assumption

end PtxFrontendProgram
