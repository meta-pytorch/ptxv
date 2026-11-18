/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Grammar

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 6000000
set_option Elab.async false

namespace PtxFrontendProgram

theorem source_scalar_operand (operand : Operand) :
    strict_memory_frontend_parse.scalar_operand operand = .ok (decide (ScalarOperand operand)) := by
  cases operand <;> simp [strict_memory_frontend_parse.scalar_operand,ScalarOperand]

theorem source_address_base (operand : Operand) :
    strict_memory_frontend_parse.address_base operand = .ok (decide (AddressBase operand)) := by
  cases operand <;> simp [strict_memory_frontend_parse.address_base,AddressBase]

theorem source_instruction_operands (instruction : Instruction)
    (computed : strict_memory_frontend_parse.instruction_operands_supported instruction = .ok true) :
    OperandRestrictions instruction := by
  cases instruction with
  | Move bits destination source =>
    cases source <;>
      simp_all [strict_memory_frontend_parse.instruction_operands_supported,OperandRestrictions]
    all_goals scalar_tac
  | Binary bits destination operation left right =>
    cases left <;> cases right <;>
      simp_all [strict_memory_frontend_parse.instruction_operands_supported,
        strict_memory_frontend_parse.scalar_operand,OperandRestrictions,ScalarOperand,bind_ok]
  | MultiplyWide destination left right =>
    cases left <;> cases right <;>
      simp_all [strict_memory_frontend_parse.instruction_operands_supported,
        strict_memory_frontend_parse.scalar_operand,OperandRestrictions,ScalarOperand,bind_ok]
  | Compare bits destination comparison left right =>
    cases left <;> cases right <;>
      simp_all [strict_memory_frontend_parse.instruction_operands_supported,
        strict_memory_frontend_parse.scalar_operand,OperandRestrictions,ScalarOperand,bind_ok]
  | ConvertUnsigned destination source fromBits toBits =>
    cases source <;>
      simp_all [strict_memory_frontend_parse.instruction_operands_supported,
        strict_memory_frontend_parse.scalar_operand,OperandRestrictions,ScalarOperand]
  | Load destination destinationBits bytes signed address =>
    rcases address with ⟨base,index,offset⟩
    cases base <;>
      simp_all [strict_memory_frontend_parse.instruction_operands_supported,
        strict_memory_frontend_parse.address_base,OperandRestrictions,AddressBase]
  | Store bytes address source =>
    rcases address with ⟨base,index,offset⟩
    cases base <;> cases source <;>
      simp_all [strict_memory_frontend_parse.instruction_operands_supported,
        strict_memory_frontend_parse.address_base,strict_memory_frontend_parse.scalar_operand,
        OperandRestrictions,AddressBase,ScalarOperand,bind_ok]
  | Branch target => simp [OperandRestrictions]
  | Exit => simp [OperandRestrictions]
  | Nop => simp [strict_memory_frontend_parse.instruction_operands_supported] at computed

end PtxFrontendProgram
