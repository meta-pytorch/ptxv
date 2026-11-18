/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Code.Types

set_option autoImplicit false

namespace ValueFlow.Source

namespace V

abbrev Pointer := PtxvMemorySafety.ptxv_memory_write.value_flow.Pointer
abbrev Value := PtxvMemorySafety.ptxv_memory_write.value_flow.Value
abbrev SpecialRegister := PtxvMemorySafety.ptxv_memory_write.value_flow.SpecialRegister
abbrev Operand := PtxvMemorySafety.ptxv_memory_write.value_flow.Operand
abbrev Guard := PtxvMemorySafety.ptxv_memory_write.value_flow.Guard
abbrev Address := PtxvMemorySafety.ptxv_memory_write.value_flow.Address
abbrev BinaryOperation := PtxvMemorySafety.ptxv_memory_write.value_flow.BinaryOperation
abbrev Comparison := PtxvMemorySafety.ptxv_memory_write.value_flow.Comparison
abbrev Instruction := PtxvMemorySafety.ptxv_memory_write.value_flow.Instruction
abbrev GuardedInstruction := PtxvMemorySafety.ptxv_memory_write.value_flow.GuardedInstruction
abbrev Action := PtxvMemorySafety.ptxv_memory_write.value_flow.Action
abbrev ThreadIds := PtxvMemorySafety.ptxv_memory_write.value_flow.ThreadIds
abbrev Registers := PtxvMemorySafety.ptxv_memory_write.value_flow.Registers

end V

end ValueFlow.Source
