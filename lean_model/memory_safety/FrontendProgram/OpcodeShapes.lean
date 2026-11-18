/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Grammar

open Aeneas Aeneas.Std PtxvMemorySafety

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option Elab.async false

namespace PtxFrontendProgram

local macro "opcode_table" : tactic =>
  `(tactic| simp_all [opcodeNames,binaryNames,comparisonNames])

theorem opcode_count : opcodeNames.length=52 := by decide

theorem opcode_spellings_unique : (opcodeNames.map Prod.fst).Nodup := by decide

theorem opcode_no_nop (text : String) : (text,.Nop)∉opcodeNames := by
  opcode_table

theorem opcode_move_width (text : String) (bits : U8)
    (member : (text,.Move bits)∈opcodeNames) : bits=32#u8 ∨ bits=64#u8 := by
  opcode_table
  aesop

theorem opcode_conversion_widths (text : String) (fromBits toBits : U8)
    (member : (text,.Convert fromBits toBits)∈opcodeNames) :
    (fromBits=32#u8 ∧ toBits=64#u8) ∨ (fromBits=64#u8 ∧ toBits=32#u8) := by
  opcode_table
  aesop

theorem opcode_binary_shape (text : String) (bits : U8)
    (operation : ptxv_memory_write.value_flow.BinaryOperation)
    (member : (text,.Binary bits operation)∈opcodeNames) :
    (bits=32#u8 ∨ bits=64#u8) ∧
    (operation=.Add ∨ operation=.Subtract ∨ operation=.And ∨ operation=.Or ∨
      operation=.Xor ∨ operation=.MultiplyLow) := by
  cases operation <;> opcode_table <;> aesop

theorem opcode_comparison_shape (text : String) (bits : U8)
    (comparison : ptxv_memory_write.value_flow.Comparison)
    (member : (text,.Compare bits comparison)∈opcodeNames) :
    (bits=32#u8 ∨ bits=64#u8) ∧
    (comparison=.Equal ∨ comparison=.NotEqual ∨ comparison=.LessUnsigned ∨
      comparison=.LessEqualUnsigned ∨ comparison=.GreaterUnsigned ∨ comparison=.GreaterEqualUnsigned) := by
  cases comparison <;> opcode_table <;> aesop

theorem opcode_load_shape (text : String) (bytes : U8) (signed : Bool)
    (member : (text,.Load bytes signed)∈opcodeNames) :
    (bytes=1#u8 ∨ bytes=2#u8 ∨ bytes=4#u8 ∨ bytes=8#u8) ∧
    (signed=true → bytes=1#u8 ∨ bytes=2#u8 ∨ bytes=4#u8) := by
  cases signed <;> opcode_table <;> aesop

theorem opcode_store_shape (text : String) (bytes : U8)
    (member : (text,.Store bytes)∈opcodeNames) :
    bytes=1#u8 ∨ bytes=2#u8 ∨ bytes=4#u8 ∨ bytes=8#u8 := by
  opcode_table
  aesop

end PtxFrontendProgram
