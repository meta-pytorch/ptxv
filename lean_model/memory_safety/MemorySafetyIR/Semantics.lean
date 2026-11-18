/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Code.Types

open Aeneas Aeneas.Std PtxvMemorySafety

set_option autoImplicit false

namespace MemorySafetyIR

abbrev Request := ptxv_memory_write.memory_safety.Request
abbrev AccessKind := ptxv_memory_write.memory_safety.AccessKind

structure Valuation where
  register : ptxv_ir.Reg → U8 → U64
  constant : U32 → U8 → U64
  special : ptxv_ir.SReg → U64
  predicate : ptxv_ir.Reg → Bool
  registerDefined : ptxv_ir.Reg → U8 → Bool
  constantDefined : U32 → U8 → Bool
  specialDefined : ptxv_ir.SReg → Bool

def value (valuation : Valuation) : ptxv_ir.Val → U64
  | .Reg register slots => valuation.register register slots
  | .Imm bits => bits
  | .Const offset bytes => valuation.constant offset bytes
  | .SReg register => valuation.special register

def valueDefined (valuation : Valuation) : ptxv_ir.Val → Bool
  | .Reg register slots => valuation.registerDefined register slots
  | .Imm _ => true
  | .Const offset bytes => valuation.constantDefined offset bytes
  | .SReg register => valuation.specialDefined register

def optionalValue (valuation : Valuation) : Option ptxv_ir.Val → U64
  | none => 0#u64
  | some operand => value valuation operand

def address (valuation : Valuation) (operand : ptxv_ir.Addr) : U64 :=
  ⟨operand.off.bv + (optionalValue valuation operand.base).bv +
    (optionalValue valuation operand.base2).bv⟩

def width (bytes : U8) : U64 := ⟨bytes.bv.setWidth 64⟩

def predicate (valuation : Valuation) (operand : ptxv_ir.Pred) : Bool :=
  let base := match operand.reg with
    | none => true
    | some register => valuation.predicate register
  if operand.neg then !base else base

def globalAccess (valuation : Valuation) : ptxv_ir.Op → Option Request
  | .Ld _ _ .Global bytes operand _ _ _ =>
      some ⟨address valuation operand, width bytes, width bytes, .Load⟩
  | .St .Global bytes operand source _ =>
      some ⟨address valuation operand, width bytes, width bytes,
        .Store (valueDefined valuation source)⟩
  | _ => none

def instructionAccess (valuation : Valuation) (instruction : ptxv_ir.Inst) : Option Request :=
  if predicate valuation instruction.guard then globalAccess valuation instruction.op else none

def ExactAccess (instruction : ptxv_ir.Inst) (request : Request) : Prop :=
  ∀ valuation, instructionAccess valuation instruction = some request

def Silent (instruction : ptxv_ir.Inst) : Prop :=
  ∀ valuation, instructionAccess valuation instruction = none

end MemorySafetyIR
