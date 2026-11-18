/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.GlobalAddressSource

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 5000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

def globalAddressMeaning (bindings : List (U8 × U32 × U64)) (index : U32) : Option U64 :=
  if index.val * 8 ≤ U32.max then
    (bindings.find? (fun binding => decide
      (binding.1.val = 31 ∧ binding.2.1.val = index.val * 8))).map (fun binding => binding.2.2)
  else none

def globalOperandMeaning (bindings : List (U8 × U32 × U64)) : Option ptxv_ir.Val → Option U64
  | none => some 0#u64
  | some (.Imm bits) => some bits
  | some (.SReg (.GlobalAddr index)) => globalAddressMeaning bindings index
  | _ => none

def globalResolvedAddressMeaning (bindings : List (U8 × U32 × U64))
    (operand : ptxv_ir.Addr) : Option U64 := do
  let first ← globalOperandMeaning bindings operand.base
  let second ← globalOperandMeaning bindings operand.base2
  pure ⟨operand.off.bv + first.bv + second.bv⟩

theorem source_global_address_meaning (index : U32) (bindings : Slice (U8 × U32 × U64)) :
    strict_memory_globals.global_address index bindings =
      .ok (globalAddressMeaning bindings.val index) :=
  source_global_address_first_match index bindings

theorem source_global_operand (operand : Option ptxv_ir.Val)
    (bindings : Slice (U8 × U32 × U64)) :
    strict_memory_globals.address_value operand bindings =
      .ok (globalOperandMeaning bindings.val operand) := by
  cases operand with
  | none => rfl
  | some operand =>
    cases operand with
    | Imm bits => rfl
    | Reg register slots => rfl
    | Const offset bytes => rfl
    | SReg register =>
      cases register <;>
        simp [strict_memory_globals.address_value, globalOperandMeaning, source_global_address_meaning]

theorem source_resolve_global_address (operand : ptxv_ir.Addr)
    (bindings : Slice (U8 × U32 × U64)) :
    strict_memory_globals.resolve_address operand bindings =
      .ok (globalResolvedAddressMeaning bindings.val operand) := by
  simp only [strict_memory_globals.resolve_address, source_global_operand, bind_ok]
  cases first : globalOperandMeaning bindings.val operand.base <;>
    cases second : globalOperandMeaning bindings.val operand.base2 <;>
    simp [globalResolvedAddressMeaning, first, second, IScalar.hcast,
      core.num.U64.wrapping_add, UScalar.wrapping_add, lift]

end MemorySafetyIR
