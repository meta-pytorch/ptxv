/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 5000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

def lookupGlobal (bindings : List (U8 × U32 × U64)) (offset : U32) : Option U64 :=
  (bindings.find? (fun binding => decide
    (binding.1 = ptxv_ir.GLOBAL_RELOC_BANK ∧ binding.2.1 = offset))).map (fun binding => binding.2.2)

private theorem usize_increment (index : Usize)
    (bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits) :
    (index + 1#usize : Result Usize) = .ok (Usize.ofNatCore (index.val + 1) bound) := by
  have platformBound : index.val + 1 < 2 ^ System.Platform.numBits := bound
  simp [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
    UScalar.check_bounds, platformBound, Result.ofOption]
  rfl

private theorem lookup_unroll (bindings : Slice (U8 × U32 × U64)) (offset : U32)
    (found : Option U64) (cursor : Usize) :
    strict_memory_globals.global_address_loop bindings offset found cursor = (do
      let outcome ← strict_memory_globals.global_address_loop.body bindings offset found cursor
      match outcome with
      | .cont (next, index) => strict_memory_globals.global_address_loop bindings offset next index
      | .done output => .ok output) := by
  unfold strict_memory_globals.global_address_loop
  rw [loop.eq_def]
  congr 1
  funext outcome
  cases outcome with
  | cont pair => cases pair; rfl
  | done output => rfl

private theorem lookup_found (bindings : Slice (U8 × U32 × U64)) (offset : U32)
    (value : U64) (cursor : Usize) :
    strict_memory_globals.global_address_loop bindings offset (some value) cursor = .ok (some value) := by
  rw [lookup_unroll]
  by_cases inside : cursor < Slice.len bindings <;>
    simp [strict_memory_globals.global_address_loop.body, inside, core.option.Option.is_none]

private theorem lookup_cons (binding : U8 × U32 × U64)
    (tail : List (U8 × U32 × U64)) (offset : U32) :
    lookupGlobal (binding :: tail) offset =
      if binding.1 = ptxv_ir.GLOBAL_RELOC_BANK ∧ binding.2.1 = offset
      then some binding.2.2 else lookupGlobal tail offset := by
  by_cases hit : binding.1 = ptxv_ir.GLOBAL_RELOC_BANK ∧ binding.2.1 = offset <;>
    simp [lookupGlobal, hit]

theorem source_global_lookup_loop (bindings : Slice (U8 × U32 × U64)) (offset : U32)
    (cursor : Usize) :
    strict_memory_globals.global_address_loop bindings offset none cursor =
      .ok (lookupGlobal (bindings.val.drop cursor.val) offset) := by
  generalize fuelEq : bindings.val.length - cursor.val = fuel
  induction fuel using Nat.strong_induction_on generalizing cursor with
  | h fuel ih =>
    rw [lookup_unroll]
    by_cases inside : cursor.val < bindings.val.length
    · have test : cursor < Slice.len bindings := by scalar_tac
      have bound : cursor.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := bindings.property
        scalar_tac
      let next : Usize := Usize.ofNatCore (cursor.val + 1) bound
      have nextVal : next.val = cursor.val + 1 := by simp [next]
      have increment : (cursor + 1#usize : Result Usize) = .ok next := usize_increment cursor bound
      let binding := bindings.val[cursor.val]
      have read : Slice.index_usize bindings cursor = .ok binding := by
        simp [Slice.index_usize, List.getElem?_eq_getElem inside, binding]
        rfl
      have tail := ih (bindings.val.length - next.val) (by omega) next rfl
      rw [List.drop_eq_getElem_cons inside, lookup_cons]
      change _ = (Result.ok (if binding.1 = ptxv_ir.GLOBAL_RELOC_BANK ∧ binding.2.1 = offset
        then some binding.2.2 else lookupGlobal (bindings.val.drop (cursor.val + 1)) offset) : Result (Option U64))
      by_cases bank : binding.1 = ptxv_ir.GLOBAL_RELOC_BANK
      · by_cases location : binding.2.1 = offset
        · simp only [strict_memory_globals.global_address_loop.body, test, ite_true,
            core.option.Option.is_none, Option.isNone, Aeneas.Std.uncurry,
            read, bind_ok, bank, location, increment, lookup_found, and_self]
        · simp only [strict_memory_globals.global_address_loop.body, test, ite_true,
            core.option.Option.is_none, Option.isNone, Aeneas.Std.uncurry,
            read, bind_ok, bank, location, increment, tail, nextVal,
            and_false, ite_false]
      · simp only [strict_memory_globals.global_address_loop.body, test, ite_true,
          core.option.Option.is_none, Option.isNone, Aeneas.Std.uncurry,
          read, bind_ok, bank, increment, tail, nextVal, false_and, ite_false]
    · have test : ¬cursor < Slice.len bindings := by scalar_tac
      have drop : bindings.val.drop cursor.val = [] := List.drop_eq_nil_of_le (by omega)
      simp [strict_memory_globals.global_address_loop.body, test, drop, lookupGlobal]

theorem source_global_address_lookup (index : U32) (bindings : Slice (U8 × U32 × U64)) :
    strict_memory_globals.global_address index bindings = .ok
      ((index.checked_mul 8#u32).bind (fun offset => lookupGlobal bindings.val offset)) := by
  unfold strict_memory_globals.global_address
  simp only [lift]
  cases product : index.checked_mul 8#u32 with
  | none => simp
  | some offset => simpa using source_global_lookup_loop bindings offset 0#usize

theorem source_global_address_first_match (index : U32) (bindings : Slice (U8 × U32 × U64)) :
    strict_memory_globals.global_address index bindings = .ok
      (if index.val * 8 ≤ U32.max then
        (bindings.val.find? (fun binding => decide
          (binding.1.val = 31 ∧ binding.2.1.val = index.val * 8))).map (fun binding => binding.2.2)
       else none) := by
  rw [source_global_address_lookup]
  have meaning := U32.checked_mul_bv_spec index 8#u32
  cases product : index.checked_mul 8#u32 with
  | none =>
    simp only [product] at meaning
    have overflow : U32.max < index.val * 8 := by simpa [product] using meaning
    simp [Nat.not_le_of_lt overflow]
  | some offset =>
    simp only [product] at meaning
    have fits : index.val * 8 ≤ U32.max := by simpa [product] using meaning.1
    have offsetVal : offset.val = index.val * 8 := by simpa [product] using meaning.2.1
    simp only [Option.bind_some, fits, ite_true]
    unfold lookupGlobal
    have predicateEq :
        (fun binding : U8 × U32 × U64 => decide
          (binding.1 = ptxv_ir.GLOBAL_RELOC_BANK ∧ binding.2.1 = offset)) =
        (fun binding : U8 × U32 × U64 => decide
          (binding.1.val = 31 ∧ binding.2.1.val = index.val * 8)) := by
      funext binding
      have propositions :
          (binding.1 = ptxv_ir.GLOBAL_RELOC_BANK ∧ binding.2.1 = offset) ↔
          (binding.1.val = 31 ∧ binding.2.1.val = index.val * 8) := by
        constructor
        · rintro ⟨bankEq, locationEq⟩
          constructor
          · simp [bankEq, ptxv_ir.GLOBAL_RELOC_BANK]
          · simpa [locationEq] using offsetVal
        · rintro ⟨bankEq, locationEq⟩
          constructor
          · apply UScalar.eq_of_val_eq
            simpa [ptxv_ir.GLOBAL_RELOC_BANK] using bankEq
          · apply UScalar.eq_of_val_eq
            exact locationEq.trans offsetVal.symm
      simp only [propositions]
    rw [predicateEq]

end MemorySafetyIR
