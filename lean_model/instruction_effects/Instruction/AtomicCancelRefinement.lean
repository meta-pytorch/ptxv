/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AtomicCancelShape
open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace AtomicCancel

def int32Result (kind : IntOp) (left right : BitVec 32) : Option (BitVec 32) :=
  match kind with
  | .Sub => some (left - right)
  | .Xor => some (left ^^^ right)
  | _ => none

def cancellationValue (operation : Op) (values : Reg → BitVec 32) : Option (BitVec 32) :=
  match operation with
  | .Int kind _ _ (.Reg left _) (.Reg right _) => int32Result kind (values left) (values right)
  | _ => none

theorem cancels_true (operation : Op) (register : Reg)
    (accepted : atomic_cancel.cancels operation register = .ok true) : Shape operation register := by
  cases operation <;> try simp [atomic_cancel.cancels, U8.ofNat, UScalar.ofNat, UScalar.ofNatCore] at accepted
  rename_i kind bits destination left right
  by_cases width : bits = 32#u8
  · subst bits
    cases left <;> try simp [atomic_cancel.cancels, U8.ofNat, UScalar.ofNat, UScalar.ofNatCore] at accepted
    rename_i source slots
    cases right <;> try simp [atomic_cancel.cancels, U8.ofNat, UScalar.ofNat, UScalar.ofNatCore] at accepted
    rename_i source2 slots2
    by_cases count : slots = 1#u8
    · subst slots
      by_cases count2 : slots2 = 1#u8
      · subst slots2
        cases kind <;> try simp [atomic_cancel.cancels, U8.ofNat, UScalar.ofNat, UScalar.ofNatCore] at accepted
        all_goals
          classical
          rcases destination with ⟨file, index⟩
          cases file <;> simp [File.Insts.CoreCmpPartialEqFile.eq,
            read_discriminant, File.read_discriminant] at accepted
          all_goals
            by_cases same : source = register
            · simp [same_register_eq, same] at accepted
              subst source
              subst source2
              simp [Shape]
            · simp [same_register_eq, same] at accepted
      · split at accepted <;> simp_all [U8.ofNat, UScalar.ofNat, UScalar.ofNatCore]
    · split at accepted <;> simp_all [U8.ofNat, UScalar.ofNat, UScalar.ofNatCore]
  · split at accepted <;> simp_all [U8.ofNat, UScalar.ofNat, UScalar.ofNatCore]

theorem source_cancellation_value_zero (operation : Op) (register : Reg)
    (accepted : atomic_cancel.cancels operation register = .ok true)
    (values : Reg → BitVec 32) : cancellationValue operation values = some 0 := by
  obtain ⟨kind, destination, choice, privateFile, shape⟩ := cancels_true operation register accepted
  rcases choice with choice | choice <;> subst kind <;> subst operation <;>
    simp [cancellationValue, int32Result]

theorem source_cancellation_return_independent (operation : Op) (register : Reg)
    (accepted : atomic_cancel.cancels operation register = .ok true)
    (before after : Reg → BitVec 32) :
    Shape operation register ∧ cancellationValue operation before = some 0 ∧
      cancellationValue operation before = cancellationValue operation after := by
  exact ⟨cancels_true operation register accepted,
    source_cancellation_value_zero operation register accepted before,
    (source_cancellation_value_zero operation register accepted before).trans
      (source_cancellation_value_zero operation register accepted after).symm⟩

end AtomicCancel
