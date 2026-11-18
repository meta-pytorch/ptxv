/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.StoreRegisterMembership
import Instruction.StoreSourceRoles

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

def narrowStoreObservation (result : Option (Reg × U32)) : Option (Key × Nat) :=
  result.map (fun entry => (key entry.1, entry.2.val))

def narrowWithControls (operation : Op) (control : List Key) : Option (Key × Nat) :=
  match narrowStoreDescriptor operation with
  | none => none
  | some (register, mask) =>
      if register.file = .V ∨ register ∈ control then none else some (register, mask)

@[step] theorem narrow_store_descriptor_spec (operation : Op) :
    store_window.narrow_descriptor operation ⦃ result =>
      narrowStoreObservation result = narrowStoreDescriptor operation ⦄ := by
  cases operation <;> try (solve | simp [store_window.narrow_descriptor,
    narrowStoreDescriptor, narrowStoreObservation])
  case St space bytes address value sync =>
    have h1 : bytes.val = 1 ↔ bytes = 1#u8 := by scalar_tac
    have h2 : bytes.val = 2 ↔ bytes = 2#u8 := by scalar_tac
    cases value <;> by_cases hb1 : bytes = 1#u8 <;> by_cases hb2 : bytes = 2#u8 <;>
      simp [store_window.narrow_descriptor, narrowStoreDescriptor, narrowStoreObservation,
        h1, h2, hb1, hb2]
    all_goals simp_all [lift, bind_ok]

@[step] theorem narrow_store_sink_spec (operation : Op) (control : Slice Reg) :
    store_window.narrow_sink operation control ⦃ result =>
      narrowStoreObservation result = narrowWithControls operation (control.val.map key) ⦄ := by
  unfold store_window.narrow_sink
  step with narrow_store_descriptor_spec operation as ⟨descriptor, hdescriptor⟩
  cases descriptor with
  | none =>
    simp only [narrowStoreObservation, Option.map_none] at hdescriptor
    simp [narrowWithControls, ← hdescriptor, narrowStoreObservation]
  | some descriptor =>
    rcases descriptor with ⟨register, mask⟩
    simp only [narrowStoreObservation, Option.map_some] at hdescriptor
    step with file_eq register.file .V as ⟨virtual, hvirtual⟩
    rw [hvirtual]
    by_cases hf : register.file = .V
    · simp [hf, narrowWithControls, ← hdescriptor, key, narrowStoreObservation]
    · simp only [hf, decide_false, Bool.false_eq_true, ite_false]
      step with has_register_spec control register as ⟨found, hfound⟩
      have hfile : (key register).file ≠ .V := hf
      by_cases hc : key register ∈ control.val.map key
      all_goals simp only [hfound, hc, decide_true, decide_false, Bool.false_eq_true, ite_true, ite_false]
      all_goals simp only [narrowWithControls, ← hdescriptor, hfile, hc, false_or, or_true,
        ite_true, ite_false, narrowStoreObservation, Option.map_none, Option.map_some, WP.spec_ok]

end Instruction
