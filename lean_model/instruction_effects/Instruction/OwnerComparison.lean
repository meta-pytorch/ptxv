/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.OwnerRefinement

open Aeneas Aeneas.Std PtxvIr

set_option autoImplicit false

namespace RegisterOwner.Comparison

structure Entry (Claim : Type) where
  register : U32 × U8 × U16
  claim : Claim

abbrev Store (Input Value : Type) := Key → Input → Value

def Sound {Input Value Claim : Type} (domain : Input → Prop) (owners : List Nat)
    (store : Store Input Value) (meaning : Claim → Input → Value)
    (entries : List (Entry Claim)) : Prop :=
  ∀ entry ∈ entries, ∀ input, domain input →
    store (recordKey owners entry.register) input = meaning entry.claim input

def write {Input Value : Type} (store : Store Input Value) (location : Key)
    (replacement : Input → Value) : Store Input Value :=
  fun key input => if key = location then replacement input else store key input

def invalidate {Claim : Type} (owners : List Nat) (location : Key)
    (entries : List (Entry Claim)) : List (Entry Claim) :=
  entries.filter fun entry => decide (recordKey owners entry.register ≠ location)

theorem retained_evidence_survives_write {Input Value Claim : Type}
    (domain : Input → Prop) (owners : List Nat) (store : Store Input Value)
    (meaning : Claim → Input → Value) (entries : List (Entry Claim))
    (sound : Sound domain owners store meaning entries) (location : Key)
    (replacement : Input → Value) :
    Sound domain owners (write store location replacement) meaning
      (invalidate owners location entries) := by
  intro entry member input valid_input
  have kept := List.mem_filter.mp member
  have framed : recordKey owners entry.register ≠ location := of_decide_eq_true kept.2
  simpa [write, framed] using sound entry kept.1 input valid_input

theorem filter_preserves_evidence {Input Value Claim : Type}
    (domain : Input → Prop) (owners : List Nat) (store : Store Input Value)
    (meaning : Claim → Input → Value) (entries : List (Entry Claim))
    (sound : Sound domain owners store meaning entries) (location : Key)
    (replacement : Input → Value) (keep : Entry Claim → Bool)
    (checked : ∀ entry, keep entry = true → recordKey owners entry.register ≠ location) :
    Sound domain owners (write store location replacement) meaning (entries.filter keep) := by
  intro entry member input valid_input
  have kept := List.mem_filter.mp member
  have framed := checked entry kept.2
  simpa [write, framed] using sound entry kept.1 input valid_input

def guardedWrite {Input Value : Type} (store : Store Input Value) (location : Key)
    (guard : Input → Bool) (replacement : Input → Value) : Store Input Value :=
  write store location fun input => if guard input then replacement input else store location input

theorem invalidation_preserves_guarded_evidence {Input Value Claim : Type}
    (domain : Input → Prop) (owners : List Nat) (store : Store Input Value)
    (meaning : Claim → Input → Value) (entries : List (Entry Claim))
    (sound : Sound domain owners store meaning entries) (location : Key)
    (guard : Input → Bool) (replacement : Input → Value) :
    Sound domain owners (guardedWrite store location guard replacement) meaning
      (invalidate owners location entries) :=
  retained_evidence_survives_write domain owners store meaning entries sound location _

@[step]
theorem source_retention_frames_written_location (thread : Usize) (owners : Slice U32)
    (register : Reg) (record : U32 × U8 × U16) :
    register_owner.record_aliases thread owners register record ⦃ result =>
      result = false ↔ recordKey (owners.val.map UScalar.val) record ≠
        canonicalKey thread.val (owners.val.map UScalar.val) register ⦄ := by
  step with rust_record_aliases thread owners register record as ⟨result, exact_decision⟩
  simp [exact_decision, eq_comm]

def witnessRecord : U32 × U8 × U16 := (0#u32, 4#u8, 0#u16)
def witnessRegister : Reg := ⟨.UP, 0#u16⟩
def witnessLocation : Key := canonicalKey 1 [0, 0] witnessRegister

theorem peer_uniform_write_aliases_existing_record :
    recordKey [0, 0] witnessRecord = witnessLocation := by
  simp [recordKey, canonicalKey, witnessRecord, witnessLocation, witnessRegister, shared, fileTag]

theorem per_thread_only_invalidation_is_unsound :
    let entries : List (Entry Bool) := [⟨witnessRecord, false⟩]
    let store : Store Unit Bool := fun _ _ => false
    let meaning : Bool → Unit → Bool := fun claim _ => claim
    Sound (fun _ => True) [0, 0] store meaning entries ∧
      ¬Sound (fun _ => True) [0, 0] (write store witnessLocation (fun _ => true)) meaning
        (entries.filter fun entry => entry.register.1 != 1#u32) := by
  constructor
  · intro entry member input valid_input
    have entry_eq : entry = (⟨witnessRecord, false⟩ : Entry Bool) := by simpa using member
    subst entry
    rfl
  · intro unsound
    have wrong := unsound ⟨witnessRecord, false⟩ (by simp [witnessRecord]) () trivial
    simp [write, peer_uniform_write_aliases_existing_record] at wrong

theorem captured_false_guard_does_not_justify_retaining_evidence :
    let entries : List (Entry Bool) := [⟨witnessRecord, false⟩]
    let store : Store Bool Bool := fun _ _ => false
    let meaning : Bool → Bool → Bool := fun claim _ => claim
    let next := guardedWrite store witnessLocation id (fun _ => true)
    Sound (fun _ => True) [0, 0] store meaning entries ∧
      next witnessLocation false = store witnessLocation false ∧
      ¬Sound (fun _ => True) [0, 0] next meaning entries := by
  refine ⟨?_, ?_, ?_⟩
  · intro entry member input valid_input
    have entry_eq : entry = (⟨witnessRecord, false⟩ : Entry Bool) := by simpa using member
    subst entry
    rfl
  · simp [guardedWrite, write]
  · intro unsound
    have wrong := unsound ⟨witnessRecord, false⟩ (by simp) true trivial
    simp [guardedWrite, write, peer_uniform_write_aliases_existing_record] at wrong

end RegisterOwner.Comparison
