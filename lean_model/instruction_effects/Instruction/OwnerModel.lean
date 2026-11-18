/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Types

open Aeneas Aeneas.Std PtxvIr

set_option autoImplicit false

namespace RegisterOwner

def shared : File → Bool
  | .UR | .UP => true
  | _ => false

def fileTag : File → Nat
  | .V => 0
  | .R => 1
  | .UR => 2
  | .P => 3
  | .UP => 4

theorem file_tag_injective (left right : File) : fileTag left = fileTag right ↔ left = right := by
  cases left <;> cases right <;> simp [fileTag]

structure Key where
  owner : Nat
  file : Nat
  index : Nat
  deriving DecidableEq

def canonicalKey (thread : Nat) (owners : List Nat) (register : Reg) : Key :=
  ⟨if shared register.file then owners[thread]?.getD (thread % 2 ^ 32)
    else thread % 2 ^ 32, fileTag register.file, register.idx.val⟩

theorem canonical_key_namespace (thread : Nat) (owners : List Nat) (left right : Reg)
    (equal : canonicalKey thread owners left = canonicalKey thread owners right) :
    left.file = right.file ∧ left.idx = right.idx := by
  have hf := congrArg Key.file equal
  have hi := congrArg Key.index equal
  exact ⟨(file_tag_injective left.file right.file).mp hf, UScalar.eq_of_val_eq hi⟩

def firstOwner (warps : List U32) (warp : U32) : Option Nat :=
  if warp ∈ warps then some (warps.idxOf warp) else none

def ownerTable (warps : List U32) : List Nat := warps.map warps.idxOf

def LeadersSound (warps : List U32) (first : List (Option U32)) : Prop :=
  ∀ warp : U32, (h : warp.val < first.length) →
    (first[warp.val]'h).map UScalar.val = firstOwner warps warp

theorem empty_leaders_sound (count : Nat) :
    LeadersSound [] (List.replicate count none) := by
  intro warp h
  simp [firstOwner]

@[simp]
theorem owner_table_length (warps : List U32) : (ownerTable warps).length = warps.length := by
  simp [ownerTable]

theorem first_owner_append (warps : List U32) (added warp : U32) :
    firstOwner (warps ++ [added]) warp =
      if warp ∈ warps then firstOwner warps warp
      else if added = warp then some warps.length else none := by
  by_cases hmem : warp ∈ warps
  · simp [firstOwner, List.idxOf_append, hmem]
  · by_cases heq : added = warp
    · subst added
      simp [firstOwner, List.idxOf_append, hmem]
    · simp [firstOwner, hmem, heq, Ne.symm heq]

theorem owner_table_append (warps : List U32) (warp : U32) :
    ownerTable (warps ++ [warp]) = ownerTable warps ++
      [if warp ∈ warps then warps.idxOf warp else warps.length] := by
  unfold ownerTable
  rw [List.map_append]
  congr 1
  · apply List.map_congr_left
    intro value hvalue
    simp [List.idxOf_append, hvalue]
  · simp [List.idxOf_append]

theorem first_owner_bounded (warps : List U32) (warp : U32) (owner : Nat)
    (h : firstOwner warps warp = some owner) : owner < warps.length := by
  unfold firstOwner at h
  split at h
  · rename_i hmem
    simp only [Option.some.injEq] at h
    subst owner
    exact List.idxOf_lt_length_of_mem hmem
  · simp at h

theorem first_owner_stable_append (warps : List U32) (added warp : U32)
    (present : added ∈ warps) :
    firstOwner (warps ++ [added]) warp = firstOwner warps warp := by
  by_cases hmem : warp ∈ warps
  · simp [first_owner_append, hmem]
  · have different : added ≠ warp := by intro h; subst added; exact hmem present
    have hval : warp.val ≠ added.val := by
      intro h
      exact different (UScalar.eq_of_val_eq h.symm)
    simp [firstOwner, hmem, hval]

theorem leaders_existing (warps : List U32) (first : List (Option U32))
    (warp owner : U32) (hw : warp.val < first.length)
    (sound : LeadersSound warps first) (slot : first[warp.val] = some owner) :
    warp ∈ warps ∧ owner.val = warps.idxOf warp ∧
      LeadersSound (warps ++ [warp]) first := by
  have howner := sound warp hw
  rw [slot] at howner
  simp only [Option.map_some] at howner
  have present : warp ∈ warps := by
    by_contra absent
    simp [firstOwner, absent] at howner
  have hvalue : owner.val = warps.idxOf warp := by
    simpa [firstOwner, present] using howner
  refine ⟨present, hvalue, ?_⟩
  intro other ho
  rw [first_owner_stable_append warps warp other present]
  exact sound other ho

theorem leaders_new (warps : List U32) (first : List (Option U32))
    (warp owner : U32) (hw : warp.val < first.length)
    (sound : LeadersSound warps first) (slot : first[warp.val] = none)
    (fresh : owner.val = warps.length) :
    warp ∉ warps ∧ LeadersSound (warps ++ [warp]) (first.set warp.val (some owner)) := by
  have absent : warp ∉ warps := by
    intro present
    have h := sound warp hw
    simp [slot, firstOwner, present] at h
  refine ⟨absent, ?_⟩
  intro other ho
  have holt : other.val < first.length := by simpa using ho
  by_cases equal : other = warp
  · subst other
    simp [first_owner_append, absent, fresh]
  · have different : warp.val ≠ other.val := by
      intro h
      exact equal (UScalar.eq_of_val_eq h.symm)
    simp only [List.getElem_set, ite_eq_right different]
    rw [sound other holt]
    by_cases present : other ∈ warps
    · simp [first_owner_append, present]
    · simp [firstOwner, present, Ne.symm different]

theorem owner_table_alias_iff (warps : List U32) (left right : Nat)
    (hl : left < warps.length) (hr : right < warps.length) :
    (ownerTable warps)[left]'(by simpa using hl) =
      (ownerTable warps)[right]'(by simpa using hr) ↔ warps[left] = warps[right] := by
  simp only [ownerTable, List.getElem_map]
  constructor
  · intro h
    have hlmem : warps[left] ∈ warps := List.getElem_mem hl
    have hrmem : warps[right] ∈ warps := List.getElem_mem hr
    have hleft := List.getElem_idxOf (List.idxOf_lt_length_of_mem hlmem)
    have hright := List.getElem_idxOf (List.idxOf_lt_length_of_mem hrmem)
    have hview := congrArg (fun index => warps[index]?) h
    simpa only [List.getElem?_eq_getElem (List.idxOf_lt_length_of_mem hlmem),
      List.getElem?_eq_getElem (List.idxOf_lt_length_of_mem hrmem),
      hleft, hright, Option.some.injEq] using hview
  · intro h
    rw [h]

def ValidWarps (warps : List U32) (count : Nat) : Prop :=
  ∀ warp ∈ warps, warp.val < count

def OwnerLoopOutcome (warps : List U32) (count : Nat)
    (result : core.result.Result (alloc.vec.Vec U32) register_owner.TopologyError) : Prop :=
  match result with
  | .Ok owners => owners.val.map UScalar.val = ownerTable warps ∧ ValidWarps warps count
  | .Err .InvalidWarp => ∃ warp ∈ warps, count ≤ warp.val
  | .Err _ => False

structure LoopInvariant (warps : Slice U32) (count : Usize)
    (first : alloc.vec.Vec (Option U32)) (owners : alloc.vec.Vec U32) (thread : Usize) : Prop where
  thread_bound : thread.val ≤ warps.val.length
  first_length : first.val.length = count.val
  owners_length : owners.val.length = thread.val
  owners_sound : owners.val.map UScalar.val = ownerTable (warps.val.take thread.val)
  leaders_sound : LeadersSound (warps.val.take thread.val) first.val
  valid_prefix : ValidWarps (warps.val.take thread.val) count.val


def OwnerOutcome (warps : List U32) (count : Nat)
    (result : core.result.Result (alloc.vec.Vec U32) register_owner.TopologyError) : Prop :=
  if 2 ^ 32 - 1 < warps.length then result = .Err .TooManyThreads
  else if warps.length < count then result = .Err .TooManyWarps
  else OwnerLoopOutcome warps count result

def recordKey (owners : List Nat) (record : U32 × U8 × U16) : Key :=
  ⟨if record.2.1.val = 2 ∨ record.2.1.val = 4
    then owners[record.1.val]?.getD record.1.val else record.1.val,
    record.2.1.val, record.2.2.val⟩

theorem record_key_agrees (owners : List Nat) (record : U32 × U8 × U16) (register : Reg)
    (same_file : fileTag register.file = record.2.1.val)
    (same_index : register.idx.val = record.2.2.val) :
    recordKey owners record = canonicalKey record.1.val owners register := by
  have bounded : record.1.val < 4294967296 := by scalar_tac
  cases register with
  | mk file index =>
    cases file <;>
      simp_all [recordKey, canonicalKey, shared, fileTag, Nat.mod_eq_of_lt bounded]

end RegisterOwner
