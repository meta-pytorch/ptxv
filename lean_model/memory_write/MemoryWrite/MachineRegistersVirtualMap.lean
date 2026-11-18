/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineRegistersVirtual

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineRegistersVirtualMap

open MachineRegistersVirtual

abbrev PartialMask (mask : U64) : Prop := mask ≠ 0#u64 ∧ mask ≠ core.num.U64.MAX

def Sorted (entries : List Entry) : Prop :=
  entries.Pairwise (fun left right => left.1.val < right.1.val)

def Canonical (entries : List Entry) : Prop :=
  Sorted entries ∧ ∀ entry ∈ entries, PartialMask entry.2

def lookup (entries : List Entry) (key : U16) : Option U64 :=
  match entries with
  | [] => none
  | (entryKey, mask) :: rest => if entryKey = key then some mask else lookup rest key

def update (entries : List Entry) (key : U16) (mask : U64) : List Entry :=
  match entries with
  | [] => if PartialMask mask then [(key, mask)] else []
  | entry :: rest =>
    if entry.1.val < key.val then entry :: update rest key mask
    else if entry.1 = key then
      if PartialMask mask then (key, mask) :: rest else rest
    else if PartialMask mask then (key, mask) :: entry :: rest else entry :: rest

def LowerBound (entries : List Entry) (key : U16) (position : Nat) : Prop :=
  position ≤ entries.length ∧
  (∀ i, i < position → entries[i]!.1.val < key.val) ∧
  (position < entries.length → key.val ≤ entries[position]!.1.val)

theorem erase_from_past (entries : List Entry) (skipIndex start : Nat)
    (past : skipIndex < start) : eraseFrom entries skipIndex start = entries := by
  induction entries generalizing start with
  | nil => rfl
  | cons entry rest ih =>
    have distinct : start ≠ skipIndex := by omega
    simp only [eraseFrom, distinct, ite_false]
    rw [ih (start + 1) (by omega)]

theorem erase_from_eq (entries : List Entry) (skipIndex start : Nat) :
    eraseFrom entries (skipIndex + start) start = entries.eraseIdx skipIndex := by
  induction entries generalizing skipIndex start with
  | nil => simp [eraseFrom]
  | cons entry rest ih =>
    cases skipIndex with
    | zero =>
      simp only [Nat.zero_add, eraseFrom, ite_true, List.eraseIdx_cons_zero]
      exact erase_from_past rest start (start + 1) (by omega)
    | succ skipIndex =>
      have distinct : start ≠ skipIndex + 1 + start := by omega
      simp only [eraseFrom, distinct, ite_false]
      have shifted : skipIndex + 1 + start = skipIndex + (start + 1) := by omega
      rw [shifted, ih]
      rfl

theorem erase_from_zero (entries : List Entry) (position : Nat) :
    eraseFrom entries position 0 = entries.eraseIdx position := by
  simpa using erase_from_eq entries position 0

theorem lower_bound_tail (entry : Entry) (rest : List Entry) (key : U16) (position : Nat)
    (located : LowerBound (entry :: rest) key (position + 1)) :
    entry.1.val < key.val ∧ LowerBound rest key position := by
  rcases located with ⟨bound, earlier, after⟩
  constructor
  · simpa using earlier 0 (by omega)
  refine ⟨by simpa using bound, ?_, ?_⟩
  · intro i before
    simpa using earlier (i + 1) (by omega)
  · intro inside
    simpa using after (by simpa using inside)

theorem has_key_succ (entry : Entry) (rest : List Entry) (key : U16) (position : Nat) :
    hasKey (entry :: rest) key (position + 1) = hasKey rest key position := by rfl

theorem update_at_eq (entries : List Entry) (key : U16) (mask : U64) (position : Nat)
    (located : LowerBound entries key position) :
    updateAt entries key mask position = update entries key mask := by
  induction entries generalizing position with
  | nil =>
    have atZero : position = 0 := by simpa [LowerBound] using located.1
    subst position
    by_cases proper : PartialMask mask <;>
      simp [updateAt, hasKey, update, PartialMask] at *
  | cons entry rest ih =>
    cases position with
    | zero =>
      have first : key.val ≤ entry.1.val := by simpa using located.2.2 (by simp)
      have notBefore : ¬entry.1.val < key.val := by omega
      by_cases equal : entry.1 = key <;> by_cases proper : PartialMask mask <;>
        simp [updateAt, hasKey, update, notBefore, equal, proper, PartialMask,
          erase_from_zero] at *
    | succ position =>
      obtain ⟨before, tail⟩ := lower_bound_tail entry rest key position located
      rw [update, ite_eq_left before, ←ih position tail]
      simp only [updateAt, has_key_succ, erase_from_zero]
      by_cases enabled : (mask != 0#u64 && mask != core.num.U64.MAX) = true <;>
        simp only [enabled, ↓reduceIte]
      all_goals
        by_cases found : hasKey rest key position = true <;>
          simp only [found, ↓reduceIte] <;> rfl

theorem lookup_none (entries : List Entry) (key : U16)
    (absent : ∀ entry ∈ entries, entry.1 ≠ key) : lookup entries key = none := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
    rcases entry with ⟨entryKey, mask⟩
    have distinct : entryKey ≠ key := absent (entryKey, mask) (by simp)
    simp only [lookup, distinct, ite_false]
    apply ih
    intro candidate member
    exact absent candidate (by simp [member])

theorem update_membership (entries : List Entry) (key : U16) (mask : U64) (candidate : Entry)
    (included : candidate ∈ update entries key mask) :
    (candidate = (key, mask) ∧ PartialMask mask) ∨ candidate ∈ entries := by
  induction entries with
  | nil =>
    rw [update] at included
    split at included
    · rename_i proper
      exact Or.inl ⟨List.mem_singleton.mp included, proper⟩
    · simp only [List.not_mem_nil] at included
  | cons entry rest ih =>
    rw [update] at included
    split at included
    · rcases List.mem_cons.mp included with same | tail
      · right; simp [same]
      · rcases ih tail with fresh | old
        · exact Or.inl fresh
        · right; simp [old]
    · split at included
      · split at included
        · rename_i proper
          rcases List.mem_cons.mp included with fresh | old
          · exact Or.inl ⟨fresh, proper⟩
          · right; simp [old]
        · right; simp [included]
      · split at included
        · rename_i proper
          rcases List.mem_cons.mp included with fresh | old
          · exact Or.inl ⟨fresh, proper⟩
          · exact Or.inr old
        · exact Or.inr included

theorem update_sorted (entries : List Entry) (key : U16) (mask : U64)
    (sorted : Sorted entries) : Sorted (update entries key mask) := by
  induction entries with
  | nil => rw [update]; split <;> simp [Sorted]
  | cons entry rest ih =>
    obtain ⟨headLess, tailSorted⟩ := List.pairwise_cons.mp sorted
    rw [update]
    split
    · rename_i before
      apply List.pairwise_cons.mpr
      refine ⟨?_, ih tailSorted⟩
      intro candidate included
      rcases update_membership rest key mask candidate included with fresh | old
      · simpa [fresh.1] using before
      · exact headLess candidate old
    · rename_i notBefore
      split
      · rename_i equal
        split
        · apply List.pairwise_cons.mpr
          refine ⟨?_, tailSorted⟩
          simpa only [equal] using headLess
        · exact tailSorted
      · rename_i distinct
        split
        · apply List.pairwise_cons.mpr
          refine ⟨?_, sorted⟩
          intro candidate included
          have keyBefore : key.val < entry.1.val := by scalar_tac
          rcases List.mem_cons.mp included with same | later
          · simpa [same] using keyBefore
          · exact Nat.lt_trans keyBefore (headLess candidate later)
        · exact sorted

theorem update_canonical (entries : List Entry) (key : U16) (mask : U64)
    (canonical : Canonical entries) : Canonical (update entries key mask) := by
  refine ⟨update_sorted entries key mask canonical.1, ?_⟩
  intro candidate included
  rcases update_membership entries key mask candidate included with fresh | old
  · simpa [fresh.1] using fresh.2
  · exact canonical.2 candidate old

theorem update_lookup_other (entries : List Entry) (key other : U16) (mask : U64)
    (distinct : other ≠ key) : lookup (update entries key mask) other = lookup entries other := by
  induction entries with
  | nil =>
    rw [update]
    split <;> simp only [lookup, ite_eq_right distinct.symm]
  | cons entry rest ih =>
    rcases entry with ⟨entryKey, entryMask⟩
    rw [update]
    split
    · simp only [lookup]
      split <;> simp_all only
    · split
      · rename_i equal
        change entryKey = key at equal
        rw [equal]
        split <;> simp only [lookup, ite_eq_right distinct.symm]
      · split <;> simp only [lookup, ite_eq_right distinct.symm]

theorem update_lookup_selected (entries : List Entry) (key : U16) (mask : U64)
    (sorted : Sorted entries) :
    lookup (update entries key mask) key = if PartialMask mask then some mask else none := by
  induction entries with
  | nil =>
    rw [update]
    split <;> simp_all only [lookup, ↓reduceIte]
  | cons entry rest ih =>
    rcases entry with ⟨entryKey, entryMask⟩
    obtain ⟨headLess, tailSorted⟩ := List.pairwise_cons.mp sorted
    rw [update]
    split
    · rename_i before
      have distinct : entryKey ≠ key := by scalar_tac
      simpa only [lookup, distinct, ite_false] using ih tailSorted
    · split
      · rename_i equal
        have absent : lookup rest key = none := by
          apply lookup_none
          intro candidate included
          have later := headLess candidate included
          scalar_tac
        by_cases proper : PartialMask mask
        · simp only [ite_eq_left proper, lookup, ite_true]
        · simp only [ite_eq_right proper, absent]
      · rename_i distinct
        have absent : lookup ((entryKey, entryMask) :: rest) key = none := by
          apply lookup_none
          intro candidate included
          rcases List.mem_cons.mp included with same | later
          · simpa [same] using distinct
          · have after := headLess candidate later
            scalar_tac
        by_cases proper : PartialMask mask
        · simp only [ite_eq_left proper, lookup, ite_true]
        · simp only [ite_eq_right proper, absent]

theorem sorted_unique (entries : List Entry) (sorted : Sorted entries) :
    (entries.map Prod.fst).Nodup := by
  rw [List.nodup_iff_pairwise_ne, List.pairwise_map]
  apply sorted.imp
  intro left right less same
  have values := congrArg UScalar.val same
  omega

theorem canonical_empty : Canonical [] := by simp [Canonical, Sorted]

theorem source_set_undefined_map (registers : machine_registers.LaneRegisters)
    (key : U16) (mask : U64) (canonical : Canonical registers.vund.val)
    (room : registers.vund.val.length < Usize.max) :
    machine_registers.set_virtual_undefined registers key mask ⦃ updated =>
      updated.regs = registers.regs ∧ updated.preds = registers.preds ∧
      updated.pdef = registers.pdef ∧ updated.written = registers.written ∧
      updated.vund.val = update registers.vund.val key mask ∧
      Canonical updated.vund.val ∧ (updated.vund.val.map Prod.fst).Nodup ∧
      lookup updated.vund.val key = (if PartialMask mask then some mask else none) ∧
      ∀ other, other ≠ key → lookup updated.vund.val other = lookup registers.vund.val other ⦄ := by
  apply WP.spec_mono (source_set_undefined registers key mask room)
  intro updated changed
  rcases changed with ⟨words, predicates, definitions, markers, position, located, sparse⟩
  have model : updated.vund.val = update registers.vund.val key mask := by
    rw [sparse]
    exact update_at_eq registers.vund.val key mask position.val located
  refine ⟨words, predicates, definitions, markers, model, ?_, ?_, ?_, ?_⟩
  · rw [model]
    exact update_canonical registers.vund.val key mask canonical
  · rw [model]
    exact sorted_unique _ (update_sorted registers.vund.val key mask canonical.1)
  · rw [model]
    exact update_lookup_selected registers.vund.val key mask canonical.1
  · intro other distinct
    rw [model]
    exact update_lookup_other registers.vund.val key other mask distinct

theorem source_write_virtual_map (registers : machine_registers.LaneRegisters)
    (index : U16) (value undefined : U64) (canonical : Canonical registers.vund.val)
    (wordBound : index.val < registers.regs.val.length)
    (writtenBound : index.val / 64 < registers.written.val.length)
    (room : registers.vund.val.length < Usize.max) :
    machine_registers.write_virtual registers index value undefined ⦃ updated =>
      updated.regs.val = registers.regs.val.set index.val value ∧
      updated.written.val = registers.written.val.set (index.val / 64)
        (MachineRegisters.bitWord registers.written.val[index.val / 64]! index
          (undefined != core.num.U64.MAX)) ∧
      updated.preds = registers.preds ∧ updated.pdef = registers.pdef ∧
      Canonical updated.vund.val ∧ (updated.vund.val.map Prod.fst).Nodup ∧
      lookup updated.vund.val index = (if PartialMask undefined then some undefined else none) ∧
      ∀ other, other ≠ index → lookup updated.vund.val other = lookup registers.vund.val other ⦄ := by
  apply WP.spec_mono (source_write_virtual registers index value undefined wordBound writtenBound room)
  intro updated changed
  rcases changed with ⟨words, markers, predicates, definitions, position, located, sparse⟩
  have model : updated.vund.val = update registers.vund.val index undefined := by
    rw [sparse]
    exact update_at_eq registers.vund.val index undefined position.val located
  refine ⟨words, markers, predicates, definitions, ?_, ?_, ?_, ?_⟩
  · rw [model]
    exact update_canonical registers.vund.val index undefined canonical
  · rw [model]
    exact sorted_unique _ (update_sorted registers.vund.val index undefined canonical.1)
  · rw [model]
    exact update_lookup_selected registers.vund.val index undefined canonical.1
  · intro other distinct
    rw [model]
    exact update_lookup_other registers.vund.val index other undefined distinct

end MemoryWrite.MachineRegistersVirtualMap
