/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ClockFilter

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

def readBit : Nat := 2 ^ 31

def readActor (actor : Nat) : Nat := if actor < readBit then actor + readBit else actor

theorem read_actor_matches_high_bit (actor : Nat) (bounded : actor < 2 * readBit) :
    readActor actor = actor ||| readBit := by
  by_cases low : actor < readBit
  · rw [readActor, ite_eq_left low]
    exact (Nat.or_two_pow_eq_add_of_lt low).symm
  · have partBound : actor - readBit < 2 ^ 31 := by unfold readBit at *; omega
    have shape : actor - readBit + readBit = actor := by omega
    have flagged := Nat.or_two_pow_eq_add_of_lt partBound
    have redundant : actor ||| readBit = actor := by
      calc
        actor ||| readBit = ((actor - readBit) ||| readBit) ||| readBit := by
          rw [show (actor - readBit) ||| readBit = actor from flagged.trans shape]
        _ = (actor - readBit) ||| readBit := by rw [Nat.or_assoc, Nat.or_self]
        _ = actor := flagged.trans shape
    simp only [readActor, low, ite_false, redundant]

def freshReads (entries : List Entry) : List Entry :=
  (entries.filter (fun e => decide (e.1 < readBit))).map (fun e => (e.1 + readBit, e.2))

def inheritedReads (entries : List Entry) : List Entry :=
  entries.filter (fun e => decide (readBit ≤ e.1))

def releaseEntries (entries : List Entry) (reader done : Nat) : List Entry :=
  overlay (mergeEntries (freshReads entries) (inheritedReads entries)) (readActor reader, done)

theorem fresh_reads_sorted (entries : List Entry) (ordered : Sorted entries) :
    Sorted (freshReads entries) := by
  exact List.Pairwise.map (fun e : Entry => (e.1 + readBit, e.2))
    (fun _ _ less => Nat.add_lt_add_right less readBit)
    (ordered.filter (fun e => decide (e.1 < readBit)))

theorem inherited_reads_sorted (entries : List Entry) (ordered : Sorted entries) :
    Sorted (inheritedReads entries) := ordered.filter _

theorem read_partition_length (entries : List Entry) :
    (freshReads entries).length + (inheritedReads entries).length = entries.length := by
  induction entries with
  | nil => simp [freshReads, inheritedReads]
  | cons entry rest ih =>
    by_cases low : entry.1 < readBit
    · have high : ¬readBit ≤ entry.1 := by omega
      simp only [freshReads, inheritedReads, List.filter_cons, low, high, decide_true,
        decide_false, Bool.false_eq_true, ite_true, ite_false, List.map_cons, List.length_cons] at ih ⊢
      omega
    · have high : readBit ≤ entry.1 := by omega
      simp only [freshReads, inheritedReads, List.filter_cons, low, high, decide_true,
        decide_false, Bool.false_eq_true, ite_true, ite_false, List.map_cons, List.length_cons] at ih ⊢
      omega

theorem read_partition_value (entries : List Entry) :
    BarrierReduction.join (entriesValue (freshReads entries)) (entriesValue (inheritedReads entries)) =
      entriesValue (entries.map (fun e => (readActor e.1, e.2))) := by
  induction entries with
  | nil => funext actor; simp [freshReads, inheritedReads, entriesValue, BarrierReduction.join]
  | cons entry rest ih =>
    funext actor
    have previous := congrFun ih actor
    by_cases low : entry.1 < readBit
    · have high : ¬readBit ≤ entry.1 := by omega
      simp only [freshReads, inheritedReads, List.filter_cons, low, high, decide_true,
        decide_false, Bool.false_eq_true, ite_true, ite_false, List.map_cons, entriesValue, readActor] at *
      simp only [BarrierReduction.join] at previous ⊢
      rw [←previous]
      omega
    · have high : readBit ≤ entry.1 := by omega
      simp only [freshReads, inheritedReads, List.filter_cons, low, high, decide_true,
        decide_false, Bool.false_eq_true, ite_true, ite_false, List.map_cons, entriesValue, readActor] at *
      simp only [BarrierReduction.join] at previous ⊢
      rw [←previous]
      omega

theorem release_entries_sorted (entries : List Entry) (reader done : Nat)
    (ordered : Sorted entries) : Sorted (releaseEntries entries reader done) :=
  overlay_sorted _ _ (merge_entries_sorted _ _
    (fresh_reads_sorted entries ordered) (inherited_reads_sorted entries ordered))

theorem release_entries_value (entries : List Entry) (reader done : Nat) :
    entriesValue (releaseEntries entries reader done) =
      value (entries.map (fun e => (readActor e.1, e.2))) (readActor reader, done) := by
  rw [releaseEntries, overlay_value]
  funext actor
  simp only [value, merge_entries_value, read_partition_value]

theorem select_tail_length (base : List Entry) (own : Entry) (lower : Nat) :
    (selectTail base own lower).length ≤ base.length + 1 := by
  have filtered := List.length_filter_le (p := fun e : Entry => decide (lower ≤ e.1)) base
  unfold selectTail
  split
  · change (overlay (base.filter (fun e => decide (lower ≤ e.1))) own).length ≤ base.length + 1
    have overlaid := overlay_length (base.filter (fun e => decide (lower ≤ e.1))) own
    omega
  · change (base.filter (fun e => decide (lower ≤ e.1))).length ≤ base.length + 1
    omega

theorem entries_value_map_shift (entries : List Entry) (shift actor : Nat) :
    entriesValue (entries.map (fun e => (e.1 + shift, e.2))) actor =
      if shift ≤ actor then entriesValue entries (actor - shift) else 0 := by
  induction entries with
  | nil => simp [entriesValue]
  | cons entry rest ih =>
    by_cases eligible : shift ≤ actor
    · have same : entry.1 + shift = actor ↔ entry.1 = actor - shift := by omega
      simp [entriesValue, ih, eligible, same]
    · have different : entry.1 + shift ≠ actor := by omega
      simp [entriesValue, ih, eligible, different]

def readReleaseValue (before : Nat → Nat) (lower reader done actor : Nat) : Nat :=
  max
    (max (if readBit ≤ actor ∧ lower ≤ actor then before actor else 0)
      (if readBit ≤ actor ∧ actor - readBit < readBit ∧ lower ≤ actor - readBit
        then before (actor - readBit) else 0))
    (if readActor reader = actor then done else 0)

theorem fresh_reads_value (entries : List Entry) (actor : Nat) :
    entriesValue (freshReads entries) actor =
      if readBit ≤ actor ∧ actor - readBit < readBit then entriesValue entries (actor - readBit) else 0 := by
  unfold freshReads
  rw [entries_value_map_shift]
  have selected := entries_value_filter_actor entries (fun key => decide (key < readBit)) (actor - readBit)
  simp only [decide_eq_true_eq] at selected
  rw [selected]
  split_ifs <;> simp_all

theorem inherited_reads_value (entries : List Entry) (actor : Nat) :
    entriesValue (inheritedReads entries) actor =
      if readBit ≤ actor then entriesValue entries actor else 0 := by
  simpa only [inheritedReads, decide_eq_true_eq] using
    entries_value_filter_actor entries (fun key => decide (readBit ≤ key)) actor

theorem release_entries_refines_read_projection (base : List Entry) (own : Entry)
    (lower reader done : Nat) :
    entriesValue (releaseEntries (selectTail base own lower) reader done) =
      readReleaseValue (value base own) lower reader done := by
  rw [releaseEntries, overlay_value]
  funext actor
  change max (entriesValue (mergeEntries (freshReads (selectTail base own lower))
    (inheritedReads (selectTail base own lower))) actor)
    (if readActor reader = actor then done else 0) = _
  rw [merge_entries_value]
  change max (max (entriesValue (freshReads (selectTail base own lower)) actor)
    (entriesValue (inheritedReads (selectTail base own lower)) actor))
    (if readActor reader = actor then done else 0) = _
  rw [fresh_reads_value, inherited_reads_value]
  simp only [select_tail_value]
  unfold readReleaseValue
  split_ifs <;> omega

end MemoryWrite.SparseClock
