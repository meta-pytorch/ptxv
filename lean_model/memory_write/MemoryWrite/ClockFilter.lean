/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ClockMeet

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.SparseClock

def tailEntries (base : List Entry) (own : Entry) (lower : Nat) : List Entry :=
  (overlay base own).filter (fun entry => decide (lower ≤ entry.1))

theorem filter_preserves_sortedness (base : List Entry) (keep : Entry → Bool)
    (sorted : Sorted base) : Sorted (base.filter keep) := sorted.filter keep

theorem entries_value_filter_actor (base : List Entry) (keep : Nat → Bool) (actor : Nat) :
    entriesValue (base.filter (fun entry => keep entry.1)) actor =
      if keep actor then entriesValue base actor else 0 := by
  induction base with
  | nil => simp [entriesValue]
  | cons head tail ih =>
    by_cases same : head.1 = actor
    · cases ha : keep actor <;> simp [entriesValue, same, ha, ih]
    · cases h : keep head.1 <;> cases ha : keep actor <;>
        simp [entriesValue, same, ih, h, ha]

theorem tail_entries_sorted (base : List Entry) (own : Entry) (lower : Nat)
    (sorted : Sorted base) : Sorted (tailEntries base own lower) :=
  filter_preserves_sortedness _ _ (overlay_sorted base own sorted)

theorem tail_entries_value (base : List Entry) (own : Entry) (lower actor : Nat) :
    entriesValue (tailEntries base own lower) actor =
      if lower ≤ actor then value base own actor else 0 := by
  unfold tailEntries
  rw [entries_value_filter_actor (overlay base own) (fun key => decide (lower ≤ key)) actor]
  simp only [decide_eq_true_eq, overlay_value]

theorem located_filter_eq_drop (base : List Entry) (actor index : Nat) (found : Bool)
    (located : Located base actor index found) :
    base.filter (fun entry => decide (actor ≤ entry.1)) = base.drop index := by
  induction base generalizing index with
  | nil => simp
  | cons head tail ih =>
    cases index with
    | zero =>
      have all : ∀ e ∈ head :: tail, actor ≤ e.1 := by
        intro e member
        obtain ⟨k, hk, rfl⟩ := List.mem_iff_getElem.mp member
        cases found with
        | false => exact Nat.le_of_lt (located.2.2 k hk (by omega))
        | true =>
          obtain ⟨hi, equal, after⟩ := located.2.2
          by_cases zero : k = 0
          · simpa only [zero, List.getElem_cons_zero] using Nat.le_of_eq equal.symm
          · exact Nat.le_of_lt (after k hk (by omega))
      change (head :: tail).filter (fun entry => decide (actor ≤ entry.1)) = head :: tail
      apply List.filter_eq_self.mpr
      intro e member
      simpa only [decide_eq_true_eq] using all e member
    | succ index =>
      have before : head.1 < actor := located.2.1 0 (by simp) (by omega)
      have no : ¬actor ≤ head.1 := by omega
      simp only [List.filter_cons, no, decide_false, Bool.false_eq_true, ite_false,
        List.drop_succ_cons]
      exact ih index (located_tail head tail actor index found located)

def selectTail (base : List Entry) (own : Entry) (lower : Nat) : List Entry :=
  let selected := base.filter (fun entry => decide (lower ≤ entry.1))
  if lower ≤ own.1 then overlay selected own else selected

theorem select_tail_sorted (base : List Entry) (own : Entry) (lower : Nat)
    (sorted : Sorted base) : Sorted (selectTail base own lower) := by
  have filtered := filter_preserves_sortedness base (fun entry => decide (lower ≤ entry.1)) sorted
  unfold selectTail
  split
  · exact overlay_sorted _ _ filtered
  · exact filtered

theorem select_tail_value (base : List Entry) (own : Entry) (lower actor : Nat) :
    entriesValue (selectTail base own lower) actor =
      if lower ≤ actor then value base own actor else 0 := by
  have projected := entries_value_filter_actor base (fun key => decide (lower ≤ key)) actor
  simp only [decide_eq_true_eq] at projected
  unfold selectTail
  by_cases included : lower ≤ own.1
  · simp only [included, ite_true, overlay_value, value, projected]
    by_cases actorIncluded : lower ≤ actor
    · simp [actorIncluded]
    · have different : own.1 ≠ actor := by omega
      simp [actorIncluded, different]
  · simp only [included, ite_false, projected, value]
    by_cases actorIncluded : lower ≤ actor
    · have different : own.1 ≠ actor := by omega
      simp [actorIncluded, different]
    · simp [actorIncluded]

def novelEntries (entries : List Entry) (seen : Nat → Nat) : List Entry :=
  entries.filter (fun entry => decide (seen entry.1 < entry.2))

theorem novel_entries_sorted (entries : List Entry) (seen : Nat → Nat)
    (sorted : Sorted entries) : Sorted (novelEntries entries seen) :=
  filter_preserves_sortedness _ _ sorted

theorem novel_entries_value (entries : List Entry) (seen : Nat → Nat) (actor : Nat)
    (sorted : Sorted entries) :
    entriesValue (novelEntries entries seen) actor =
      if seen actor < entriesValue entries actor then entriesValue entries actor else 0 := by
  induction entries with
  | nil => simp [novelEntries, entriesValue]
  | cons head tail ih =>
    have order := List.pairwise_cons.mp sorted
    by_cases same : head.1 = actor
    · have tailZero := entries_value_of_strict_lower_bound tail actor (by simpa only [same] using order.1)
      have filteredZero := entries_value_of_strict_lower_bound (novelEntries tail seen) actor (by
        intro e member
        exact (by simpa only [same] using order.1 e (List.mem_filter.mp member).1))
      by_cases included : seen actor < head.2
      · simp [novelEntries, entriesValue, same, included, tailZero] at filteredZero ⊢
        omega
      · simp [novelEntries, entriesValue, same, included, tailZero] at filteredZero ⊢
        omega
    · have inductionValue := ih order.2
      cases included : decide (seen head.1 < head.2) <;>
        simpa [novelEntries, entriesValue, same, included] using inductionValue


end MemoryWrite.SparseClock
