/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.SparseClock

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

def meetEntries : List Entry → List Entry → List Entry
  | [], _ => []
  | _, [] => []
  | first :: left, second :: right =>
    if first.1 < second.1 then meetEntries left (second :: right)
    else if second.1 < first.1 then meetEntries (first :: left) right
    else
      if min first.2 second.2 = 0 then meetEntries left right
      else (first.1, min first.2 second.2) :: meetEntries left right
termination_by left right => left.length + right.length

theorem meet_entries_sublist (left right : List Entry) :
    List.Sublist ((meetEntries left right).map Prod.fst) (left.map Prod.fst) := by
  fun_induction meetEntries left right with
  | case1 right => simp
  | case2 left h => simp
  | case3 first left second right hl ih => exact ih.cons _
  | case4 first left second right hn hl ih => exact ih
  | case5 first left second right hn hn' hz ih => exact ih.cons _
  | case6 first left second right hn hn' hz ih => exact ih.cons_cons _

theorem meet_entries_sorted (left right : List Entry) (sorted : Sorted left) :
    Sorted (meetEntries left right) := by
  have sub := meet_entries_sublist left right
  have keys : (left.map Prod.fst).Pairwise (· < ·) := by simpa [List.pairwise_map, Sorted] using sorted
  simpa [List.pairwise_map, Sorted] using keys.sublist sub

theorem entries_value_of_strict_lower_bound (entries : List Entry) (actor : Nat)
    (bound : ∀ e ∈ entries, actor < e.1) : entriesValue entries actor = 0 := by
  induction entries with
  | nil => rfl
  | cons head tail ih =>
    have neq : head.1 ≠ actor := by have := bound head (by simp); omega
    simp only [entriesValue, neq, ite_false, Nat.zero_max]
    apply ih
    intro e he; exact bound e (by simp [he])

theorem meet_entries_value (left right : List Entry) (leftSorted : Sorted left)
    (rightSorted : Sorted right) :
    entriesValue (meetEntries left right) = fun actor =>
      min (entriesValue left actor) (entriesValue right actor) := by
  funext actor
  fun_induction meetEntries left right with
  | case1 right => simp [entriesValue]
  | case2 left h => simp [entriesValue]
  | case3 first left second right hl ih =>
    have leftTail := (List.pairwise_cons.mp leftSorted).2
    have equality := ih leftTail rightSorted
    by_cases same : first.1 = actor
    · have rightZero := entries_value_of_strict_lower_bound (second :: right) actor (by
        intro e member
        rcases List.mem_cons.mp member with rfl | member
        · omega
        · exact Nat.lt_trans (by omega) ((List.pairwise_cons.mp rightSorted).1 e member))
      simp only [rightZero, Nat.min_zero] at equality ⊢
      exact equality
    · simpa only [entriesValue, same, ite_false, Nat.zero_max] using equality
  | case4 first left second right hn hl ih =>
    have rightTail := (List.pairwise_cons.mp rightSorted).2
    have equality := ih leftSorted rightTail
    by_cases same : second.1 = actor
    · have leftZero := entries_value_of_strict_lower_bound (first :: left) actor (by
        intro e member
        rcases List.mem_cons.mp member with rfl | member
        · omega
        · exact Nat.lt_trans (by omega) ((List.pairwise_cons.mp leftSorted).1 e member))
      simp only [leftZero, Nat.zero_min] at equality ⊢
      exact equality
    · simpa only [entriesValue, same, ite_false, Nat.zero_max] using equality
  | case5 first left second right hn hn' hz ih =>
    have sameKey : first.1 = second.1 := by omega
    have lt := List.pairwise_cons.mp leftSorted
    have rt := List.pairwise_cons.mp rightSorted
    have equality := ih lt.2 rt.2
    by_cases same : first.1 = actor
    · have leftZero := entries_value_of_strict_lower_bound left actor (by simpa only [same] using lt.1)
      have rightZero := entries_value_of_strict_lower_bound right actor (by simpa only [←sameKey, same] using rt.1)
      simpa [entriesValue, same, ←sameKey, leftZero, rightZero, hz] using equality
    · simpa [entriesValue, same, ←sameKey] using equality
  | case6 first left second right hn hn' hz ih =>
    have sameKey : first.1 = second.1 := by omega
    have lt := List.pairwise_cons.mp leftSorted
    have rt := List.pairwise_cons.mp rightSorted
    have equality := ih lt.2 rt.2
    by_cases same : first.1 = actor
    · have leftZero := entries_value_of_strict_lower_bound left actor (by simpa only [same] using lt.1)
      have rightZero := entries_value_of_strict_lower_bound right actor (by simpa only [←sameKey, same] using rt.1)
      have zero : entriesValue (meetEntries left right) actor = 0 := by
        simpa only [leftZero, rightZero, Nat.min_self] using equality
      simp [entriesValue, same, ←sameKey, leftZero, rightZero, zero]
    · simpa [entriesValue, same, ←sameKey] using equality

end MemoryWrite.SparseClock
