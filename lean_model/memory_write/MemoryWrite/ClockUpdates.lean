/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ClockFilter

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

def WeakSorted (entries : List Entry) : Prop :=
  entries.Pairwise (fun a b => a.1 ≤ b.1)

def pushCoalesced (base : List Entry) (entry : Entry) : List Entry :=
  if h : 0 < base.length then
    let last := base[base.length - 1]
    if last.1 = entry.1 then
      base.set (base.length - 1) (last.1, max last.2 entry.2)
    else base ++ [entry]
  else base ++ [entry]

def coalesceInto : List Entry → List Entry → List Entry
  | base, [] => base
  | base, entry :: rest => coalesceInto (pushCoalesced base entry) rest

@[simp] theorem push_coalesced_nil (entry : Entry) : pushCoalesced [] entry = [entry] := by
  simp [pushCoalesced]

@[simp] theorem push_coalesced_snoc (base : List Entry) (last entry : Entry) :
    pushCoalesced (base ++ [last]) entry =
      if last.1 = entry.1 then base ++ [(last.1, max last.2 entry.2)]
      else base ++ [last, entry] := by
  simp [pushCoalesced, List.getElem_append_right, List.set_append_right,
    List.append_assoc]

theorem push_coalesced_has_last_key (base : List Entry) (entry : Entry) :
    ∃ front last, pushCoalesced base entry = front ++ [last] ∧ last.1 = entry.1 := by
  induction base using List.reverseRecOn with
  | nil => exact ⟨[], entry, by simp, rfl⟩
  | append_singleton base last _ =>
    rw [push_coalesced_snoc]
    split
    · rename_i same
      exact ⟨base, (last.1, max last.2 entry.2), rfl, same⟩
    · exact ⟨base ++ [last], entry, by simp, rfl⟩

theorem coalesce_into_increasing_suffix (front : List Entry) (last : Entry) (suffix : List Entry)
    (ordered : Sorted suffix) (lower : ∀ e ∈ suffix, last.1 < e.1) :
    coalesceInto (front ++ [last]) suffix = (front ++ [last]) ++ suffix := by
  induction suffix generalizing front last with
  | nil => simp [coalesceInto]
  | cons entry rest ih =>
    have different : last.1 ≠ entry.1 := Nat.ne_of_lt (lower entry (by simp))
    have parts := List.pairwise_cons.mp ordered
    rw [coalesceInto, push_coalesced_snoc, ite_eq_right different]
    simpa only [List.append_assoc, List.cons_append, List.nil_append] using
      ih (front ++ [last]) entry parts.2 parts.1

theorem coalesce_into_sorted_suffix (base : List Entry) (first : Entry) (rest : List Entry)
    (ordered : Sorted (first :: rest)) :
    coalesceInto base (first :: rest) = pushCoalesced base first ++ rest := by
  obtain ⟨front, last, shape, key⟩ := push_coalesced_has_last_key base first
  rw [coalesceInto, shape]
  exact coalesce_into_increasing_suffix front last rest (List.pairwise_cons.mp ordered).2
    (by simpa only [key] using (List.pairwise_cons.mp ordered).1)

theorem push_coalesced_length (base : List Entry) (entry : Entry) :
    (pushCoalesced base entry).length ≤ base.length + 1 := by
  induction base using List.reverseRecOn with
  | nil => simp
  | append_singleton base last _ =>
    simp only [push_coalesced_snoc]
    split <;> simp

theorem entries_value_append (left right : List Entry) :
    entriesValue (left ++ right) = BarrierReduction.join (entriesValue left) (entriesValue right) := by
  induction left with
  | nil => funext actor; simp [entriesValue, BarrierReduction.join]
  | cons head tail ih =>
    funext actor
    simp [entriesValue, ih, BarrierReduction.join, Nat.max_assoc]

theorem push_coalesced_value (base : List Entry) (entry : Entry) :
    entriesValue (pushCoalesced base entry) = value base entry := by
  induction base using List.reverseRecOn with
  | nil => funext actor; simp [entriesValue, value]
  | append_singleton base last _ =>
    rw [push_coalesced_snoc]
    split
    · rename_i equal
      funext actor
      by_cases same : entry.1 = actor <;>
        simp [entries_value_append, value, entriesValue, BarrierReduction.join, equal, same, Nat.max_assoc]
    · funext actor
      simp [entries_value_append, value, entriesValue, BarrierReduction.join, Nat.max_assoc]

theorem push_coalesced_keys (base : List Entry) (entry : Entry) (key : Nat)
    (old : ∀ e ∈ base, e.1 ≤ key) (new : entry.1 ≤ key) :
    ∀ e ∈ pushCoalesced base entry, e.1 ≤ key := by
  induction base using List.reverseRecOn with
  | nil => simpa using new
  | append_singleton base last _ =>
    have before : ∀ e ∈ base, e.1 ≤ key := fun e he => old e (List.mem_append_left _ he)
    have final : last.1 ≤ key := old last (by simp)
    rw [push_coalesced_snoc]
    split
    · intro e he
      simp only [List.mem_append, List.mem_singleton] at he
      rcases he with he | rfl
      · exact before e he
      · exact final
    · intro e he
      simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at he
      rcases he with he | rfl | rfl
      · exact before e he
      · exact final
      · exact new

theorem push_coalesced_sorted (base : List Entry) (entry : Entry)
    (ordered : Sorted base) (bound : ∀ e ∈ base, e.1 ≤ entry.1) :
    Sorted (pushCoalesced base entry) := by
  induction base using List.reverseRecOn with
  | nil => simp [Sorted]
  | append_singleton base last _ =>
    have parts := List.pairwise_append.mp ordered
    have previous : Sorted base := parts.1
    have beforeLast : ∀ e ∈ base, e.1 < last.1 :=
      fun e he => parts.2.2 e he last (by simp)
    have final : last.1 ≤ entry.1 := bound last (by simp)
    rw [push_coalesced_snoc]
    split
    · exact List.pairwise_append.mpr ⟨previous, by simp, by
        intro e he b hb
        simp only [List.mem_singleton] at hb
        subst b
        exact beforeLast e he⟩
    · rename_i different
      have less : last.1 < entry.1 := by omega
      apply List.pairwise_append.mpr
      refine ⟨previous, by simp [less], ?_⟩
      intro e he b hb
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hb
      rcases hb with rfl | rfl
      · exact beforeLast e he
      · exact Nat.lt_trans (beforeLast e he) less

theorem coalesce_into_value (base entries : List Entry) :
    entriesValue (coalesceInto base entries) =
      BarrierReduction.join (entriesValue base) (entriesValue entries) := by
  induction entries generalizing base with
  | nil => funext actor; simp [coalesceInto, entriesValue, BarrierReduction.join]
  | cons entry rest ih =>
    rw [coalesceInto, ih, push_coalesced_value]
    funext actor
    simp [value, entriesValue, BarrierReduction.join, Nat.max_assoc]

theorem coalesce_into_sorted (base entries : List Entry)
    (old : Sorted base) (new : WeakSorted entries)
    (boundary : ∀ a ∈ base, ∀ b ∈ entries, a.1 ≤ b.1) :
    Sorted (coalesceInto base entries) := by
  induction entries generalizing base with
  | nil => exact old
  | cons entry rest ih =>
    obtain ⟨headBound, restSorted⟩ := (List.pairwise_cons.mp new)
    apply ih _ (push_coalesced_sorted base entry old (by
      intro e he; exact boundary e he entry (by simp))) restSorted
    intro e he b hb
    exact push_coalesced_keys base entry b.1
      (fun a ha => boundary a ha b (by simp [hb])) (headBound b hb) e he

theorem merge_keys_at_least (left right : List Entry) (bound : Nat)
    (hl : ∀ e ∈ left, bound ≤ e.1) (hr : ∀ e ∈ right, bound ≤ e.1) :
    ∀ e ∈ mergeEntries left right, bound ≤ e.1 := by
  fun_induction mergeEntries left right with
  | case1 right => exact hr
  | case2 left _ => exact hl
  | case3 first left second right equal ih =>
    simp only [List.forall_mem_cons] at hl hr ⊢
    exact ⟨hl.1, ih hl.2 hr.2⟩
  | case4 first left second right different less ih =>
    simp only [List.forall_mem_cons] at hl ⊢
    exact ⟨hl.1, ih hl.2 hr⟩
  | case5 first left second right different greater ih =>
    simp only [List.forall_mem_cons] at hr ⊢
    exact ⟨hr.1, ih hl hr.2⟩

theorem merge_entries_weak_sorted (left right : List Entry)
    (hl : WeakSorted left) (hr : WeakSorted right) : WeakSorted (mergeEntries left right) := by
  fun_induction mergeEntries left right with
  | case1 right => exact hr
  | case2 left _ => exact hl
  | case3 first left second right equal ih =>
    simp only [WeakSorted, List.pairwise_cons] at hl hr ⊢
    exact ⟨merge_keys_at_least left right first.1 hl.1 (by simpa only [equal] using hr.1),
      ih hl.2 hr.2⟩
  | case4 first left second right different less ih =>
    have hrFull := hr
    simp only [WeakSorted, List.pairwise_cons] at hl hr ⊢
    refine ⟨merge_keys_at_least left (second :: right) first.1 hl.1 ?_, ih hl.2 hrFull⟩
    simp only [List.forall_mem_cons]
    exact ⟨Nat.le_of_lt less, fun e he => Nat.le_trans (Nat.le_of_lt less) (hr.1 e he)⟩
  | case5 first left second right different greater ih =>
    have hlFull := hl
    simp only [WeakSorted, List.pairwise_cons] at hl hr ⊢
    have less : second.1 ≤ first.1 := by omega
    refine ⟨merge_keys_at_least (first :: left) right second.1 ?_ hr.1, ih hlFull hr.2⟩
    simp only [List.forall_mem_cons]
    exact ⟨less, fun e he => Nat.le_trans less (hl.1 e he)⟩

def mergeUpdates (base updates : List Entry) : List Entry :=
  coalesceInto [] (mergeEntries base updates)

theorem merge_updates_sorted (base updates : List Entry)
    (old : Sorted base) (new : WeakSorted updates) : Sorted (mergeUpdates base updates) := by
  apply coalesce_into_sorted [] _ (by simp [Sorted])
    (merge_entries_weak_sorted base updates (old.imp (by intros; omega)) new)
  simp

theorem merge_updates_value (base updates : List Entry) :
    entriesValue (mergeUpdates base updates) =
      BarrierReduction.join (entriesValue base) (entriesValue updates) := by
  rw [mergeUpdates, coalesce_into_value, merge_entries_value]
  funext actor
  simp [entriesValue, BarrierReduction.join]

end MemoryWrite.SparseClock
