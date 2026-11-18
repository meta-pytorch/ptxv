/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierReduction

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.SparseClock

abbrev Entry := Nat × Nat

def Sorted (entries : List Entry) : Prop :=
  entries.Pairwise (fun a b => a.1 < b.1)

def entriesValue : List Entry → Nat → Nat
  | [], _ => 0
  | entry :: rest, actor => max (if entry.1 = actor then entry.2 else 0) (entriesValue rest actor)

def value (base : List Entry) (own : Entry) (actor : Nat) : Nat :=
  max (entriesValue base actor) (if own.1 = actor then own.2 else 0)

def mergeEntries : List Entry → List Entry → List Entry
  | [], right => right
  | left, [] => left
  | first :: left, second :: right =>
    if first.1 = second.1 then
      (first.1, max first.2 second.2) :: mergeEntries left right
    else if first.1 < second.1 then
      first :: mergeEntries left (second :: right)
    else second :: mergeEntries (first :: left) right
termination_by left right => left.length + right.length

@[simp] theorem merge_nil_right (left : List Entry) : mergeEntries left [] = left := by
  cases left <;> simp [mergeEntries]

theorem merge_entries_value (left right : List Entry) :
    entriesValue (mergeEntries left right) =
      BarrierReduction.join (entriesValue left) (entriesValue right) := by
  funext actor
  fun_induction mergeEntries left right <;>
    simp_all only [entriesValue, BarrierReduction.join, Nat.max_zero, Nat.zero_max]
  all_goals split_ifs <;>
    simp_all [Nat.max_assoc, Nat.max_left_comm]

theorem merge_keys_lower (left right : List Entry) (bound : Nat)
    (hl : ∀ e ∈ left, bound < e.1) (hr : ∀ e ∈ right, bound < e.1) :
    ∀ e ∈ mergeEntries left right, bound < e.1 := by
  fun_induction mergeEntries left right with
  | case1 right => exact hr
  | case2 left h => exact hl
  | case3 first left second right heq ih =>
    simp only [List.forall_mem_cons] at hl hr ⊢
    exact ⟨hl.1, ih hl.2 hr.2⟩
  | case4 first left second right hne hlt ih =>
    simp only [List.forall_mem_cons] at hl ⊢
    exact ⟨hl.1, ih hl.2 hr⟩
  | case5 first left second right hne hlt ih =>
    simp only [List.forall_mem_cons] at hr ⊢
    exact ⟨hr.1, ih hl hr.2⟩

theorem merge_entries_sorted (left right : List Entry)
    (hl : Sorted left) (hr : Sorted right) : Sorted (mergeEntries left right) := by
  fun_induction mergeEntries left right with
  | case1 right => exact hr
  | case2 left h => exact hl
  | case3 first left second right heq ih =>
    simp only [Sorted, List.pairwise_cons] at hl hr ⊢
    refine ⟨?_, ih hl.2 hr.2⟩
    exact merge_keys_lower left right first.1 hl.1 (by simpa only [heq] using hr.1)
  | case4 first left second right hne hlt ih =>
    have hrFull := hr
    simp only [Sorted, List.pairwise_cons] at hl hr ⊢
    refine ⟨?_, ih hl.2 hrFull⟩
    apply merge_keys_lower left (second :: right) first.1 hl.1
    simp only [List.forall_mem_cons]
    exact ⟨hlt, fun e he => Nat.lt_trans hlt (hr.1 e he)⟩
  | case5 first left second right hne hlt ih =>
    have hlFull := hl
    simp only [Sorted, List.pairwise_cons] at hl hr ⊢
    refine ⟨?_, ih hlFull hr.2⟩
    apply merge_keys_lower (first :: left) right second.1 _ hr.1
    simp only [List.forall_mem_cons]
    have hs : second.1 < first.1 := by omega
    exact ⟨hs, fun e he => Nat.lt_trans hs (hl.1 e he)⟩

def overlay (base : List Entry) (own : Entry) : List Entry :=
  if own.2 = 0 then base else mergeEntries base [own]

theorem overlay_value (base : List Entry) (own : Entry) :
    entriesValue (overlay base own) = value base own := by
  funext actor
  unfold overlay
  split
  · rename_i hz
    simp [value, hz]
  · rw [merge_entries_value]
    simp [BarrierReduction.join, entriesValue, value]

theorem overlay_sorted (base : List Entry) (own : Entry) (hb : Sorted base) :
    Sorted (overlay base own) := by
  unfold overlay
  split
  · exact hb
  · exact merge_entries_sorted base [own] hb (by simp [Sorted])

inductive JoinPlan where
  | keep
  | shareRight
  | replace (entries : List Entry)

def applyPlan (left right : List Entry) : JoinPlan → List Entry
  | .keep => left
  | .shareRight => right
  | .replace entries => entries

def entryCovered (base : List Entry) (entry : Entry) : Bool :=
  base.any (fun candidate => decide (candidate.1 = entry.1)) &&
    decide (entry.2 ≤ entriesValue base entry.1)

def baseCovers (left right : List Entry) : Bool :=
  right.all (entryCovered left)

theorem entry_le_entriesValue (base : List Entry) (entry : Entry) (h : entry ∈ base) :
    entry.2 ≤ entriesValue base entry.1 := by
  induction base with
  | nil => simp at h
  | cons head tail ih =>
    rcases List.mem_cons.mp h with rfl | h
    · simp [entriesValue]
    · exact Nat.le_trans (ih h) (Nat.le_max_right _ _)

theorem base_covers_value (left right : List Entry) (h : baseCovers left right = true) :
    ∀ actor, entriesValue right actor ≤ entriesValue left actor := by
  have he : ∀ entry ∈ right, entry.2 ≤ entriesValue left entry.1 := by
    intro entry member
    have he := List.all_eq_true.mp h entry member
    have he' : (left.any (fun candidate => decide (candidate.1 = entry.1))) = true ∧
        entry.2 ≤ entriesValue left entry.1 := by simpa [entryCovered] using he
    exact he'.2
  clear h
  intro actor
  induction right with
  | nil => simp [entriesValue]
  | cons head tail ih =>
    simp only [List.forall_mem_cons] at he
    simp only [entriesValue, Nat.max_le]
    refine ⟨?_, ih he.2⟩
    split
    · rename_i hk; simpa [hk] using he.1
    · omega

def joinPlan (left : List Entry) (leftOwn : Entry) (right : List Entry)
    (rightOwn : Entry) (sharedBase : Bool) : JoinPlan :=
  if right = [] ∧ rightOwn.2 = 0 then .keep
  else if sharedBase ∧ (rightOwn.2 = 0 ∨ rightOwn.2 ≤ value left leftOwn rightOwn.1) then .keep
  else if (rightOwn.2 = 0 ∨ rightOwn.2 ≤ entriesValue left rightOwn.1) ∧
      baseCovers left right then .keep
  else if rightOwn.2 = 0 ∧ baseCovers right left then .shareRight
  else .replace (mergeEntries left (overlay right rightOwn))

theorem join_plan_value (left : List Entry) (leftOwn : Entry) (right : List Entry)
    (rightOwn : Entry) (sharedBase : Bool) (hshared : sharedBase = true → left = right) :
    value (applyPlan left right (joinPlan left leftOwn right rightOwn sharedBase)) leftOwn =
      BarrierReduction.join (value left leftOwn) (value right rightOwn) := by
  funext actor
  unfold joinPlan
  split
  · rename_i h
    obtain ⟨rfl, hz⟩ := h
    simp [applyPlan, value, entriesValue, BarrierReduction.join, hz]
  · split
    · rename_i h
      have heq := hshared h.1
      subst right
      simp only [applyPlan, BarrierReduction.join, value]
      have hcoverage := h.2
      by_cases hactor : rightOwn.1 = actor
      · subst actor
        simp only [value] at hcoverage
        split_ifs at hcoverage ⊢ <;> omega
      · simp only [hactor, ite_false, Nat.max_zero]
        omega
    · split
      · rename_i h
        have hb := base_covers_value left right h.2 actor
        simp only [applyPlan, value, BarrierReduction.join]
        by_cases hactor : rightOwn.1 = actor
        · subst actor
          rcases h.1 with hz | hc <;> simp only [ite_true] <;> omega
        · simp only [hactor, ite_false]; omega
      · split
        · rename_i h
          have hb := base_covers_value right left h.2 actor
          simp only [applyPlan, value, BarrierReduction.join, h.1, ite_self]
          omega
        · simp only [applyPlan, value, merge_entries_value, overlay_value,
            BarrierReduction.join]
          omega

theorem join_plan_sorted (left : List Entry) (leftOwn : Entry) (right : List Entry)
    (rightOwn : Entry) (sharedBase : Bool) (hl : Sorted left) (hr : Sorted right) :
    Sorted (applyPlan left right (joinPlan left leftOwn right rightOwn sharedBase)) := by
  unfold joinPlan
  split
  · exact hl
  · split
    · exact hl
    · split
      · exact hl
      · split
        · exact hr
        · exact merge_entries_sorted left _ hl (overlay_sorted right rightOwn hr)

def Located (base : List Entry) (actor index : Nat) (found : Bool) : Prop :=
  index ≤ base.length ∧
    (∀ k (hk : k < base.length), k < index → base[k].1 < actor) ∧
    if found then
      ∃ hi : index < base.length, base[index].1 = actor ∧
        ∀ k (hk : k < base.length), index < k → actor < base[k].1
    else ∀ k (hk : k < base.length), index ≤ k → actor < base[k].1

theorem sorted_index_lt (base : List Entry) (hs : Sorted base) (i j : Nat)
    (hi : i < base.length) (hj : j < base.length) (hij : i < j) :
    base[i].1 < base[j].1 :=
  List.pairwise_iff_getElem.mp hs i j hi hj hij

theorem located_tail (head : Entry) (tail : List Entry) (actor index : Nat) (found : Bool)
    (h : Located (head :: tail) actor (index + 1) found) : Located tail actor index found := by
  refine ⟨by simpa using h.1, ?_, ?_⟩
  · intro k hk hki
    have hh := h.2.1 (k + 1) (by simpa using hk) (by omega)
    change tail[k].1 < actor at hh
    exact hh
  · cases found with
    | false =>
      intro k hk hki
      have hh := h.2.2 (k + 1) (by simpa using hk) (by omega)
      change actor < tail[k].1 at hh
      exact hh
    | true =>
      obtain ⟨hi, heq, hafter⟩ := h.2.2
      refine ⟨by simpa using hi, ?_, ?_⟩
      · simpa using heq
      · intro k hk hki
        have hh := hafter (k + 1) (by simpa using hk) (by omega)
        change actor < tail[k].1 at hh
        exact hh

def installOwn (base : List Entry) (own : Entry) (index : Nat) (found : Bool) : List Entry :=
  if found then base.modify index (fun e => (e.1, max e.2 own.2))
  else base.insertIdx index own

theorem merge_singleton_located (base : List Entry) (own : Entry) (index : Nat) (found : Bool)
    (h : Located base own.1 index found) :
    mergeEntries base [own] = installOwn base own index found := by
  induction base generalizing index with
  | nil =>
    have hi : index = 0 := by have := h.1; simp at this; omega
    subst index
    cases found with
    | false => simp [mergeEntries, installOwn]
    | true => obtain ⟨hi, _⟩ := h.2.2; simp at hi
  | cons head tail ih =>
    cases index with
    | zero =>
      cases found with
      | false =>
        have hgt := h.2.2 0 (by simp) (by omega)
        simp only [List.getElem_cons_zero] at hgt
        have hne : head.1 ≠ own.1 := by omega
        have hnlt : ¬head.1 < own.1 := by omega
        simp [mergeEntries, hne, hnlt, installOwn]
      | true =>
        obtain ⟨hi, heq, _⟩ := h.2.2
        simp only [List.getElem_cons_zero] at heq
        simp [mergeEntries, heq, installOwn]
    | succ index =>
      have hlt := h.2.1 0 (by simp) (by omega)
      simp only [List.getElem_cons_zero] at hlt
      have hne : head.1 ≠ own.1 := by omega
      rw [mergeEntries, ite_eq_right hne, ite_eq_left hlt]
      rw [ih index (located_tail head tail own.1 index found h)]
      cases found <;> simp [installOwn]

theorem entries_value_zero (base : List Entry) (actor : Nat)
    (h : ∀ e ∈ base, e.1 ≠ actor) : entriesValue base actor = 0 := by
  induction base with
  | nil => rfl
  | cons head tail ih =>
    simp only [List.forall_mem_cons] at h
    simp [entriesValue, h.1, ih h.2]

theorem located_value (base : List Entry) (actor index : Nat) (found : Bool)
    (h : Located base actor index found) :
    entriesValue base actor = if found then (base[index]?.getD (0, 0)).2 else 0 := by
  induction base generalizing index with
  | nil =>
    cases found with
    | false => rfl
    | true => obtain ⟨hi, _⟩ := h.2.2; simp at hi
  | cons head tail ih =>
    cases index with
    | zero =>
      cases found with
      | false =>
        have hn : head.1 ≠ actor := by have := h.2.2 0 (by simp) (by omega); simpa using Nat.ne_of_gt this
        have ht : ∀ e ∈ tail, e.1 ≠ actor := by
          intro e he
          obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem he
          have hh := h.2.2 (k + 1) (by simpa using hk) (by omega)
          change actor < tail[k].1 at hh
          omega
        simp [entriesValue, hn, entries_value_zero tail actor ht]
      | true =>
        obtain ⟨hi, heq, hafter⟩ := h.2.2
        change head.1 = actor at heq
        have ht : ∀ e ∈ tail, e.1 ≠ actor := by
          intro e he
          obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem he
          have hh := hafter (k + 1) (by simpa using hk) (by omega)
          change actor < tail[k].1 at hh
          omega
        simp [entriesValue, heq, entries_value_zero tail actor ht]
    | succ index =>
      have hlt := h.2.1 0 (by simp) (by omega)
      change head.1 < actor at hlt
      have hn : head.1 ≠ actor := by omega
      simp only [entriesValue, hn, ite_false, Nat.zero_max]
      simpa using ih index (located_tail head tail actor index found h)

theorem sorted_index_lower (base : List Entry) (hs : Sorted base) (index : Nat)
    (hi : index < base.length) : index ≤ base[index].1 := by
  induction index with
  | zero => omega
  | succ index ih =>
    have hip : index < base.length := by omega
    have hprev := ih hip
    have hlt := sorted_index_lt base hs index (index + 1) hip hi (by omega)
    omega

theorem merge_entries_length (left right : List Entry) :
    (mergeEntries left right).length ≤ left.length + right.length := by
  fun_induction mergeEntries left right <;> simp_all only [List.length_cons, List.length_nil] <;> omega

theorem overlay_length (base : List Entry) (own : Entry) :
    (overlay base own).length ≤ base.length + 1 := by
  unfold overlay
  split
  · omega
  · simpa using merge_entries_length base [own]


def scanCovers : List Entry → List Entry → Bool
  | _, [] => true
  | [], _ :: _ => false
  | a :: left, b :: right =>
    if a.1 < b.1 then scanCovers left (b :: right)
    else if a.1 = b.1 then decide (b.2 ≤ a.2) && scanCovers left right
    else false
termination_by left right => left.length + right.length

theorem entry_covered_cons_other (head entry : Entry) (tail : List Entry)
    (hne : head.1 ≠ entry.1) :
    entryCovered (head :: tail) entry = entryCovered tail entry := by
  simp [entryCovered, entriesValue, hne]

theorem base_covers_cons_other (head : Entry) (left right : List Entry)
    (hne : ∀ entry ∈ right, head.1 ≠ entry.1) :
    baseCovers (head :: left) right = baseCovers left right := by
  induction right with
  | nil => rfl
  | cons entry rest ih =>
    have hn := List.forall_mem_cons.mp hne
    change (entryCovered (head :: left) entry && baseCovers (head :: left) rest) =
      (entryCovered left entry && baseCovers left rest)
    rw [entry_covered_cons_other head entry left hn.1, ih hn.2]

theorem entry_covered_cons_same (head entry : Entry) (tail : List Entry)
    (hs : Sorted (head :: tail)) (he : head.1 = entry.1) :
    entryCovered (head :: tail) entry = decide (entry.2 ≤ head.2) := by
  have ht : ∀ e ∈ tail, e.1 ≠ entry.1 := by
    intro e hm
    have hp := (List.pairwise_cons.mp hs).1 e hm
    omega
  simp [entryCovered, entriesValue, he, entries_value_zero tail entry.1 ht]

theorem entry_covered_too_small (head entry : Entry) (tail : List Entry)
    (hs : Sorted (head :: tail)) (hlt : entry.1 < head.1) :
    entryCovered (head :: tail) entry = false := by
  have hn : ∀ e ∈ head :: tail, e.1 ≠ entry.1 := by
    intro e hm
    rcases List.mem_cons.mp hm with rfl | hm
    · omega
    · have hp := (List.pairwise_cons.mp hs).1 e hm
      omega
  have ha : (head :: tail).any (fun e => decide (e.1 = entry.1)) = false := by
    apply List.any_eq_false.mpr
    intro e hm
    simp [hn e hm]
  simp [entryCovered, ha]

theorem scan_covers_eq (left right : List Entry) (hl : Sorted left) (hr : Sorted right) :
    scanCovers left right = baseCovers left right := by
  fun_induction scanCovers left right with
  | case1 left => simp [baseCovers]
  | case2 b right => simp [baseCovers, entryCovered]
  | case3 a left b right less ih =>
    have hn : ∀ entry ∈ b :: right, a.1 ≠ entry.1 := by
      intro entry hm
      rcases List.mem_cons.mp hm with rfl | hm
      · omega
      · have hp := (List.pairwise_cons.mp hr).1 entry hm
        omega
    rw [base_covers_cons_other a left (b :: right) hn]
    exact ih (List.pairwise_cons.mp hl).2 hr
  | case4 a left b right notless equal ih =>
    have hn : ∀ entry ∈ right, a.1 ≠ entry.1 := by
      intro entry hm
      have hp := (List.pairwise_cons.mp hr).1 entry hm
      omega
    change (decide (b.2 ≤ a.2) && scanCovers left right) =
      (entryCovered (a :: left) b && baseCovers (a :: left) right)
    rw [entry_covered_cons_same a b left hl equal,
      base_covers_cons_other a left right hn,
      ih (List.pairwise_cons.mp hl).2 (List.pairwise_cons.mp hr).2]
  | case5 a left b right notless notequal =>
    have hlt : b.1 < a.1 := by omega
    change false = (entryCovered (a :: left) b && baseCovers (a :: left) right)
    rw [entry_covered_too_small a b left hl hlt]
    rfl

end MemoryWrite.SparseClock
