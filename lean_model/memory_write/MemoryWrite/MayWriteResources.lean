/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteInitialized

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

structure DirectoryResources (sets : cache_graph.WorkSets) : Prop where
  count : sets.pages.val.length = sets.directory.val.length
  bounded : ∀ entry ∈ sets.directory.val, entry.page.val ≤ Usize.max / 64

theorem StrictKeys.index_le {length : Nat} {key : Nat → Nat}
    (strict : StrictKeys length key) (index : Nat) (inside : index < length) : index ≤ key index := by
  induction index with
  | zero => exact Nat.zero_le _
  | succ index ih =>
      have previous := ih (by omega)
      have increasing := strict index (index + 1) (by omega) inside
      omega

theorem directory_resources_space {sets : cache_graph.WorkSets}
    (invariant : DirectoryInvariant sets) (resources : DirectoryResources sets) :
    sets.directory.val.length < Usize.max ∧ sets.pages.val.length < Usize.max := by
  have maxLarge : 64 < Usize.max := by scalar_tac
  have bound : sets.directory.val.length ≤ Usize.max / 64 + 1 := by
    by_cases nonempty : 0 < sets.directory.val.length
    · have lastBound : sets.directory.val.length - 1 < sets.directory.val.length := by omega
      have indexBound := invariant.strict.index_le (sets.directory.val.length - 1) lastBound
      have member : sets.directory.val[sets.directory.val.length - 1]! ∈ sets.directory.val := by
        rw [←List.Inhabited_getElem_eq_getElem! _ _ lastBound]
        exact List.getElem_mem lastBound
      have upper := resources.bounded _ member
      omega
    · omega
  have division := Nat.div_mul_le_self Usize.max 64
  rw [resources.count]
  omega

theorem page_addition_preserves_resources {before after : cache_graph.WorkSets}
    (resources : DirectoryResources before) {page storage : Usize}
    (pageBound : page.val ≤ Usize.max / 64)
    (addition : PageAddition before after page storage) : DirectoryResources after := by
  obtain ⟨_, position, bound, directoryEq, slots, _, pagesEq⟩ := addition
  constructor
  · simp only [pagesEq, directoryEq, List.length_append, List.length_singleton,
      List.length_insertIdx_of_le_length bound.1, resources.count]
  · intro entry member
    rw [directoryEq, List.mem_insertIdx bound.1] at member
    rcases member with equal | old
    · subst entry
      exact pageBound
    · exact resources.bounded entry old

theorem ensured_page_preserves_resources {before after : cache_graph.WorkSets}
    (resources : DirectoryResources before) {page storage : Usize}
    (pageBound : page.val ≤ Usize.max / 64)
    (ensured : PageEnsured before after page storage) : DirectoryResources after := by
  rcases ensured with ⟨equal, _⟩ | ⟨addition, _⟩
  · simpa only [equal] using resources
  · exact page_addition_preserves_resources resources pageBound addition

theorem set_slot_preserves_resources {sets : cache_graph.WorkSets}
    (resources : DirectoryResources sets) (storage offset : Usize)
    (value : Option (alloc.vec.Vec U64)) : DirectoryResources (setSlot sets storage offset value) := by
  constructor
  · simpa only [setSlot, alloc.vec.Vec.set_val_eq, List.length_set] using resources.count
  · exact resources.bounded

theorem pc_ensured_preserves_resources {before after : cache_graph.WorkSets}
    (resources : DirectoryResources before) {pc count : Usize}
    {answer : Usize × Usize × Bool} (ensured : PcEnsured before after pc count answer) :
    DirectoryResources after := by
  obtain ⟨middle, _, _, ⟨page, pageValue, pageEnsured⟩, _, _, previous, _, _, next⟩ := ensured
  have pageBound : page.val ≤ Usize.max / 64 := by
    rw [pageValue]
    exact Nat.div_le_div_right (show pc.val ≤ Usize.max by scalar_tac)
  have middleResources := ensured_page_preserves_resources resources pageBound pageEnsured
  cases previous with
  | none =>
      obtain ⟨words, _, afterEq⟩ := next
      rw [afterEq]
      exact set_slot_preserves_resources middleResources _ _ _
  | some _ => simpa only [next] using middleResources

@[step]
theorem rust_ensure_pc_resourced (sets : cache_graph.WorkSets) (pc count : Usize)
    (invariant : DirectoryInvariant sets) (resources : DirectoryResources sets) :
    cache_graph.ensure_pc sets pc count ⦃ answer result =>
      InitializedPc sets result pc count answer ∧ DirectoryResources result ⦄ := by
  have space := directory_resources_space invariant resources
  apply WP.spec_mono (rust_ensure_pc sets pc count invariant space.1 space.2)
  intro result ensured
  exact ⟨pc_ensured_initializes ensured, pc_ensured_preserves_resources resources ensured⟩

end MemoryWrite.MayWrite
