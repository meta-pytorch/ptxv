/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteMainLoop

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

def emptySets : cache_graph.WorkSets :=
  ⟨alloc.vec.Vec.new cache_graph.PageIndex, alloc.vec.Vec.new _⟩

theorem empty_directory : DirectoryInvariant emptySets := by
  constructor
  · intro left right less inside
    simp only [emptySets, alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.length_nil] at inside
    omega
  · intro entry member
    simp only [emptySets, alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.not_mem_nil] at member
  · intro slots member
    simp only [emptySets, alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.not_mem_nil] at member
  · intro left member
    simp only [emptySets, alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.not_mem_nil] at member

theorem empty_resources : DirectoryResources emptySets := by
  constructor
  · rfl
  · intro entry member
    simp only [emptySets, alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.not_mem_nil] at member

theorem empty_words_absent (pc : Nat) (words : alloc.vec.Vec U64) : ¬WordsAt emptySets pc words := by
  rintro ⟨storage, ⟨entry, member, _⟩, _⟩
  simp only [emptySets, alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.not_mem_nil] at member

theorem empty_sized (count : Nat) : RowsSized emptySets count := by
  intro pc words atWords
  exact False.elim (empty_words_absent pc words atWords)

structure SetupPost (graph : Graph) (count : Nat) (start words : Usize)
    (sets : cache_graph.WorkSets) (work : alloc.vec.Vec Usize) (pending : Usize) : Prop where
  storage : StorageInvariant sets work pending count words.val
  settled : WorkInvariant graph (workView sets work pending)
  startKnown : (workSolution sets).present start.val


theorem successful_setup (graph : Graph) (count : Nat) (start words : Usize)
    (startBound : start.val < count) (answer : Usize × Usize × Bool)
    (sets : cache_graph.WorkSets) (pending : Usize) (work : alloc.vec.Vec Usize)
    (initializedOk : cache_graph.ensure_pc emptySets start words = .ok (answer, sets))
    (pushedOk : cache_graph.push_work (alloc.vec.Vec.new Usize) 0#usize start = .ok (pending, work)) :
    SetupPost graph count start words sets work pending := by
  have initialization := rust_ensure_pc_resourced emptySets start words empty_directory empty_resources
  simp only [initializedOk, WP.spec_ok, WP.uncurry'_pair] at initialization
  obtain ⟨initialized, resources⟩ := initialization
  have sized := initialized.sized (empty_sized words.val)
  have pushed := successful_push_work (alloc.vec.Vec.new Usize) 0#usize start pending work
    (by simp only [alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.length_nil]; scalar_tac) pushedOk
  have queueEq : queueView work pending = [start.val] := by
    simpa only [queueView, alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.take_nil, List.reverse_nil,
      List.map_nil] using pushed.2.1
  have present : ∀ pc, (workSolution sets).present pc ↔ pc = start.val := by
    intro pc
    rw [initialized.present empty_directory]
    have absent : ¬(workSolution emptySets).present pc := by
      rintro ⟨old, atWords⟩
      exact empty_words_absent pc old atWords
    simp only [absent, false_or]
  have startKnown := (present start.val).mpr rfl
  refine ⟨?_, ?_, startKnown⟩
  · refine ⟨initialized.invariant, resources, sized, pushed.2.2, ?_, ?_⟩
    · intro pc known
      simpa only [(present pc).mp known] using startBound
    · intro pc member
      have same : pc = start.val := by simpa only [queueEq, List.mem_singleton] using member
      exact (present pc).mpr same
  · intro pc absent known
    have same := (present pc).mp known
    exact False.elim (absent (by simpa only [workView, queueEq, List.mem_singleton] using same))

end MemoryWrite.MayWrite
