/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWritePaths

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

@[step]
theorem rust_valid_indices (indices : Slice Usize) (bound : Usize) :
    cache_graph.valid_indices indices bound ⦃ valid => valid = true →
      ∀ position < indices.val.length, indices.val[position]!.val < bound.val ⦄ := by
  unfold cache_graph.valid_indices cache_graph.valid_indices_loop
  apply loop.spec_decr_nat
    (fun state : Usize × Bool => indices.val.length - state.1.val)
    (fun state : Usize × Bool => state.1.val ≤ indices.val.length ∧
      (state.2 = true → ∀ position < state.1.val, indices.val[position]!.val < bound.val))
  · rintro ⟨index, valid⟩ ⟨indexBound, prefixBound⟩
    dsimp only at indexBound prefixBound
    unfold cache_graph.valid_indices_loop.body
    by_cases inside : index.val < indices.val.length
    · have test : index < Slice.len indices := by scalar_tac
      simp only [test, ite_true]
      cases valid with
      | false => simp
      | true =>
        simp only [ite_true]
        step with Slice.index_usize_spec indices index inside as ⟨entry, entryEq⟩
        have entryAt : entry = indices.val[index.val]! := by
          rw [entryEq, List.Inhabited_getElem_eq_getElem! indices.val index.val inside]
        step with Usize.add_spec (x := index) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
        refine ⟨?_, ?_, by scalar_tac⟩
        · scalar_tac
        · intro valid position positionBound
          by_cases atIndex : position = index.val
          · subst position
            have entryBound : entry < bound := of_decide_eq_true valid
            scalar_tac
          · exact prefixBound rfl position (by omega)
    · have test : ¬index < Slice.len indices := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      intro valid position positionBound
      exact prefixBound valid position (by omega)
  · exact ⟨by scalar_tac, by intro _ position positionBound; scalar_tac⟩

def GraphBounded (graph : cache_graph.MayWriteGraph) : Prop :=
  graph.succ.val.length = graph.reach.val.length ∧
  graph.writes.val.length = graph.reach.val.length ∧
  (∀ pc < graph.reach.val.length, ∀ successor ∈ graph.succ.val[pc]!.val,
    successor.val < graph.reach.val.length) ∧
  ∀ pc < graph.reach.val.length, ∀ register ∈ graph.writes.val[pc]!.val,
    register.val < graph.register_count.val

theorem indices_bound_members (indices : List Usize) (bound : Nat)
    (bounded : ∀ position < indices.length, indices[position]!.val < bound) :
    ∀ entry ∈ indices, entry.val < bound := by
  intro entry member
  obtain ⟨position, inside, atPosition⟩ := List.getElem_of_mem member
  have := bounded position inside
  simpa only [←List.Inhabited_getElem_eq_getElem! _ _ inside, atPosition] using this

def RowsBounded (successors writes : alloc.vec.Vec (alloc.vec.Vec Usize))
    (count registers limit : Nat) : Prop :=
  ∀ pc < limit,
    (∀ successor ∈ successors.val[pc]!.val, successor.val < count) ∧
    ∀ register ∈ writes.val[pc]!.val, register.val < registers

@[step]
theorem rust_valid_graph_loop (successors writes : alloc.vec.Vec (alloc.vec.Vec Usize))
    (count registers : Usize)
    (successorLength : successors.val.length = count.val)
    (writeLength : writes.val.length = count.val) :
    cache_graph.valid_graph_loop successors writes registers count 0#usize true ⦃ valid =>
      valid = true → RowsBounded successors writes count.val registers.val count.val ⦄ := by
  unfold cache_graph.valid_graph_loop
  apply loop.spec_decr_nat (fun state : Usize × Bool => count.val - state.1.val)
    (fun state : Usize × Bool => state.1.val ≤ count.val ∧
      (state.2 = true → RowsBounded successors writes count.val registers.val state.1.val))
  · rintro ⟨pc, valid⟩ ⟨pcBound, prefixBound⟩
    dsimp only at pcBound prefixBound
    unfold cache_graph.valid_graph_loop.body
    by_cases inside : pc.val < count.val
    · have test : pc < count := by scalar_tac
      simp only [test, ite_true]
      cases valid with
      | false => simp
      | true =>
          simp only [ite_true]
          have successorPcBound : pc.val < successors.val.length := by rw [successorLength]; exact inside
          have writePcBound : pc.val < writes.val.length := by rw [writeLength]; exact inside
          step with alloc.vec.Vec.index_usize_spec successors pc successorPcBound as ⟨row, rowEq⟩
          have rowAt : row = successors.val[pc.val]! := by
            rw [rowEq, List.Inhabited_getElem_eq_getElem! _ _ successorPcBound]
          step with rust_valid_indices (alloc.vec.Vec.deref row) count as ⟨successorValid, successorBounds⟩
          simp only [alloc.vec.Vec.deref, Slice.from_val] at successorBounds
          cases successorValid with
          | false =>
              simp only [Bool.false_eq_true, ite_false]
              step with Usize.add_spec (x := pc) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
              refine ⟨by scalar_tac, ?_, by scalar_tac⟩
              intro impossible
              cases impossible
          | true =>
              simp only [ite_true]
              step with alloc.vec.Vec.index_usize_spec writes pc writePcBound as ⟨destinations, destinationsEq⟩
              have destinationsAt : destinations = writes.val[pc.val]! := by
                rw [destinationsEq, List.Inhabited_getElem_eq_getElem! _ _ writePcBound]
              step with rust_valid_indices (alloc.vec.Vec.deref destinations) registers as ⟨writeValid, writeBounds⟩
              simp only [alloc.vec.Vec.deref, Slice.from_val] at writeBounds
              step with Usize.add_spec (x := pc) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
              refine ⟨by scalar_tac, ?_, by scalar_tac⟩
              intro valid earlier earlierBound
              by_cases atCurrent : earlier = pc.val
              · subst earlier
                exact ⟨by simpa only [rowAt] using indices_bound_members row.val count.val (successorBounds rfl),
                  by simpa only [destinationsAt] using indices_bound_members destinations.val registers.val (writeBounds valid)⟩
              · exact prefixBound rfl earlier (by omega)
    · have test : ¬pc < count := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      intro valid
      simpa only [show pc.val = count.val by omega] using prefixBound valid
  · exact ⟨by scalar_tac, by intro _ earlier earlierBound; scalar_tac⟩

@[step]
theorem rust_valid_graph (graph : cache_graph.MayWriteGraph) :
    cache_graph.valid_graph graph ⦃ valid => valid = true → GraphBounded graph ⦄ := by
  unfold cache_graph.valid_graph
  dsimp only
  split
  · simp
  · rename_i successorsEqual
    have successorLength : graph.succ.val.length = graph.reach.val.length := by scalar_tac
    split
    · simp
    · rename_i writesEqual
      have writeLength : graph.writes.val.length = graph.reach.val.length := by scalar_tac
      step with rust_valid_graph_loop graph.succ graph.writes (alloc.vec.Vec.len graph.reach)
        graph.register_count (by simpa only [alloc.vec.Vec.len_val] using successorLength)
        (by simpa only [alloc.vec.Vec.len_val] using writeLength) as ⟨valid, bounded⟩
      rename_i validTrue
      have rows := bounded validTrue
      exact ⟨successorLength, writeLength, fun pc inside => (rows pc inside).1,
        fun pc inside => (rows pc inside).2⟩

end MemoryWrite.MayWrite
