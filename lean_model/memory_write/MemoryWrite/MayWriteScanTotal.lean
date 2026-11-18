/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteSuccessorTotal
import MemoryWrite.MayWriteScan

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

theorem rust_successor_scan_returns (successors : alloc.vec.Vec (alloc.vec.Vec Usize))
    (sets : cache_graph.WorkSets) (words : Usize) (work : alloc.vec.Vec Usize) (pending pc : Usize)
    (out : alloc.vec.Vec U64) (count : Nat)
    (initial : StorageInvariant sets work pending count words.val)
    (pcBound : pc.val < successors.val.length)
    (successorBounds : ∀ successor ∈ successors.val[pc.val]!.val, successor.val < count)
    (outLength : out.val.length = words.val)
    (space : rank count (words.val * 64) (workSolution sets) pending.val < Usize.max) :
    cache_graph.may_write_sets_loop0_loop1 successors sets words work pending pc out 0#usize
      ⦃ result => StorageInvariant result.1 result.2.1 result.2.2 count words.val ∧
        rank count (words.val * 64) (workSolution result.1) result.2.2.val ≤
          rank count (words.val * 64) (workSolution sets) pending.val ⦄ := by
  unfold cache_graph.may_write_sets_loop0_loop1
  apply loop.spec_decr_nat
    (fun state : cache_graph.WorkSets × alloc.vec.Vec Usize × Usize × Usize =>
      successors.val[pc.val]!.val.length - state.2.2.2.val)
    (fun state => StorageInvariant state.1 state.2.1 state.2.2.1 count words.val ∧
      state.2.2.2.val ≤ successors.val[pc.val]!.val.length ∧
      rank count (words.val * 64) (workSolution state.1) state.2.2.1.val ≤
        rank count (words.val * 64) (workSolution sets) pending.val)
  · rintro ⟨currentSets, currentWork, currentPending, index⟩ ⟨storage, indexBound, rankBound⟩
    have indexLe : index.val ≤ successors.val[pc.val]!.val.length := indexBound
    have currentRank : rank count (words.val * 64) (workSolution currentSets) currentPending.val ≤
        rank count (words.val * 64) (workSolution sets) pending.val := rankBound
    unfold cache_graph.may_write_sets_loop0_loop1.body
    step with alloc.vec.Vec.index_usize_spec successors pc pcBound as ⟨row, rowEqual⟩
    have rowAt : row = successors.val[pc.val]! := by
      rw [rowEqual, List.Inhabited_getElem_eq_getElem! _ _ pcBound]
    by_cases inside : index.val < row.val.length
    · have test : index < alloc.vec.Vec.len row := by scalar_tac
      simp only [test, ite_true]
      step with alloc.vec.Vec.index_usize_spec row index inside as ⟨successor, successorEqual⟩
      have successorMember : successor ∈ successors.val[pc.val]!.val := by
        have member : successor ∈ row.val := by rw [successorEqual]; exact List.getElem_mem inside
        simpa only [rowAt] using member
      have pendingSpace : currentPending.val < Usize.max :=
        lt_of_le_of_lt ((pending_le_rank count (words.val * 64) (workSolution currentSets)
          currentPending.val).trans currentRank) space
      step with rust_propagate_successor_total currentSets currentWork currentPending successor words
        (alloc.vec.Vec.deref out) count (successorBounds successor successorMember) storage.directory storage.resources
        storage.sized (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using outLength)
        storage.pendingBound pendingSpace as ⟨nextPending, nextSets, nextWork, post, nextRank⟩
      simp only [alloc.vec.Vec.deref, Slice.from_val] at post
      step with Usize.add_spec (x := index) (y := 1#usize) (by scalar_tac) as ⟨nextIndex, nextIndexValue⟩
      have nextStorage := post.storage storage (successorBounds successor successorMember)
      have rowInside : index.val < successors.val[pc.val]!.val.length := by simpa only [rowAt] using inside
      refine ⟨nextStorage, ?_, nextRank.trans currentRank, ?_⟩ <;> scalar_tac
    · have test : ¬index < alloc.vec.Vec.len row := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      exact ⟨storage, currentRank⟩
  · exact ⟨initial, by scalar_tac, Nat.le_refl _⟩

@[step]
theorem rust_successor_scan_total (successors : alloc.vec.Vec (alloc.vec.Vec Usize))
    (sets : cache_graph.WorkSets) (words : Usize) (work : alloc.vec.Vec Usize) (pending pc : Usize)
    (out : alloc.vec.Vec U64) (count : Nat)
    (initial : StorageInvariant sets work pending count words.val)
    (pcBound : pc.val < successors.val.length)
    (successorBounds : ∀ successor ∈ successors.val[pc.val]!.val, successor.val < count)
    (outLength : out.val.length = words.val)
    (space : rank count (words.val * 64) (workSolution sets) pending.val < Usize.max) :
    cache_graph.may_write_sets_loop0_loop1 successors sets words work pending pc out 0#usize
      ⦃ result => ScanPost (workView sets work pending) successors.val[pc.val]!.val out.val count words.val result ∧
        rank count (words.val * 64) (workSolution result.1) result.2.2.val ≤
          rank count (words.val * 64) (workSolution sets) pending.val ⦄ := by
  obtain ⟨result, executed, resultSpec⟩ := MemoryWrite.spec_returns _ _
    (rust_successor_scan_returns successors sets words work pending pc out count initial pcBound
      successorBounds outLength space)
  simp only [executed, WP.spec_ok]
  exact ⟨successful_successor_scan successors sets words work pending pc out count initial pcBound
    successorBounds outLength result executed, resultSpec.2⟩

end MemoryWrite.MayWrite
