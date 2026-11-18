/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWritePropagation

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

structure StorageInvariant (sets : cache_graph.WorkSets) (work : alloc.vec.Vec Usize)
    (pending : Usize) (pcCount wordCount : Nat) : Prop where
  directory : DirectoryInvariant sets
  resources : DirectoryResources sets
  sized : RowsSized sets wordCount
  pendingBound : pending.val ≤ work.val.length
  knownBound : ∀ pc, (workSolution sets).present pc → pc < pcCount
  queuedKnown : ∀ pc ∈ queueView work pending, (workSolution sets).present pc

theorem SuccessorPost.storage {sets : cache_graph.WorkSets} {work : alloc.vec.Vec Usize}
    {pending successor words : Usize} {outgoing : List U64}
    {result : Usize × cache_graph.WorkSets × alloc.vec.Vec Usize} {count : Nat}
    (post : SuccessorPost sets work pending successor words outgoing result)
    (before : StorageInvariant sets work pending count words.val) (successorBound : successor.val < count) :
    StorageInvariant result.2.1 result.2.2 result.1 count words.val := by
  refine ⟨post.invariant, post.resources, post.sized, post.pendingBound, ?_, ?_⟩
  · intro address known
    rcases (post.present address).mp known with old | same
    · exact before.knownBound address old
    · simpa only [same] using successorBound
  · intro address member
    rcases post.queue with oldQueue | newQueue
    · rw [oldQueue] at member
      exact post.effect.growth.1 address (before.queuedKnown address member)
    · rw [newQueue] at member
      rcases List.mem_cons.mp member with same | old
      · simpa only [same, workView] using post.effect.targetPresent
      · exact post.effect.growth.1 address (before.queuedKnown address old)

structure ScanInvariant (before : WorkState) (successors : List Usize) (outgoing : List U64)
    (count wordCount : Nat)
    (state : cache_graph.WorkSets × alloc.vec.Vec Usize × Usize × Usize) : Prop where
  storage : StorageInvariant state.1 state.2.1 state.2.2.1 count wordCount
  indexBound : state.2.2.2.val ≤ successors.length
  growth : Grows before.solution (workSolution state.1)
  stable : StableWhenUnqueued before (workView state.1 state.2.1 state.2.2.1)
  processed : ∀ position < state.2.2.2.val,
    (workSolution state.1).present successors[position]!.val ∧
    ∀ register, WordContains outgoing register →
      (workSolution state.1).incoming successors[position]!.val register

structure ScanPost (before : WorkState) (successors : List Usize) (outgoing : List U64)
    (count wordCount : Nat)
    (result : cache_graph.WorkSets × alloc.vec.Vec Usize × Usize) : Prop where
  storage : StorageInvariant result.1 result.2.1 result.2.2 count wordCount
  growth : Grows before.solution (workSolution result.1)
  stable : StableWhenUnqueued before (workView result.1 result.2.1 result.2.2)
  propagated : ∀ successor ∈ successors,
    (workSolution result.1).present successor.val ∧
    ∀ register, WordContains outgoing register →
      (workSolution result.1).incoming successor.val register

theorem successful_successor_scan (successors : alloc.vec.Vec (alloc.vec.Vec Usize))
    (sets : cache_graph.WorkSets) (words : Usize) (work : alloc.vec.Vec Usize) (pending pc : Usize)
    (out : alloc.vec.Vec U64) (count : Nat)
    (initial : StorageInvariant sets work pending count words.val)
    (pcBound : pc.val < successors.val.length)
    (successorBounds : ∀ successor ∈ successors.val[pc.val]!.val, successor.val < count)
    (outLength : out.val.length = words.val)
    (result : cache_graph.WorkSets × alloc.vec.Vec Usize × Usize)
    (success : cache_graph.may_write_sets_loop0_loop1 successors sets words work pending pc out 0#usize = .ok result) :
    ScanPost (workView sets work pending) successors.val[pc.val]!.val out.val count words.val result := by
  unfold cache_graph.may_write_sets_loop0_loop1 at success
  have execution := MemoryWrite.successful_loop_has_execution _ _ _ success
  apply execution.preserves
    (ScanInvariant (workView sets work pending) successors.val[pc.val]!.val out.val count words.val)
    (ScanPost (workView sets work pending) successors.val[pc.val]!.val out.val count words.val)
  · exact ⟨initial, by scalar_tac, ⟨fun _ known => known, fun _ _ bit => bit⟩,
      fun _ absent => ⟨absent, Iff.rfl, fun _ => Iff.rfl⟩,
      by intro position inside; scalar_tac⟩
  · rintro ⟨currentSets, currentWork, currentPending, index⟩ invariant flow bodyOk
    have indexBound : index.val ≤ successors.val[pc.val]!.val.length := invariant.indexBound
    have processed : ∀ position < index.val,
        (workSolution currentSets).present successors.val[pc.val]!.val[position]!.val ∧
        ∀ register, WordContains out.val register →
          (workSolution currentSets).incoming successors.val[pc.val]!.val[position]!.val register :=
      invariant.processed
    unfold cache_graph.may_write_sets_loop0_loop1.body at bodyOk
    obtain ⟨row, rowOk, tailOk⟩ := MemoryWrite.bind_success _ _ _ bodyOk
    have rowEq := rust_vec_index successors pc pcBound
    simp only [rowOk, WP.spec_ok] at rowEq
    have rowAt : row = successors.val[pc.val]! := by
      rw [rowEq, List.Inhabited_getElem_eq_getElem! _ _ pcBound]
    by_cases inside : index.val < row.val.length
    · have test : index < alloc.vec.Vec.len row := by scalar_tac
      simp only [test, ite_true] at tailOk
      obtain ⟨successor, successorOk, propagatedOk⟩ := MemoryWrite.bind_success _ _ _ tailOk
      have successorEq := rust_vec_index row index inside
      simp only [successorOk, WP.spec_ok] at successorEq
      have successorAt : successor = row.val[index.val]! := by
        rw [successorEq, List.Inhabited_getElem_eq_getElem! _ _ inside]
      have successorMember : successor ∈ successors.val[pc.val]!.val := by
        have member : successor ∈ row.val := by rw [successorEq]; exact List.getElem_mem inside
        simpa only [rowAt] using member
      obtain ⟨⟨nextPending, nextSets, nextWork⟩, propagated, incrementOk⟩ :=
        MemoryWrite.bind_success _ _ _ propagatedOk
      have propagatedPost := successful_propagate_successor currentSets currentWork currentPending
        successor words (alloc.vec.Vec.deref out) invariant.storage.directory invariant.storage.resources
        invariant.storage.sized (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using outLength)
        invariant.storage.pendingBound (nextPending, nextSets, nextWork) propagated
      simp only [alloc.vec.Vec.deref, Slice.from_val] at propagatedPost
      obtain ⟨nextIndex, increment, completed⟩ := MemoryWrite.bind_success _ _ _ incrementOk
      have nextIndexSpec := Usize.add_spec (x := index) (y := 1#usize) (by scalar_tac)
      simp only [increment, WP.spec_ok] at nextIndexSpec
      have nextIndexValue : nextIndex.val = index.val + 1 := nextIndexSpec
      have flowEq : ControlFlow.cont (nextSets, nextWork, nextPending, nextIndex) = flow :=
        Result.ok_injective completed
      rw [←flowEq]
      have nextStorage := propagatedPost.storage invariant.storage (successorBounds successor successorMember)
      refine ⟨nextStorage, ?_, invariant.growth.trans propagatedPost.effect.growth,
        invariant.stable.trans propagatedPost.effect.stable, ?_⟩
      · have rowInside : index.val < successors.val[pc.val]!.val.length := by simpa only [rowAt] using inside
        change nextIndex.val ≤ successors.val[pc.val]!.val.length
        omega
      · intro position positionBound
        change position < nextIndex.val at positionBound
        by_cases atCurrent : position = index.val
        · subst position
          have targetEq : successor.val = successors.val[pc.val]!.val[index.val]!.val := by
            simpa only [rowAt] using congrArg UScalar.val successorAt
          exact ⟨by simpa only [targetEq, workView] using propagatedPost.effect.targetPresent,
            fun register bit => by simpa only [targetEq, workView] using propagatedPost.effect.included register bit⟩
        · have previous := processed position (by omega)
          exact ⟨propagatedPost.effect.growth.1 _ previous.1,
            fun register bit => propagatedPost.effect.growth.2 _ register (previous.2 register bit)⟩
    · have test : ¬index < alloc.vec.Vec.len row := by scalar_tac
      simp only [test, ite_false] at tailOk
      have flowEq : ControlFlow.done (currentSets, currentWork, currentPending) = flow :=
        Result.ok_injective tailOk
      rw [←flowEq]
      refine ⟨invariant.storage, invariant.growth, invariant.stable, ?_⟩
      intro successor member
      obtain ⟨position, positionBound, atPosition⟩ := List.getElem_of_mem member
      have outside : ¬index.val < successors.val[pc.val]!.val.length := by simpa only [rowAt] using inside
      have completed := processed position (by omega)
      simpa only [←List.Inhabited_getElem_eq_getElem! _ _ positionBound, atPosition] using completed

end MemoryWrite.MayWrite
