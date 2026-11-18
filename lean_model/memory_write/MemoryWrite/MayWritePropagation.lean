/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteMergeEffect
import MemoryWrite.MayWriteQueue

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

structure Propagated (before after : WorkState) (target : Nat) (outgoing : List U64) : Prop where
  growth : Grows before.solution after.solution
  stable : StableWhenUnqueued before after
  targetPresent : after.solution.present target
  included : ∀ register, WordContains outgoing register → after.solution.incoming target register

theorem initialization_merge_propagates {before middle after : cache_graph.WorkSets}
    {pc count : Usize} {answer : Usize × Usize × Bool} {outgoing : List U64} {changed : Bool}
    (beforeInvariant : DirectoryInvariant before)
    (initialized : InitializedPc before middle pc count answer)
    (merged : MergedPc middle after pc.val outgoing changed)
    (beforeQueue afterQueue : List Nat)
    (queued : if answer.2.2 || changed then afterQueue = pc.val :: beforeQueue
      else afterQueue = beforeQueue) :
    Propagated ⟨workSolution before, beforeQueue⟩ ⟨workSolution after, afterQueue⟩ pc.val outgoing := by
  have growth : Grows (workSolution before) (workSolution after) := by
    constructor
    · intro address known
      exact (merged.present address).mpr ((initialized.present beforeInvariant address).mpr (Or.inl known))
    · intro address register bit
      exact (merged.incoming address register).mpr
        (Or.inl ((initialized.incoming beforeInvariant address register).mpr bit))
  obtain ⟨currentWords, currentAt, _, _⟩ := initialized.current
  refine ⟨growth, ?_, (merged.present pc.val).mpr ⟨currentWords, currentAt⟩, ?_⟩
  · intro address absent
    by_cases mustQueue : (answer.2.2 || changed) = true
    · have queueEq : afterQueue = pc.val :: beforeQueue := by simpa only [mustQueue, ite_true] using queued
      have absentParts : address ≠ pc.val ∧ address ∉ beforeQueue := by simpa only [queueEq, List.mem_cons, not_or] using absent
      refine ⟨absentParts.2, ?_, ?_⟩
      · rw [merged.present, initialized.present beforeInvariant]
        simp only [absentParts.1, or_false]
      · intro register
        rw [merged.incoming, initialized.incoming beforeInvariant]
        simp only [absentParts.1, false_and, or_false]
    · have freshFalse : answer.2.2 = false := by cases h : answer.2.2 <;> simp_all
      have changedFalse : changed = false := by cases h : changed <;> simp_all
      have queueEq : afterQueue = beforeQueue := by
        simpa only [freshFalse, changedFalse, Bool.or_false, Bool.false_eq_true, ite_false] using queued
      have wordsEqual := merged.unchanged changedFalse address
      refine ⟨by simpa only [queueEq] using absent, ?_, ?_⟩
      · constructor
        · rintro ⟨words, atWords⟩
          have middleAt := (wordsEqual words).mp atWords
          rcases (initialized.present beforeInvariant address).mp ⟨words, middleAt⟩ with known | same
          · exact known
          · subst address
            obtain ⟨existing, _, _, previous⟩ := initialized.current
            exact ⟨existing, previous freshFalse⟩
        · intro known
          exact growth.1 address known
      · intro register
        constructor
        · rintro ⟨words, atWords, bit⟩
          exact (initialized.incoming beforeInvariant address register).mp
            ⟨words, (wordsEqual words).mp atWords, bit⟩
        · exact growth.2 address register
  · intro register bit
    exact (merged.incoming pc.val register).mpr (Or.inr ⟨rfl, bit⟩)

structure SuccessorPost (sets : cache_graph.WorkSets) (work : alloc.vec.Vec Usize)
    (pending successor words : Usize) (outgoing : List U64)
    (result : Usize × cache_graph.WorkSets × alloc.vec.Vec Usize) : Prop where
  invariant : DirectoryInvariant result.2.1
  resources : DirectoryResources result.2.1
  sized : RowsSized result.2.1 words.val
  pendingBound : result.1.val ≤ result.2.2.val.length
  present : ∀ address, (workSolution result.2.1).present address ↔
    (workSolution sets).present address ∨ address = successor.val
  queue : queueView result.2.2 result.1 = queueView work pending ∨
    queueView result.2.2 result.1 = successor.val :: queueView work pending
  effect : Propagated (workView sets work pending) (workView result.2.1 result.2.2 result.1)
    successor.val outgoing

theorem successful_propagate_successor (sets : cache_graph.WorkSets)
    (work : alloc.vec.Vec Usize) (pending successor words : Usize) (out : Slice U64)
    (invariant : DirectoryInvariant sets) (resources : DirectoryResources sets)
    (sized : RowsSized sets words.val) (outLength : out.val.length = words.val)
    (pendingBound : pending.val ≤ work.val.length)
    (result : Usize × cache_graph.WorkSets × alloc.vec.Vec Usize)
    (success : cache_graph.propagate_successor sets work pending successor words out = .ok result) :
    SuccessorPost sets work pending successor words out.val result := by
  unfold cache_graph.propagate_successor at success
  obtain ⟨⟨⟨storage, offset, fresh⟩, middle⟩, initializedOk, tailOk⟩ :=
    MemoryWrite.bind_success _ _ _ success
  have initialization := rust_ensure_pc_resourced sets successor words invariant resources
  simp only [initializedOk, WP.spec_ok, WP.uncurry'_pair] at initialization
  obtain ⟨initialized, middleResources⟩ := initialization
  have middleSized := initialized.sized sized
  obtain ⟨oldWords, oldAt, _⟩ := initialized.current
  have lengths : oldWords.val.length = out.val.length :=
    (middleSized successor.val oldWords oldAt).trans outLength.symm
  obtain ⟨⟨changed, after⟩, mergedOk, returnOk⟩ := MemoryWrite.bind_success _ _ _ tailOk
  change (if fresh then Aeneas.Std.bind (cache_graph.push_work work pending successor)
      (fun (next, nextWork) => .ok (next, after, nextWork))
    else if changed then Aeneas.Std.bind (cache_graph.push_work work pending successor)
      (fun (next, nextWork) => .ok (next, after, nextWork))
    else .ok (pending, after, work)) = .ok result at returnOk
  have merged := rust_merge_row_effect middle successor.val storage offset out oldWords
    initialized.invariant initialized.page initialized.offset oldAt lengths
  simp only [mergedOk, WP.spec_ok, WP.uncurry'_pair] at merged
  have finish (next : Usize) (nextWork : alloc.vec.Vec Usize)
      (bound : next.val ≤ nextWork.val.length)
      (queued : if fresh || changed then queueView nextWork next = successor.val :: queueView work pending
        else queueView nextWork next = queueView work pending) :
      SuccessorPost sets work pending successor words out.val (next, after, nextWork) :=
    by
      refine ⟨merged.invariant, merged.resources middleResources, merged.sized words.val middleSized, bound, ?_, ?_,
        initialization_merge_propagates invariant initialized merged _ _ queued⟩
      · intro address
        rw [merged.present, initialized.present invariant]
      · by_cases condition : (fresh || changed) = true
        · exact Or.inr (by simpa only [condition, ite_true] using queued)
        · exact Or.inl (by simpa only [condition, Bool.false_eq_true, ite_false] using queued)
  have finishQueued (mustQueue : (fresh || changed) = true)
      (returned : Aeneas.Std.bind (cache_graph.push_work work pending successor)
        (fun (pending1, work1) => .ok (pending1, after, work1)) = .ok result) :
      SuccessorPost sets work pending successor words out.val result := by
    obtain ⟨⟨next, nextWork⟩, pushed, completed⟩ := MemoryWrite.bind_success _ _ _ returned
    have pushedSpec := successful_push_work work pending successor next nextWork pendingBound pushed
    have resultEq : (next, after, nextWork) = result := Result.ok_injective completed
    rw [←resultEq]
    apply finish next nextWork pushedSpec.2.2
    simpa only [mustQueue, ite_true] using pushedSpec.2.1
  cases fresh with
  | true =>
      apply finishQueued (by simp)
      simpa only [ite_true] using returnOk
  | false =>
      cases changed with
      | true =>
          apply finishQueued (by simp)
          simpa only [Bool.false_eq_true, ite_false, ite_true] using returnOk
      | false =>
          simp only [Bool.false_eq_true, ite_false] at returnOk
          have resultEq : (pending, after, work) = result := Result.ok_injective returnOk
          rw [←resultEq]
          exact finish pending work pendingBound (by simp)

end MemoryWrite.MayWrite
