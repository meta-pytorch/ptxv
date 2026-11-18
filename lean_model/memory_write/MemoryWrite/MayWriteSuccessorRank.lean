/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteInitializationProgress

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

theorem successful_propagate_rank (sets : cache_graph.WorkSets)
    (work : alloc.vec.Vec Usize) (pending successor words : Usize) (out : Slice U64)
    (count : Nat) (successorBound : successor.val < count)
    (invariant : DirectoryInvariant sets) (resources : DirectoryResources sets)
    (sized : RowsSized sets words.val) (outLength : out.val.length = words.val)
    (pendingBound : pending.val ≤ work.val.length)
    (result : Usize × cache_graph.WorkSets × alloc.vec.Vec Usize)
    (success : cache_graph.propagate_successor sets work pending successor words out = .ok result) :
    rank count (words.val * 64) (workSolution result.2.1) result.1.val ≤
      rank count (words.val * 64) (workSolution sets) pending.val := by
  unfold cache_graph.propagate_successor at success
  obtain ⟨⟨⟨storage, offset, fresh⟩, middle⟩, initializedOk, tailOk⟩ :=
    MemoryWrite.bind_success _ _ _ success
  have initialization := rust_ensure_pc_resourced sets successor words invariant resources
  simp only [initializedOk, WP.spec_ok, WP.uncurry'_pair] at initialization
  obtain ⟨initialized, _⟩ := initialization
  have middleSized := initialized.sized sized
  obtain ⟨oldWords, oldAt, _⟩ := initialized.current
  have oldLength := middleSized successor.val oldWords oldAt
  have lengths : oldWords.val.length = out.val.length := oldLength.trans outLength.symm
  obtain ⟨⟨changed, after⟩, mergedOk, returnOk⟩ := MemoryWrite.bind_success _ _ _ tailOk
  change (if fresh then Aeneas.Std.bind (cache_graph.push_work work pending successor)
      (fun (next, nextWork) => .ok (next, after, nextWork))
    else if changed then Aeneas.Std.bind (cache_graph.push_work work pending successor)
      (fun (next, nextWork) => .ok (next, after, nextWork))
    else .ok (pending, after, work)) = .ok result at returnOk
  have merged := rust_merge_row_effect middle successor.val storage offset out oldWords
    initialized.invariant initialized.page initialized.offset oldAt lengths
  simp only [mergedOk, WP.spec_ok, WP.uncurry'_pair] at merged
  have progress := rust_merge_row_adds_incoming middle successor.val storage offset out oldWords
    initialized.invariant initialized.page initialized.offset oldAt lengths
  simp only [mergedOk, WP.spec_ok, WP.uncurry'_pair] at progress
  rw [oldLength] at progress
  have growth := initialization_merge_growth invariant initialized merged
  have finishQueued (mustQueue : (fresh || changed) = true)
      (returned : Aeneas.Std.bind (cache_graph.push_work work pending successor)
        (fun (next, nextWork) => .ok (next, after, nextWork)) = .ok result) :
      rank count (words.val * 64) (workSolution result.2.1) result.1.val ≤
        rank count (words.val * 64) (workSolution sets) pending.val := by
    obtain ⟨⟨next, nextWork⟩, pushed, completed⟩ := MemoryWrite.bind_success _ _ _ returned
    have pushedSpec := successful_push_work work pending successor next nextWork pendingBound pushed
    have resultEq : (next, after, nextWork) = result := Result.ok_injective completed
    rw [←resultEq]
    have strict := queued_initialization_merge_strict count successorBound invariant initialized merged progress mustQueue
    simpa only [pushedSpec.1] using rank_grows_with_queue count (words.val * 64)
      (workSolution sets) (workSolution after) pending.val growth strict
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
          exact rank_grows_no_queue count (words.val * 64) (workSolution sets) (workSolution after)
            pending.val growth

end MemoryWrite.MayWrite
