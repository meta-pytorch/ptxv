/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteSuccessorRank
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

theorem rust_propagate_successor_returns (sets : cache_graph.WorkSets)
    (work : alloc.vec.Vec Usize) (pending successor words : Usize) (out : Slice U64)
    (invariant : DirectoryInvariant sets) (resources : DirectoryResources sets)
    (sized : RowsSized sets words.val) (outLength : out.val.length = words.val)
    (pendingBound : pending.val ≤ work.val.length) (space : pending.val < Usize.max) :
    cache_graph.propagate_successor sets work pending successor words out ⦃ _ => True ⦄ := by
  unfold cache_graph.propagate_successor
  apply WP.spec_bind (rust_ensure_pc_resourced sets successor words invariant resources)
  rintro ⟨⟨storage, offset, fresh⟩, middle⟩ ⟨initialized, _⟩
  have middleSized := initialized.sized sized
  obtain ⟨oldWords, oldAt, _⟩ := initialized.current
  have lengths : oldWords.val.length = out.val.length :=
    (middleSized successor.val oldWords oldAt).trans outLength.symm
  apply WP.spec_bind (rust_merge_row_effect middle successor.val storage offset out oldWords
    initialized.invariant initialized.page initialized.offset oldAt lengths)
  rintro ⟨changed, after⟩ merged
  change (if fresh then Aeneas.Std.bind (cache_graph.push_work work pending successor)
      (fun (next, nextWork) => .ok (next, after, nextWork))
    else if changed then Aeneas.Std.bind (cache_graph.push_work work pending successor)
      (fun (next, nextWork) => .ok (next, after, nextWork))
    else .ok (pending, after, work)) ⦃ _ => True ⦄
  have push : Aeneas.Std.bind (cache_graph.push_work work pending successor)
      (fun (next, nextWork) => .ok (next, after, nextWork)) ⦃ _ => True ⦄ := by
    apply WP.spec_bind (rust_push_work work pending successor pendingBound space)
    rintro ⟨next, nextWork⟩ _
    simp only [WP.spec_ok]
  cases fresh with
  | true => simpa only [ite_true] using push
  | false =>
      cases changed with
      | true => simpa only [Bool.false_eq_true, ite_false, ite_true] using push
      | false => simp only [Bool.false_eq_true, ite_false, WP.spec_ok]

@[step]
theorem rust_propagate_successor_total (sets : cache_graph.WorkSets)
    (work : alloc.vec.Vec Usize) (pending successor words : Usize) (out : Slice U64)
    (count : Nat) (successorBound : successor.val < count)
    (invariant : DirectoryInvariant sets) (resources : DirectoryResources sets)
    (sized : RowsSized sets words.val) (outLength : out.val.length = words.val)
    (pendingBound : pending.val ≤ work.val.length) (space : pending.val < Usize.max) :
    cache_graph.propagate_successor sets work pending successor words out ⦃ result =>
      SuccessorPost sets work pending successor words out.val result ∧
      rank count (words.val * 64) (workSolution result.2.1) result.1.val ≤
        rank count (words.val * 64) (workSolution sets) pending.val ⦄ := by
  obtain ⟨result, executed, _⟩ := MemoryWrite.spec_returns _ _
    (rust_propagate_successor_returns sets work pending successor words out invariant resources sized
      outLength pendingBound space)
  simp only [executed, WP.spec_ok]
  exact ⟨successful_propagate_successor sets work pending successor words out invariant resources sized
    outLength pendingBound result executed,
    successful_propagate_rank sets work pending successor words out count successorBound invariant resources
      sized outLength pendingBound result executed⟩

end MemoryWrite.MayWrite
