/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteScanTotal
import MemoryWrite.MayWriteBounds
import MemoryWrite.MayWriteWordCount
import MemoryWrite.MayWriteCopyIncoming

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.MayWrite

private theorem main_progress_storage_pop {sets : cache_graph.WorkSets} {work : alloc.vec.Vec Usize}
    {pending remaining pc : Usize} {count words : Nat}
    (invariant : StorageInvariant sets work pending count words)
    (remainingBound : remaining.val ≤ pending.val)
    (queueEq : queueView work pending = pc.val :: queueView work remaining) :
    StorageInvariant sets work remaining count words := by
  refine ⟨invariant.directory, invariant.resources, invariant.sized, by have := invariant.pendingBound; omega,
    invariant.knownBound, ?_⟩
  intro address member
  exact invariant.queuedKnown address (by rw [queueEq]; exact List.mem_cons_of_mem _ member)

theorem rust_main_loop_returns (graph : cache_graph.MayWriteGraph) (bit : U32) (words : Usize)
    (sets : cache_graph.WorkSets) (work : alloc.vec.Vec Usize) (pending : Usize)
    (bounded : GraphBounded graph) (wordCountValue : words.val = wordCount graph.register_count.val)
    (initial : StorageInvariant sets work pending graph.reach.val.length words.val)
    (space : rank graph.reach.val.length (words.val * 64) (workSolution sets) pending.val < Usize.max) :
    cache_graph.may_write_sets_loop0 graph.reach graph.succ graph.writes bit sets words work pending
      ⦃ result => DirectoryInvariant result ∧ DirectoryResources result ∧ RowsSized result words.val ⦄ := by
  unfold cache_graph.may_write_sets_loop0
  apply loop.spec_decr_nat
    (fun state : cache_graph.WorkSets × alloc.vec.Vec Usize × Usize =>
      rank graph.reach.val.length (words.val * 64) (workSolution state.1) state.2.2.val)
    (fun state => StorageInvariant state.1 state.2.1 state.2.2 graph.reach.val.length words.val ∧
      rank graph.reach.val.length (words.val * 64) (workSolution state.1) state.2.2.val ≤
        rank graph.reach.val.length (words.val * 64) (workSolution sets) pending.val)
  · rintro ⟨currentSets, currentWork, currentPending⟩ ⟨storage, rankBound⟩
    have currentStorage : StorageInvariant currentSets currentWork currentPending graph.reach.val.length words.val := storage
    have currentRank : rank graph.reach.val.length (words.val * 64) (workSolution currentSets) currentPending.val ≤
        rank graph.reach.val.length (words.val * 64) (workSolution sets) pending.val := rankBound
    unfold cache_graph.may_write_sets_loop0.body
    by_cases pendingZero : currentPending = 0#usize
    · have test : (currentPending != 0#usize) = false := by simp only [pendingZero, bne_self_eq_false]
      simp only [test, Bool.false_eq_true, ite_false, WP.spec_ok]
      exact ⟨storage.directory, storage.resources, storage.sized⟩
    · have test : (currentPending != 0#usize) = true := by simpa only [bne_iff_ne] using pendingZero
      simp only [test, ite_true]
      have pendingPositive : 0 < currentPending.val := by scalar_tac
      step with Usize.sub_spec (x := currentPending) (y := 1#usize) (by scalar_tac)
        as ⟨remaining, remainingSpec⟩
      have remainingValue : remaining.val = currentPending.val - 1 := remainingSpec
      have remainingBound : remaining.val < currentWork.val.length := by have := currentStorage.pendingBound; omega
      step with alloc.vec.Vec.index_usize_spec currentWork remaining remainingBound as ⟨pc, pcSpec⟩
      have pcValue : pc = currentWork.val[remaining.val]! := by
        rw [pcSpec, List.Inhabited_getElem_eq_getElem! _ _ remainingBound]
      have queueEq := pop_prefix_queue currentWork currentPending remaining pc currentStorage.pendingBound
        pendingPositive remainingValue pcValue
      have currentKnown := currentStorage.queuedKnown pc.val (by rw [queueEq]; exact List.mem_cons_self)
      have pcBound := currentStorage.knownBound pc.val currentKnown
      have poppedStorage := main_progress_storage_pop currentStorage (by omega) queueEq
      have poppedRank := rank_pop graph.reach.val.length (words.val * 64) (workSolution currentSets)
        currentPending.val remaining.val pendingPositive remainingValue
      step with alloc.vec.Vec.index_usize_spec graph.reach pc pcBound as ⟨flag, flagSpec⟩
      simp only [lift, bind_ok]
      by_cases stopped : flag &&& bit = 0#u32
      · simp only [stopped, ite_true, WP.spec_ok]
        exact ⟨poppedStorage, (Nat.le_of_lt poppedRank).trans currentRank, poppedRank⟩
      · simp only [stopped, ite_false]
        obtain ⟨oldWords, oldAt⟩ := currentKnown
        step with rust_copy_incoming currentSets pc currentStorage.directory oldWords oldAt as ⟨incoming, copiedSpec⟩
        have incomingLength : incoming.val.length = words.val := by
          rw [copiedSpec]
          exact currentStorage.sized pc.val oldWords oldAt
        have writePcBound : pc.val < graph.writes.val.length := by rw [bounded.2.1]; exact pcBound
        have writeWordBounds : ∀ register ∈ graph.writes.val[pc.val]!.val, register.val / 64 < incoming.val.length := by
          intro register member
          rw [incomingLength, wordCountValue]
          exact graph_write_word_bound graph bounded pc pcBound register member
        step with rust_outgoing graph.writes pc incoming writePcBound writeWordBounds
          as ⟨outgoing, outgoingLength, containsIncoming, containsWrites⟩
        have successorPcBound : pc.val < graph.succ.val.length := by rw [bounded.1]; exact pcBound
        step with rust_successor_scan_total graph.succ currentSets words currentWork remaining pc outgoing
          graph.reach.val.length poppedStorage successorPcBound (bounded.2.2.1 pc.val pcBound)
          (outgoingLength.trans incomingLength) (lt_trans (lt_of_lt_of_le poppedRank currentRank) space)
          as ⟨nextSets, nextWork, nextPending, post, nextRank⟩
        exact ⟨post.storage, nextRank.trans ((Nat.le_of_lt poppedRank).trans currentRank),
          lt_of_le_of_lt nextRank poppedRank⟩
  · exact ⟨initial, Nat.le_refl _⟩

end MemoryWrite.MayWrite
