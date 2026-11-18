/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteScan
import MemoryWrite.MayWriteOutgoing
import MemoryWrite.MayWriteCopyIncoming
import MemoryWrite.MayWriteWordCount

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

structure MainInvariant (graph : cache_graph.MayWriteGraph) (bit : U32) (words : Usize)
    (initial : Solution) (state : cache_graph.WorkSets × alloc.vec.Vec Usize × Usize) : Prop where
  storage : StorageInvariant state.1 state.2.1 state.2.2 graph.reach.val.length words.val
  settled : WorkInvariant (graphView graph bit) (workView state.1 state.2.1 state.2.2)
  growth : Grows initial (workSolution state.1)

structure MainPost (graph : cache_graph.MayWriteGraph) (bit : U32) (words : Usize)
    (initial : Solution) (result : cache_graph.WorkSets) : Prop where
  directory : DirectoryInvariant result
  resources : DirectoryResources result
  sized : RowsSized result words.val
  knownBound : ∀ pc, (workSolution result).present pc → pc < graph.reach.val.length
  closed : Closed (graphView graph bit) (workSolution result)
  growth : Grows initial (workSolution result)

theorem storage_pop {sets : cache_graph.WorkSets} {work : alloc.vec.Vec Usize}
    {pending remaining pc : Usize} {count words : Nat}
    (invariant : StorageInvariant sets work pending count words)
    (remainingBound : remaining.val ≤ pending.val)
    (queueEq : queueView work pending = pc.val :: queueView work remaining) :
    StorageInvariant sets work remaining count words := by
  refine ⟨invariant.directory, invariant.resources, invariant.sized, by have := invariant.pendingBound; omega,
    invariant.knownBound, ?_⟩
  intro address member
  exact invariant.queuedKnown address (by rw [queueEq]; exact List.mem_cons_of_mem _ member)

theorem successful_main_loop (graph : cache_graph.MayWriteGraph) (bit : U32) (words : Usize)
    (sets : cache_graph.WorkSets) (work : alloc.vec.Vec Usize) (pending : Usize)
    (bounded : GraphBounded graph) (wordCountValue : words.val = wordCount graph.register_count.val)
    (initialStorage : StorageInvariant sets work pending graph.reach.val.length words.val)
    (initialSettled : WorkInvariant (graphView graph bit) (workView sets work pending))
    (result : cache_graph.WorkSets)
    (success : cache_graph.may_write_sets_loop0 graph.reach graph.succ graph.writes bit
      sets words work pending = .ok result) :
    MainPost graph bit words (workSolution sets) result := by
  unfold cache_graph.may_write_sets_loop0 at success
  have execution := MemoryWrite.successful_loop_has_execution _ _ _ success
  apply execution.preserves (MainInvariant graph bit words (workSolution sets))
    (MainPost graph bit words (workSolution sets))
  · exact ⟨initialStorage, initialSettled, ⟨fun _ known => known, fun _ _ bit => bit⟩⟩
  · rintro ⟨currentSets, currentWork, currentPending⟩ invariant flow bodyOk
    have storage : StorageInvariant currentSets currentWork currentPending graph.reach.val.length words.val := invariant.storage
    have settled : WorkInvariant (graphView graph bit) (workView currentSets currentWork currentPending) := invariant.settled
    unfold cache_graph.may_write_sets_loop0.body at bodyOk
    by_cases pendingZero : currentPending = 0#usize
    · have test : (currentPending != 0#usize) = false := by simp only [pendingZero, bne_self_eq_false]
      simp only [test, Bool.false_eq_true, ite_false] at bodyOk
      have flowEq : ControlFlow.done currentSets = flow := Result.ok_injective bodyOk
      rw [←flowEq]
      refine ⟨storage.directory, storage.resources, storage.sized, storage.knownBound, ?_, invariant.growth⟩
      apply exhausted_worklist_is_closed (graphView graph bit) _ settled
      simp only [workView, queueView, pendingZero]
      rfl
    · have test : (currentPending != 0#usize) = true := by simpa only [bne_iff_ne] using pendingZero
      simp only [test, ite_true] at bodyOk
      obtain ⟨remaining, removed, pcOk⟩ := MemoryWrite.bind_success _ _ _ bodyOk
      have pendingPositive : 0 < currentPending.val := by scalar_tac
      have remainingSpec := Usize.sub_spec (x := currentPending) (y := 1#usize) (by scalar_tac)
      simp only [removed, WP.spec_ok] at remainingSpec
      have remainingValue : remaining.val = currentPending.val - 1 := remainingSpec.1
      have remainingBound : remaining.val < currentWork.val.length := by have := storage.pendingBound; omega
      obtain ⟨pc, indexed, reachOk⟩ := MemoryWrite.bind_success _ _ _ pcOk
      have pcSpec := rust_vec_index currentWork remaining remainingBound
      simp only [indexed, WP.spec_ok] at pcSpec
      have pcValue : pc = currentWork.val[remaining.val]! := by
        rw [pcSpec, List.Inhabited_getElem_eq_getElem! _ _ remainingBound]
      have queueEq := pop_prefix_queue currentWork currentPending remaining pc storage.pendingBound
        pendingPositive remainingValue pcValue
      have currentKnown := storage.queuedKnown pc.val (by rw [queueEq]; exact List.mem_cons_self)
      have pcBound := storage.knownBound pc.val currentKnown
      have poppedStorage := storage_pop storage (by omega) queueEq
      obtain ⟨flag, flagOk, tailOk⟩ := MemoryWrite.bind_success _ _ _ reachOk
      have flagSpec := rust_vec_index graph.reach pc pcBound
      simp only [flagOk, WP.spec_ok] at flagSpec
      have flagAt : graph.reach.val[pc.val]? = some flag := by
        rw [List.getElem?_eq_getElem pcBound, ←flagSpec]
      simp only [lift, bind_ok] at tailOk
      by_cases stopped : flag &&& bit = 0#u32
      · simp only [stopped, ite_true] at tailOk
        have flowEq : ControlFlow.cont (currentSets, currentWork, remaining) = flow := Result.ok_injective tailOk
        rw [←flowEq]
        have disabled : ¬(graphView graph bit).enabled pc.val := by
          rintro ⟨otherFlag, otherLookup, enabled⟩
          have equal : otherFlag = flag := Option.some.inj (otherLookup.symm.trans flagAt)
          exact enabled (by simpa only [equal] using stopped)
        exact ⟨poppedStorage, disabled_step_preserves_invariant (graphView graph bit)
          (workView currentSets currentWork currentPending) pc.val (queueView currentWork remaining)
          queueEq disabled settled, invariant.growth⟩
      · simp only [stopped, ite_false] at tailOk
        obtain ⟨incoming, copied, outgoingOk⟩ := MemoryWrite.bind_success _ _ _ tailOk
        obtain ⟨oldWords, oldAt⟩ := currentKnown
        have copiedSpec := rust_copy_incoming currentSets pc storage.directory oldWords oldAt
        simp only [copied, WP.spec_ok] at copiedSpec
        have incomingLength : incoming.val.length = words.val := by
          rw [copiedSpec]
          exact storage.sized pc.val oldWords oldAt
        obtain ⟨outgoing, wrote, scannedOk⟩ := MemoryWrite.bind_success _ _ _ outgoingOk
        have writePcBound : pc.val < graph.writes.val.length := by rw [bounded.2.1]; exact pcBound
        have writeWordBounds : ∀ register ∈ graph.writes.val[pc.val]!.val, register.val / 64 < incoming.val.length := by
          intro register member
          rw [incomingLength, wordCountValue]
          exact graph_write_word_bound graph bounded pc pcBound register member
        have outgoingSpec := rust_outgoing graph.writes pc incoming writePcBound writeWordBounds
        simp only [wrote, WP.spec_ok] at outgoingSpec
        obtain ⟨outgoingLength, containsIncoming, containsWrites⟩ := outgoingSpec
        obtain ⟨⟨nextSets, nextWork, nextPending⟩, scanned, completed⟩ := MemoryWrite.bind_success _ _ _ scannedOk
        have successorPcBound : pc.val < graph.succ.val.length := by rw [bounded.1]; exact pcBound
        have scanPost := successful_successor_scan graph.succ currentSets words currentWork remaining pc outgoing
          graph.reach.val.length poppedStorage successorPcBound (bounded.2.2.1 pc.val pcBound)
          (outgoingLength.trans incomingLength) (nextSets, nextWork, nextPending) scanned
        have flowEq : ControlFlow.cont (nextSets, nextWork, nextPending) = flow := Result.ok_injective completed
        rw [←flowEq]
        refine ⟨scanPost.storage, ?_, invariant.growth.trans scanPost.growth⟩
        have outgoingCovers : ∀ register,
            (workSolution currentSets).incoming pc.val register ∨ (graphView graph bit).writes pc.val register →
              WordContains outgoing.val register := by
          intro register written
          rcases written with incomingBit | ownWrite
          · obtain ⟨observed, observedAt, bit⟩ := incomingBit
            have sameWords := words_at_unique storage.directory observedAt oldAt
            apply containsIncoming register
            simpa only [copiedSpec, ←sameWords] using bit
          · obtain ⟨row, rowLookup, address, addressMember, addressValue⟩ := ownWrite
            have rowAt : row = graph.writes.val[pc.val]! := by
              rw [List.getElem?_eq_getElem writePcBound] at rowLookup
              exact (Option.some.inj rowLookup).symm.trans
                (List.Inhabited_getElem_eq_getElem! _ _ writePcBound)
            simpa only [addressValue] using containsWrites address (by simpa only [rowAt] using addressMember)
        have propagated : ∀ successor, (graphView graph bit).edge pc.val successor →
            (workSolution nextSets).present successor ∧
            ∀ register, WordContains outgoing.val register → (workSolution nextSets).incoming successor register := by
          intro successor edge
          obtain ⟨row, rowLookup, address, addressMember, addressValue⟩ := edge
          have rowAt : row = graph.succ.val[pc.val]! := by
            rw [List.getElem?_eq_getElem successorPcBound] at rowLookup
            exact (Option.some.inj rowLookup).symm.trans
              (List.Inhabited_getElem_eq_getElem! _ _ successorPcBound)
          have propagated := scanPost.propagated address (by simpa only [rowAt] using addressMember)
          simpa only [addressValue] using propagated
        exact propagation_preserves_invariant (graphView graph bit)
          (workView currentSets currentWork currentPending) pc.val (queueView currentWork remaining)
          (workView nextSets nextWork nextPending) (WordContains outgoing.val)
          queueEq settled outgoingCovers scanPost.growth scanPost.stable propagated

end MemoryWrite.MayWrite
