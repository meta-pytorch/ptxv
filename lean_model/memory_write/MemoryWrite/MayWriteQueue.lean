/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteAllocation
import MemoryWrite.PartialLoops
import MemoryWrite.MayWriteWorklist
import MemoryWrite.MayWriteStorageInvariant

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

def queueView (work : alloc.vec.Vec Usize) (pending : Usize) : List Nat :=
  (work.val.take pending.val).reverse.map UScalar.val

def workView (sets : cache_graph.WorkSets) (work : alloc.vec.Vec Usize) (pending : Usize) : WorkState :=
  ⟨workSolution sets, queueView work pending⟩

theorem push_prefix_queue (work nextWork : alloc.vec.Vec Usize) (pending next pc : Usize)
    (prefixEq : nextWork.val.take next.val = work.val.take pending.val ++ [pc]) :
    queueView nextWork next = pc.val :: queueView work pending := by
  simp only [queueView, prefixEq, List.reverse_append, List.reverse_singleton,
    List.singleton_append, List.map_cons]

theorem pop_prefix_queue (work : alloc.vec.Vec Usize) (pending remaining pc : Usize)
    (bound : pending.val ≤ work.val.length) (nonempty : 0 < pending.val)
    (remainingValue : remaining.val = pending.val - 1)
    (pcValue : pc = work.val[remaining.val]!) :
    queueView work pending = pc.val :: queueView work remaining := by
  have remainingBound : remaining.val < work.val.length := by omega
  have pendingValue : pending.val = remaining.val + 1 := by omega
  simp only [queueView, pendingValue, List.take_succ_eq_append_getElem remainingBound,
    List.reverse_append, List.reverse_singleton, List.singleton_append, List.map_cons,
    List.Inhabited_getElem_eq_getElem! _ _ remainingBound, pcValue]

theorem successful_push_work (work : alloc.vec.Vec Usize) (pending pc next : Usize)
    (result : alloc.vec.Vec Usize) (bound : pending.val ≤ work.val.length)
    (success : cache_graph.push_work work pending pc = .ok (next, result)) :
    next.val = pending.val + 1 ∧ queueView result next = pc.val :: queueView work pending ∧
      next.val ≤ result.val.length := by
  have space : pending.val < Usize.max := by
    by_contra noSpace
    have pendingMax : pending.val = Usize.max := by scalar_tac
    have workMax : work.val.length = Usize.max := by have := work.property; omega
    have full : pending = alloc.vec.Vec.len work := by scalar_tac
    have pushFailure : alloc.vec.Vec.push work pc = .fail .maximumSizeExceeded := by
      unfold alloc.vec.Vec.push
      dsimp only
      split
      · rename_i possible
        simp only [workMax] at possible
        scalar_tac
      · rfl
    unfold cache_graph.push_work at success
    simp only [full, ite_true, pushFailure, bind_fail, fail_not_ok] at success
  have specification := rust_push_work work pending pc bound space
  simp only [success, WP.spec_ok, WP.uncurry'_pair] at specification
  exact ⟨specification.1, push_prefix_queue work result pending next pc specification.2.1,
    specification.2.2⟩

end MemoryWrite.MayWrite
