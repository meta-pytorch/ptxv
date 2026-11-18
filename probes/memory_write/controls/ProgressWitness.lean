/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierReachScan

open Aeneas Aeneas.Std Result PtxvMemoryWrite

namespace ProgressWitness

def successors : Slice (alloc.vec.Vec Usize) := Slice.new _
def reach : alloc.vec.Vec U32 := alloc.vec.Vec.new _

theorem scan_empty : barrier_reach.scan successors reach = .ok (false, reach) := by
  have specified : barrier_reach.scan successors reach ⦃ result => result = (false, reach) ⦄ := by
    unfold barrier_reach.scan barrier_reach.scan_loop
    have zero : alloc.vec.Vec.len reach = 0#usize := by apply UScalar.eq_of_val_eq; simp [reach]
    rw [zero]
    apply loop.spec_decr_nat (fun _ => 0)
      (fun state : alloc.vec.Vec U32 × Bool × Usize => state = (reach, false, 0#usize))
    · rintro state equal
      cases equal
      unfold barrier_reach.scan_loop.body
      simp only [gt_iff_lt, lt_self_iff_false, ite_false, WP.spec_ok]
    · rfl
  obtain ⟨result, executed, equal⟩ := MemoryWrite.spec_returns _ _ specified
  simpa only [equal] using executed

theorem actual_body_self_loop : barrier_reach.propagate_loop.body successors reach true =
    .ok (.cont (reach, true)) := by
  unfold barrier_reach.propagate_loop.body
  rw [scan_empty]
  simp [bind_ok]

theorem no_strict_progress (measure : alloc.vec.Vec U32 × Bool → Nat) :
    ¬∃ next, barrier_reach.propagate_loop.body successors reach true = .ok (.cont next) ∧
      measure next < measure (reach, true) := by
  rintro ⟨next, executed, smaller⟩
  rw [actual_body_self_loop] at executed
  have same : next = (reach, true) := by cases Result.ok_injective executed; rfl
  rw [same] at smaller
  exact Nat.lt_irrefl _ smaller

end ProgressWitness
