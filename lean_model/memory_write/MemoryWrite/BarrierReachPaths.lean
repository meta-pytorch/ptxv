/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierReachTotal

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierReach

def Edge (successors : List (alloc.vec.Vec Usize)) (pc destination : Nat) : Prop :=
  ∃ row, successors[pc]? = some row ∧ ∃ address ∈ row.val, address.val = destination

inductive CanReach (edge : Nat → Nat → Prop) (label : Nat → Prop) : Nat → Prop
  | here {pc : Nat} : label pc → CanReach edge label pc
  | next {pc successor : Nat} : edge pc successor → CanReach edge label successor → CanReach edge label pc

theorem CanReach.map {beforeEdge afterEdge : Nat → Nat → Prop} {beforeLabel afterLabel : Nat → Prop}
    (edges : ∀ pc destination, beforeEdge pc destination → afterEdge pc destination)
    (labels : ∀ pc, beforeLabel pc → afterLabel pc) {pc : Nat}
    (path : CanReach beforeEdge beforeLabel pc) : CanReach afterEdge afterLabel pc := by
  induction path with
  | here label => exact .here (labels _ label)
  | next edge path ih => exact .next (edges _ _ edge) ih

theorem closed_paths_preserve_bits (successors : List (alloc.vec.Vec Usize)) (initial reach : List U32)
    (bounded : EdgesBounded successors reach.length) (growth : Grows initial reach)
    (closed : Closed successors reach) (bit : Bit) (pc : Nat)
    (path : CanReach (Edge successors) (fun address => HasBitAt initial address bit) pc) :
    HasBitAt reach pc bit := by
  induction path with
  | here present => exact growth _ _ present
  | @next source successor edge path ih =>
      obtain ⟨row, rowAt, address, member, destination⟩ := edge
      have sourceBound : source < reach.length := by
        have inside := (List.getElem?_eq_some_iff.mp rowAt).1
        rw [bounded.1] at inside
        exact inside
      have rowEq : row = successors[source]! := by
        have inside : source < successors.length := by rw [bounded.1]; exact sourceBound
        simpa only [List.getElem?_eq_getElem inside,
          List.Inhabited_getElem_eq_getElem! _ _ inside, Option.some.injEq] using rowAt.symm
      have successorBound := bit_at_bound reach successor bit ih
      have wordPresent := (bit_at_row reach successor successorBound bit).mp ih
      have includes := closed source sourceBound address (by simpa only [rowEq] using member)
      apply (bit_at_row reach source sourceBound bit).mpr
      apply includes bit
      simpa only [destination] using wordPresent

@[step]
theorem rust_propagate_conservative (successors : Slice (alloc.vec.Vec Usize)) (initial : alloc.vec.Vec U32)
    (bounded : EdgesBounded successors.val initial.val.length) :
    barrier_reach.propagate successors initial ⦃ reach =>
      PropagationPost successors.val initial.val reach.val ∧
      ∀ (bit : Bit) (pc : Nat),
        CanReach (Edge successors.val) (fun address => HasBitAt initial.val address bit) pc →
        HasBitAt reach.val pc bit ⦄ := by
  step with rust_propagate successors initial bounded as ⟨reach, post⟩
  refine ⟨post, ?_⟩
  intro bit pc path
  exact closed_paths_preserve_bits successors.val initial.val reach.val
    (by simpa only [post.1] using bounded) post.2.1 post.2.2 bit pc path

end MemoryWrite.BarrierReach
