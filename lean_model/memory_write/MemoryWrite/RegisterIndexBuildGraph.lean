/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexBuildRow

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

def GraphRepresents (index : register_index.RegisterIndex)
    (destinations : List (alloc.vec.Vec symbolic.RegisterKey)) (writes : List (alloc.vec.Vec Usize)) : Prop :=
  writes.length = destinations.length ∧
  ∀ pc, pc < destinations.length → RowRepresents index destinations[pc]!.val writes[pc]!.val

def GraphPrefix (index : register_index.RegisterIndex)
    (destinations : List (alloc.vec.Vec symbolic.RegisterKey)) (writes : List (alloc.vec.Vec Usize)) (pc : Nat) : Prop :=
  pc ≤ destinations.length ∧ writes.length = pc ∧
  ∀ earlier, earlier < pc → RowRepresents index destinations[earlier]!.val writes[earlier]!.val

theorem graph_prefix_append (before after : register_index.RegisterIndex)
    (preserves : Preserves before after) (destinations : List (alloc.vec.Vec symbolic.RegisterKey))
    (writes : List (alloc.vec.Vec Usize)) (pc : Nat) (row : alloc.vec.Vec Usize)
    (beforePrefix : GraphPrefix before destinations writes pc) (inside : pc < destinations.length)
    (produced : RowRepresents after destinations[pc]!.val row.val) :
    GraphPrefix after destinations (writes ++ [row]) (pc + 1) := by
  refine ⟨by omega, by simp only [List.length_append, List.length_singleton, beforePrefix.2.1], ?_⟩
  have writesLength := beforePrefix.2.1
  intro earlier bound
  by_cases old : earlier < pc
  · have writesBound : earlier < writes.length := by omega
    rw [←List.Inhabited_getElem_eq_getElem! (writes ++ [row]) earlier (by simp only [List.length_append, List.length_singleton]; omega),
      List.getElem_append_left writesBound, List.Inhabited_getElem_eq_getElem! _ _ writesBound]
    exact RowRepresents.grows before after preserves _ _ (beforePrefix.2.2 earlier old)
  · have atEnd : earlier = pc := by omega
    subst earlier
    have appendedAt : (writes ++ [row])[pc]! = row := by
      rw [←List.Inhabited_getElem_eq_getElem! (writes ++ [row]) pc (by simp only [List.length_append, List.length_singleton]; omega),
        List.getElem_append_right (by omega)]
      have atZero : pc - writes.length = 0 := by omega
      simp only [atZero, List.getElem_singleton]
    simpa only [appendedAt] using produced

theorem successful_build_graph_loop (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey))
    (before after : register_index.RegisterIndex) (writes result : alloc.vec.Vec (alloc.vec.Vec Usize))
    (pc : Usize) (valid : Valid before) (initialPrefix : GraphPrefix before destinations.val writes.val pc.val)
    (success : register_index.build_writes_loop0 destinations before writes pc = .ok (after, result)) :
    Valid after ∧ Preserves before after ∧ GraphRepresents after destinations.val result.val := by
  unfold register_index.build_writes_loop0 at success
  have execution := MemoryWrite.successful_loop_has_execution _ _ _ success
  apply execution.preserves
    (fun state : register_index.RegisterIndex × alloc.vec.Vec (alloc.vec.Vec Usize) × Usize =>
      Valid state.1 ∧ Preserves before state.1 ∧ GraphPrefix state.1 destinations.val state.2.1.val state.2.2.val)
    (fun state : register_index.RegisterIndex × alloc.vec.Vec (alloc.vec.Vec Usize) =>
      Valid state.1 ∧ Preserves before state.1 ∧ GraphRepresents state.1 destinations.val state.2.val)
    ⟨valid, Preserves.refl before, initialPrefix⟩
  rintro ⟨current, currentWrites, currentPc⟩ ⟨currentValid, grows, currentPrefix⟩ flow bodyOk
  change GraphPrefix current destinations.val currentWrites.val currentPc.val at currentPrefix
  unfold register_index.build_writes_loop0.body at bodyOk
  by_cases inside : currentPc.val < destinations.val.length
  · have test : currentPc < Slice.len destinations := by scalar_tac
    simp only [test, ite_true] at bodyOk
    obtain ⟨rowResult, rowOk, pushOk⟩ := MemoryWrite.bind_success _ _ _ bodyOk
    rcases rowResult with ⟨updated, row⟩
    have rowSpec := successful_build_row destinations currentPc current updated
      (alloc.vec.Vec.new Usize) row 0#usize inside currentValid
      ⟨by scalar_tac, by simp, by intro earlier bound; scalar_tac⟩ rowOk
    obtain ⟨appended, appendedOk, incrementOk⟩ := MemoryWrite.bind_success _ _ _ pushOk
    have appendedValue := MemoryWrite.successful_vec_push currentWrites row appended appendedOk
    obtain ⟨next, nextOk, returned⟩ := MemoryWrite.bind_success _ _ _ incrementOk
    have rawNextValue := MemoryWrite.successful_add currentPc 1#usize next nextOk
    have nextValue : next.val = currentPc.val + 1 := by scalar_tac
    have equal := Result.ok_injective returned
    cases equal
    refine ⟨rowSpec.1, grows.trans rowSpec.2.1, ?_⟩
    change GraphPrefix updated destinations.val appended.val next.val
    have graphEffect := graph_prefix_append current updated rowSpec.2.1 destinations.val
      currentWrites.val currentPc.val row currentPrefix inside rowSpec.2.2
    simpa only [appendedValue, nextValue] using graphEffect
  · have test : ¬currentPc < Slice.len destinations := by scalar_tac
    simp only [test, ite_false] at bodyOk
    have equal := Result.ok_injective bodyOk
    cases equal
    have finished : currentPc.val = destinations.val.length := by have := currentPrefix.1; omega
    exact ⟨currentValid, grows, currentPrefix.2.1.trans finished,
      fun earlier bound => currentPrefix.2.2 earlier (by omega)⟩

@[step]
theorem rust_new : register_index.RegisterIndex.new ⦃ result => Valid result ⦄ := by
  unfold register_index.RegisterIndex.new
  simp only [WP.spec_ok]
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · intro position inside; simp at inside
  · intro left right less inside; simp at inside
  · rintro key id ⟨position, inside, _, _⟩; simp at inside
  · rintro left right id ⟨position, inside, _, _⟩; simp at inside

theorem successful_build_writes (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey))
    (index : register_index.RegisterIndex) (writes : alloc.vec.Vec (alloc.vec.Vec Usize))
    (success : register_index.build_writes destinations = .ok (index, writes)) :
    Valid index ∧ GraphRepresents index destinations.val writes.val := by
  unfold register_index.build_writes at success
  obtain ⟨initial, initialOk, loopOk⟩ := MemoryWrite.bind_success _ _ _ success
  have initialSpec := rust_new
  simp only [initialOk, WP.spec_ok] at initialSpec
  have loopSpec := successful_build_graph_loop destinations initial index (alloc.vec.Vec.new _) writes 0#usize initialSpec
    ⟨by scalar_tac, by simp, by intro earlier bound; scalar_tac⟩ loopOk
  exact ⟨loopSpec.1, loopSpec.2.2⟩

end MemoryWrite.RegisterIndex
