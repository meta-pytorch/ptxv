/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Code.Funs
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Aeneas.Std.WP Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option debug.skipKernelTC false
set_option Elab.async false

namespace MemorySafety.Source

theorem source_bind_success {α β : Type} (computation : Result α) (next : α → Result β)
    (value : β) (accepted : Aeneas.Std.bind computation next = .ok value) :
    ∃ input, computation = .ok input ∧ next input = .ok value := by
  cases computation with
  | ret input => exact ⟨input, rfl, by simpa only [bind_ok] using accepted⟩
  | vis effect continuation => simp only [bind_vis, vis_not_ok] at accepted
  | div => simp only [bind_div, div_not_ok] at accepted

theorem vec_clone_exact {α : Type} (implementation : core.clone.Clone α)
    (identity : ∀ value, implementation.clone value = .ok value) (values : alloc.vec.Vec α) :
    alloc.vec.CloneVec.clone implementation values = .ok values := by
  have same := Slice.clone_spec (s := values.slice) (fun value _ => identity value)
  simp only [alloc.vec.CloneVec.clone]
  cases found : Slice.clone implementation.clone values.slice with
  | ret result =>
    have equal : values.slice = result := by simpa only [found, spec_ok] using same
    cases equal
    cases values
    simp
  | vis effect continuation => simp only [found, spec_vis] at same
  | div => simp only [found, spec_div] at same

theorem vec_push_success {α : Type} (values next : alloc.vec.Vec α) (value : α)
    (computed : alloc.vec.Vec.push values value = .ok next) :
    next.val = values.val ++ [value] := by
  unfold alloc.vec.Vec.push at computed
  dsimp only at computed
  split at computed
  · simp only [ok.injEq] at computed
    rw [← computed]
    simp [List.concat_eq_append]
  · simp at computed

theorem vec_index_success {α : Type} (values : alloc.vec.Vec α) (index : Usize) (value : α)
    (computed : alloc.vec.Vec.index_usize values index = .ok value) :
    values.val[index.val]? = some value := by
  unfold alloc.vec.Vec.index_usize at computed
  cases found : values.val[index.val]? with
  | none => simp [found] at computed
  | some input => simpa [found] using computed

theorem vec_index_mut_success {α : Type} (values : alloc.vec.Vec α) (index : Usize)
    (value : α) (replace : α → alloc.vec.Vec α)
    (computed : alloc.vec.Vec.index_mut_usize values index = .ok (value, replace)) :
    values.val[index.val]? = some value ∧ replace = alloc.vec.Vec.set values index := by
  unfold alloc.vec.Vec.index_mut_usize at computed
  obtain ⟨input, read, rest⟩ := source_bind_success _ _ _ computed
  simp only [ok.injEq, Prod.mk.injEq] at rest
  rcases rest with ⟨rfl, rfl⟩
  exact ⟨vec_index_success values index input read, rfl⟩


theorem source_state_clone (state : memory_safety.State) :
    memory_safety.State.Insts.CoreCloneClone.clone state = .ok state := by
  unfold memory_safety.State.Insts.CoreCloneClone.clone
  rw [vec_clone_exact core.clone.CloneUsize (by intro value; rfl)]
  simp only [bind_ok]
  rw [vec_clone_exact core.clone.CloneBool (by intro value; rfl)]
  simp only [bind_ok]
  rw [vec_clone_exact memory_safety.WrittenSpan.Insts.CoreCloneClone (by intro value; rfl)]
  simp only [bind_ok]
  rw [vec_clone_exact memory_safety.AccessRecord.Insts.CoreCloneClone (by intro value; rfl)]
  cases state
  simp [core.clone.impls.CloneBool.clone, lift]

theorem source_move_clone (movement : memory_safety.Move) :
    memory_safety.Move.Insts.CoreCloneClone.clone movement = .ok movement := by
  simp only [memory_safety.Move.Insts.CoreCloneClone.clone,
    memory_safety.Event.Insts.CoreCloneClone.clone, bind_ok]
  rw [vec_clone_exact core.clone.CloneUsize (by intro value; rfl)]
  cases movement
  simp

theorem source_usize_increment (index : Usize) (bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits) :
    (index + 1#usize : Result Usize) = .ok (Usize.ofNatCore (index.val + 1) bound) := by
  have platformBound : index.val + 1 < 2 ^ System.Platform.numBits := bound
  simp [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
    UScalar.check_bounds, platformBound, Result.ofOption]
  rfl

end MemorySafety.Source
