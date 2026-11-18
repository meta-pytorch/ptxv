/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Model

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

structure LoadObservation where
  destination : Key
  sourceByte : Nat
  value : BitVec 32

def selectedLoadIndices (mask : U8) (start count : Nat) : List Nat :=
  (List.range count).filter fun index => mask.bv.getLsbD (start + index)

def selectedLoadPrefix (first : Reg) (mask : U8) (start : Nat)
    (memoryWord : Nat → BitVec 32) (count : Nat) : List LoadObservation :=
  (selectedLoadIndices mask start count).mapIdx fun rank index =>
    ⟨⟨first.file, first.idx.val + rank⟩, 4 * (start + index), memoryWord (start + index)⟩

def selectedLoadHalf (first : Reg) (mask : U8) (start : Nat)
    (memoryWord : Nat → BitVec 32) : List LoadObservation :=
  selectedLoadPrefix first mask start memoryWord 4

def maskedLoadObservations (operation : Op) (memoryWord : Nat → BitVec 32) :
    List LoadObservation :=
  match operation with
  | .Ld256 high low _ _ mask _ =>
    selectedLoadHalf low mask 0 memoryWord ++ selectedLoadHalf high mask 4 memoryWord
  | _ => []

def observedLoadWrite (destination : Reg) (value : U32) (sourceByte : U8) : LoadObservation :=
  ⟨key destination, sourceByte.val, value.bv⟩

@[simp] theorem selected_load_indices_zero (mask : U8) (start : Nat) :
    selectedLoadIndices mask start 0 = [] := by simp [selectedLoadIndices]

@[simp] theorem selected_load_prefix_zero (first : Reg) (mask : U8) (start : Nat)
    (memoryWord : Nat → BitVec 32) : selectedLoadPrefix first mask start memoryWord 0 = [] := by
  simp [selectedLoadPrefix]

theorem selected_load_indices_length (mask : U8) (start count : Nat) :
    (selectedLoadIndices mask start count).length ≤ count := by
  simpa [selectedLoadIndices] using List.length_filter_le
    (fun index => mask.bv.getLsbD (start + index)) (List.range count)

theorem selected_load_indices_mono (mask : U8) (start left right : Nat) (h : left ≤ right) :
    (selectedLoadIndices mask start left).length ≤ (selectedLoadIndices mask start right).length := by
  exact (List.Sublist.filter (fun index => mask.bv.getLsbD (start + index))
    (List.range_sublist.mpr h)).length_le

theorem selected_load_indices_succ (mask : U8) (start count : Nat) :
    selectedLoadIndices mask start (count + 1) = selectedLoadIndices mask start count ++
      if mask.bv.getLsbD (start + count) then [count] else [] := by
  cases hb : mask.bv.getLsbD (start + count) <;>
    simp [selectedLoadIndices, List.range_succ, List.filter_append, hb]

theorem selected_load_prefix_succ (first : Reg) (mask : U8) (start count : Nat)
    (memoryWord : Nat → BitVec 32) :
    selectedLoadPrefix first mask start memoryWord (count + 1) =
      selectedLoadPrefix first mask start memoryWord count ++
      if mask.bv.getLsbD (start + count) then
        [⟨⟨first.file, first.idx.val + (selectedLoadIndices mask start count).length⟩,
          4 * (start + count), memoryWord (start + count)⟩] else [] := by
  rw [selectedLoadPrefix, selected_load_indices_succ]
  split <;> simp [selectedLoadPrefix]

theorem selected_load_rank_bound (mask : U8) (start count : Nat) (hcount : count < 4)
    (hselected : mask.bv.getLsbD (start + count) = true) :
    (selectedLoadIndices mask start count).length < (halfWords mask start).length := by
  have hmono := selected_load_indices_mono mask start (count + 1) 4 (by omega)
  rw [selected_load_indices_succ, hselected] at hmono
  simpa [halfWords, selectedLoadIndices] using hmono

theorem selected_load_prefix_subset (first : Reg) (mask : U8) (start count : Nat)
    (memoryWord : Nat → BitVec 32) (hcount : count ≤ 4) (action : LoadObservation)
    (hmember : action ∈ selectedLoadPrefix first mask start memoryWord count) :
    action ∈ selectedLoadHalf first mask start memoryWord := by
  have hsplit : 4 = count + (4 - count) := by omega
  unfold selectedLoadHalf selectedLoadPrefix selectedLoadIndices at *
  have hrange := congrArg List.range hsplit
  rw [List.range_add] at hrange
  rw [hrange, List.filter_append, List.mapIdx_append]
  exact List.mem_append.mpr (Or.inl hmember)

theorem selected_load_destinations (first : Reg) (mask : U8) (start : Nat)
    (memoryWord : Nat → BitVec 32) :
    (selectedLoadHalf first mask start memoryWord).map LoadObservation.destination =
      consecutive first (halfWords mask start).length := by
  apply List.ext_getElem
  · simp [selectedLoadHalf, selectedLoadPrefix, selectedLoadIndices, consecutive, halfWords]
  · intro index hi hj
    simp [selectedLoadHalf, selectedLoadPrefix, selectedLoadIndices, consecutive, halfWords]

theorem masked_load_destinations (high low : Reg) (address : Addr) (sync : Option MemSync)
    (mask : U8) (nc : Bool) (memoryWord : Nat → BitVec 32) :
    (maskedLoadObservations (.Ld256 high low address sync mask nc) memoryWord).map
        LoadObservation.destination = maskedWrites low high mask := by
  simp [maskedLoadObservations, maskedWrites, selected_load_destinations]

end Instruction
