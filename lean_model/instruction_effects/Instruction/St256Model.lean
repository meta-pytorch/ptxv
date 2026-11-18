/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.PayloadReadCoverage
import Instruction.PayloadAssembly

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

def st256Window (low high : Reg) (index : Nat) : ByteWindow :=
  let first := if index < 4 then low else high
  ⟨⟨first.file, first.idx.val + index % 4⟩, 4 * index, 4⟩

def st256Prefix (low high : Reg) (count : Nat) : List ByteWindow :=
  (List.range count).map (st256Window low high)

def st256Windows (operation : Op) : Option (List ByteWindow) :=
  match operation with
  | .St256 low high _ _ => some (st256Prefix low high 8)
  | _ => none

theorem st256_prefix_succ (low high : Reg) (count : Nat) :
    st256Prefix low high (count + 1) = st256Prefix low high count ++ [st256Window low high count] := by
  simp [st256Prefix, List.range_succ, List.map_append]

theorem st256_window_declared_read (low high : Reg) (address : Addr) (sync : Option MemSync)
    (window : ByteWindow) (hwindow : window ∈ st256Prefix low high 8) :
    window.source ∈ (memoryReads (.St256 low high address sync)).getD [] := by
  rcases List.mem_map.mp hwindow with ⟨index, hi, rfl⟩
  have hi : index < 8 := List.mem_range.mp hi
  have hrem : index % 4 < 4 := Nat.mod_lt _ (by decide)
  by_cases hlow : index < 4
  · have hm : (st256Window low high index).source ∈ consecutive low 4 := by
      apply List.mem_map.mpr
      exact ⟨index % 4, List.mem_range.mpr hrem, by simp [st256Window, hlow]⟩
    simp only [memoryReads, Option.getD_some, List.mem_append]
    exact Or.inl (Or.inr hm)
  · have hm : (st256Window low high index).source ∈ consecutive high 4 := by
      apply List.mem_map.mpr
      exact ⟨index % 4, List.mem_range.mpr hrem, by simp [st256Window, hlow]⟩
    simp only [memoryReads, Option.getD_some, List.mem_append]
    exact Or.inr hm

theorem st256_prefix_before_next (low high : Reg) (count : Nat) :
    beforeWindow (st256Prefix low high count) (st256Window low high count) := by
  intro window hm
  rcases List.mem_map.mp hm with ⟨index, hi, rfl⟩
  have hi : index < count := List.mem_range.mp hi
  change 4 * index + 4 ≤ 4 * count
  omega

theorem st256_prefix_take (low high : Reg) (count index : Nat) (hi : index ≤ count) :
    (st256Prefix low high count).take index = st256Prefix low high index := by
  unfold st256Prefix
  rw [← List.map_take, List.take_range, Nat.min_eq_left hi]

theorem st256_windows_valid (low high : Reg) (windows : List (Reg × U32 × U32))
    (hshape : windows.map byteWindow = st256Prefix low high 8) :
    payloadWindowsValid windows 32 := by
  intro index hi
  have hlength : windows.length = 8 := by
    have := congrArg List.length hshape
    simpa [st256Prefix] using this
  have hentry : byteWindow (windows[index]'hi) = st256Window low high index := by
    have hget := congrArg (fun entries : List ByteWindow => entries[index]?) hshape
    rw [getElem?_pos _ _ (by simpa using hi)] at hget
    rw [getElem?_pos _ _ (by simpa [st256Prefix] using (show index < 8 by omega))] at hget
    simpa [st256Prefix] using Option.some.inj hget
  have hfirst : (windows[index]'hi).2.1.val = 4 * index := congrArg ByteWindow.first hentry
  have hcount : (windows[index]'hi).2.2.val = 4 := congrArg ByteWindow.count hentry
  refine ⟨by omega, ?_, ?_⟩
  · intro byte hb
    omega
  · have ht : (windows.take index).map byteWindow = st256Prefix low high index := by
      rw [List.map_take, hshape, st256_prefix_take low high 8 index (by omega)]
    rw [ht, hentry]
    exact st256_prefix_before_next low high index

theorem actual_st256_window_declared_read (operation : Op) (window : ByteWindow)
    (hwindow : window ∈ (st256Windows operation).getD []) :
    window.source ∈ (memoryReads operation).getD [] := by
  cases operation <;> try (simp [st256Windows] at hwindow)
  rename_i low high address sync
  exact st256_window_declared_read low high address sync window hwindow

theorem actual_st256_windows_valid (operation : Op) (windows : List (Reg × U32 × U32))
    (hshape : some (windows.map byteWindow) = st256Windows operation) :
    payloadWindowsValid windows 32 := by
  cases operation <;> try (simp [st256Windows] at hshape)
  rename_i low high address sync
  exact st256_windows_valid low high windows hshape

end Instruction
