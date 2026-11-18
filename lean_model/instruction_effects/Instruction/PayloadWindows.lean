/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.PayloadModel

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

def beforeWindow (previousWindows : List ByteWindow) (next : ByteWindow) : Prop :=
  ∀ previous ∈ previousWindows, previous.first + previous.count ≤ next.first

theorem payload_byte_append (previousWindows : List ByteWindow) (next : ByteWindow)
    (read : Key → Nat) (index : Nat) (hbefore : beforeWindow previousWindows next) :
    payloadByte (previousWindows ++ [next]) read index =
      if next.first ≤ index ∧ index < next.first + next.count then
        wordByte (read next.source) (index - next.first)
      else payloadByte previousWindows read index := by
  unfold payloadByte
  rw [List.find?_append]
  cases hfound : previousWindows.find? (fun window =>
      decide (window.first ≤ index ∧ index < window.first + window.count)) with
  | none =>
    by_cases hn : next.first ≤ index ∧ index < next.first + next.count
    all_goals simp [hn, wordByte]
  | some previous =>
    have hm : previous ∈ previousWindows := List.mem_of_find?_eq_some hfound
    have hpbool : decide (previous.first ≤ index ∧ index < previous.first + previous.count) = true :=
      List.find?_some (p := fun window : ByteWindow =>
        decide (window.first ≤ index ∧ index < window.first + window.count)) hfound
    have hp : previous.first ≤ index ∧ index < previous.first + previous.count :=
      of_decide_eq_true hpbool
    have hb := hbefore previous hm
    have hn : ¬(next.first ≤ index ∧ index < next.first + next.count) := by omega
    simp [hn, wordByte]

theorem payload_bytes_append (total : Nat) (previousWindows : List ByteWindow) (next : ByteWindow)
    (read : Key → Nat) (hbefore : beforeWindow previousWindows next) :
    payloadBytes total (previousWindows ++ [next]) read =
      overwriteBytes (payloadBytes total previousWindows read) next.first next.count (read next.source) := by
  apply List.ext_getElem
  · simp [payloadBytes]
  · intro index hi hj
    simp only [payloadBytes, overwriteBytes, List.getElem_map, List.getElem_mapIdx,
      List.getElem_range]
    exact payload_byte_append previousWindows next read index hbefore

theorem source_windows_take (first : Reg) (count stride total index : Nat) (hi : index ≤ count) :
    (sourceWindows first count stride total).take index = sourceWindows first index stride total := by
  unfold sourceWindows
  rw [← List.map_take, List.take_range, Nat.min_eq_left hi]

theorem source_windows_before_next (first : Reg) (count stride total : Nat) :
    beforeWindow (sourceWindows first count stride total)
      ⟨⟨first.file, first.idx.val + count⟩, count * stride, min (total - count * stride) stride⟩ := by
  intro previous hm
  rcases List.mem_map.mp hm with ⟨index, hi, rfl⟩
  have hi : index < count := List.mem_range.mp hi
  change index * stride + min (total - index * stride) stride ≤ count * stride
  calc
    _ ≤ index * stride + stride := Nat.add_le_add_left (Nat.min_le_right _ _) _
    _ = (index + 1) * stride := by rw [Nat.add_mul, Nat.one_mul]
    _ ≤ count * stride := Nat.mul_le_mul_right stride (by omega)

theorem source_windows_ordered (first : Reg) (count stride total index : Nat) (hi : index < count) :
    beforeWindow ((sourceWindows first count stride total).take index)
      ((sourceWindows first count stride total)[index]'(by simpa using hi)) := by
  rw [source_windows_take first count stride total index (by omega)]
  simpa [sourceWindows] using source_windows_before_next first index stride total

theorem source_window_bounds (first : Reg) (count stride total : Nat) (window : ByteWindow)
    (hm : window ∈ sourceWindows first count stride total) :
    window.count ≤ stride ∧ ∀ byte < window.count, window.first + byte < total := by
  rcases List.mem_map.mp hm with ⟨index, hi, rfl⟩
  constructor
  · exact Nat.min_le_right _ _
  · intro byte hb
    change index * stride + byte < total
    have hcount : byte < min (total - index * stride) stride := hb
    have := Nat.min_le_left (total - index * stride) stride
    omega

end Instruction
