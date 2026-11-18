/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteCompact
import MemoryWrite.MayWriteResources

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

structure RowsBefore (rows : List cache_graph.PcWrites) (limit : Nat) : Prop where
  strict : StrictKeys rows.length (fun position => rows[position]!.pc.val)
  bounded : ∀ row ∈ rows, row.pc.val < limit

theorem RowsBefore.mono {rows : List cache_graph.PcWrites} {before after : Nat}
    (ordered : RowsBefore rows before) (increase : before ≤ after) : RowsBefore rows after :=
  ⟨ordered.strict, fun row member => lt_of_lt_of_le (ordered.bounded row member) increase⟩

theorem RowsBefore.length_le {rows : List cache_graph.PcWrites} {limit : Nat}
    (ordered : RowsBefore rows limit) : rows.length ≤ limit := by
  by_cases nonempty : 0 < rows.length
  · have inside : rows.length - 1 < rows.length := by omega
    have lower := ordered.strict.index_le (rows.length - 1) inside
    have member : rows[rows.length - 1]! ∈ rows := by
      rw [←List.Inhabited_getElem_eq_getElem! _ _ inside]
      exact List.getElem_mem inside
    have upper := ordered.bounded _ member
    omega
  · omega

theorem RowsBefore.append {rows : List cache_graph.PcWrites} {limit : Nat}
    (ordered : RowsBefore rows limit) (row : cache_graph.PcWrites) (atLimit : row.pc.val = limit) :
    RowsBefore (rows ++ [row]) (limit + 1) := by
  constructor
  · intro left right increasing rightBound
    have leftBound : left < rows.length := by
      simp only [List.length_append, List.length_singleton] at rightBound
      omega
    have leftAt : (rows ++ [row])[left]! = rows[left]! := by
      rw [←List.Inhabited_getElem_eq_getElem! (rows ++ [row]) left (by simpa using Nat.lt_add_right 1 leftBound),
        List.getElem_append_left leftBound, List.Inhabited_getElem_eq_getElem! rows left leftBound]
    by_cases rightOld : right < rows.length
    · have rightAt : (rows ++ [row])[right]! = rows[right]! := by
        rw [←List.Inhabited_getElem_eq_getElem! (rows ++ [row]) right rightBound,
          List.getElem_append_left rightOld, List.Inhabited_getElem_eq_getElem! rows right rightOld]
      simpa only [leftAt, rightAt] using ordered.strict left right increasing rightOld
    · have rightEnd : right = rows.length := by
        simp only [List.length_append, List.length_singleton] at rightBound
        omega
      have rightAt : (rows ++ [row])[right]! = row := by
        subst right
        simp only [List.getElem!_eq_getElem?_getD, List.getElem?_append_right (Nat.le_refl _),
          Nat.sub_self, List.getElem?_cons_zero, Option.getD_some]
      have member : rows[left]! ∈ rows := by
        rw [←List.Inhabited_getElem_eq_getElem! rows left leftBound]
        exact List.getElem_mem leftBound
      simpa only [leftAt, rightAt, atLimit] using ordered.bounded _ member
  · intro observed member
    rcases List.mem_append.mp member with old | added
    · exact lt_of_lt_of_le (ordered.bounded observed old) (by omega)
    · have same := List.mem_singleton.mp added
      subst observed
      omega

end MemoryWrite.MayWrite
