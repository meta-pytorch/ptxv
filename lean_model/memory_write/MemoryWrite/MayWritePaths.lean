/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.MayWrite

structure Graph where
  enabled : Nat → Prop
  edge : Nat → Nat → Prop
  writes : Nat → Nat → Prop

inductive Reachable (graph : Graph) (start : Nat) : Nat → Prop
  | start : Reachable graph start start
  | next {pc successor : Nat} : Reachable graph start pc → graph.enabled pc →
      graph.edge pc successor → Reachable graph start successor

inductive WrittenBefore (graph : Graph) (start : Nat) : Nat → Nat → Prop
  | write {pc successor register : Nat} : Reachable graph start pc →
      graph.enabled pc → graph.edge pc successor → graph.writes pc register →
      WrittenBefore graph start successor register
  | carry {pc successor register : Nat} : WrittenBefore graph start pc register →
      graph.enabled pc → graph.edge pc successor → WrittenBefore graph start successor register

theorem WrittenBefore.reachable {graph : Graph} {start pc register : Nat}
    (written : WrittenBefore graph start pc register) : Reachable graph start pc := by
  induction written with
  | write reachable enabled edge _ => exact .next reachable enabled edge
  | carry _ enabled edge ih => exact .next ih enabled edge

structure Solution where
  present : Nat → Prop
  incoming : Nat → Nat → Prop

def Closed (graph : Graph) (solution : Solution) : Prop :=
  ∀ pc, solution.present pc → graph.enabled pc → ∀ successor,
    graph.edge pc successor → solution.present successor ∧
    ∀ register, solution.incoming pc register ∨ graph.writes pc register →
      solution.incoming successor register

theorem reachable_present (graph : Graph) (start : Nat) (solution : Solution)
    (initial : solution.present start) (closed : Closed graph solution)
    {pc : Nat} (reachable : Reachable graph start pc) : solution.present pc := by
  induction reachable with
  | start => exact initial
  | next reachable enabled edge ih => exact (closed _ ih enabled _ edge).1

theorem closed_covers_path_writes (graph : Graph) (start : Nat) (solution : Solution)
    (initial : solution.present start) (closed : Closed graph solution)
    {pc register : Nat} (written : WrittenBefore graph start pc register) :
    solution.incoming pc register := by
  induction written with
  | write reachable enabled edge write =>
      exact (closed _ (reachable_present graph start solution initial closed reachable)
        enabled _ edge).2 _ (Or.inr write)
  | carry written enabled edge ih =>
      have present := reachable_present graph start solution initial closed written.reachable
      exact (closed _ present enabled _ edge).2 _ (Or.inl ih)

def ExportedMayWrite (solution : Solution) (pc register : Nat) : Prop :=
  ¬solution.present pc ∨ solution.incoming pc register

theorem closed_export_covers_path_writes (graph : Graph) (start : Nat) (solution : Solution)
    (initial : solution.present start) (closed : Closed graph solution)
    {pc register : Nat} (written : WrittenBefore graph start pc register) :
    ExportedMayWrite solution pc register :=
  Or.inr (closed_covers_path_writes graph start solution initial closed written)

def graphView (graph : cache_graph.MayWriteGraph) (bit : U32) : Graph where
  enabled pc := ∃ flag, graph.reach.val[pc]? = some flag ∧ flag &&& bit ≠ 0#u32
  edge pc successor := ∃ row, graph.succ.val[pc]? = some row ∧
    ∃ address ∈ row.val, address.val = successor
  writes pc register := ∃ row, graph.writes.val[pc]? = some row ∧
    ∃ address ∈ row.val, address.val = register

def WordContains (words : List U64) (register : Nat) : Prop :=
  ∃ word, words[register / 64]? = some word ∧ word.bv[register % 64] = true

def resultView (sets : cache_graph.MayWriteSets) : Solution where
  present pc := ∃ row ∈ sets.rows.val, row.pc.val = pc
  incoming pc register := ∃ row ∈ sets.rows.val, row.pc.val = pc ∧
    WordContains row.words.val register

end MemoryWrite.MayWrite
