/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexBuildGraph
import MemoryWrite.MayWriteQuery

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

inductive RegisterWrittenBefore (graph : cache_graph.MayWriteGraph)
    (destinations : List (alloc.vec.Vec symbolic.RegisterKey)) (bit : U32) (start : Nat) :
    Nat → symbolic.RegisterKey → Prop
  | write {pc successor : Nat} {key : symbolic.RegisterKey} :
      Reachable (graphView graph bit) start pc → (graphView graph bit).enabled pc →
      (graphView graph bit).edge pc successor →
      (∃ row, destinations[pc]? = some row ∧ key ∈ row.val) →
      RegisterWrittenBefore graph destinations bit start successor key
  | carry {pc successor : Nat} {key : symbolic.RegisterKey} :
      RegisterWrittenBefore graph destinations bit start pc key →
      (graphView graph bit).enabled pc → (graphView graph bit).edge pc successor →
      RegisterWrittenBefore graph destinations bit start successor key

theorem successful_built_lookup (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey))
    (index : register_index.RegisterIndex) (writes : alloc.vec.Vec (alloc.vec.Vec Usize))
    (built : register_index.build_writes destinations = .ok (index, writes))
    (pc position : Nat) (pcBound : pc < destinations.val.length)
    (positionBound : position < destinations.val[pc]!.val.length)
    (answer : Option Usize)
    (queried : register_index.RegisterIndex.get index destinations.val[pc]!.val[position]! = .ok answer) :
    answer = some writes.val[pc]!.val[position]! := by
  have builtSpec := successful_build_writes destinations index writes built
  have represented := builtSpec.2.2 pc pcBound
  have stored := represented.2 position positionBound
  have querySpec := rust_get index destinations.val[pc]!.val[position]! builtSpec.1.1
  simp only [queried, WP.spec_ok] at querySpec
  exact (querySpec _).mpr stored

theorem register_path_maps_to_dense (graph : cache_graph.MayWriteGraph)
    (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey)) (index : register_index.RegisterIndex)
    (writes : alloc.vec.Vec (alloc.vec.Vec Usize))
    (built : register_index.build_writes destinations = .ok (index, writes)) (graphWrites : graph.writes = writes)
    (start pc : Nat) (key : symbolic.RegisterKey) (bit : U32)
    (written : RegisterWrittenBefore graph destinations.val bit start pc key)
    (answer : Option Usize) (queried : register_index.RegisterIndex.get index key = .ok answer) :
    ∃ id, answer = some id ∧ WrittenBefore (graphView graph bit) start pc id.val := by
  have builtSpec := successful_build_writes destinations index writes built
  have querySpec := rust_get index key builtSpec.1.1
  simp only [queried, WP.spec_ok] at querySpec
  induction written with
  | @write source successor key reachable enabled edge localWrite =>
      obtain ⟨registers, atSource, member⟩ := localWrite
      have sourceBound : source < destinations.val.length := (List.getElem?_eq_some_iff.mp atSource).1
      have registersAt : registers = destinations.val[source]! := by
        have observed := (List.getElem?_eq_some_iff.mp atSource).choose_spec
        rw [List.Inhabited_getElem_eq_getElem! _ _ sourceBound] at observed
        exact observed.symm
      obtain ⟨position, positionBound, keyAt⟩ := List.getElem_of_mem member
      have represented := builtSpec.2.2 source sourceBound
      have destinationKey : destinations.val[source]!.val[position]! = key := by
        rw [←registersAt, ←List.Inhabited_getElem_eq_getElem! _ _ positionBound, keyAt]
      let id := writes.val[source]!.val[position]!
      have stored : Stored index key id := by
        simpa only [destinationKey] using represented.2 position (by simpa only [registersAt] using positionBound)
      refine ⟨id, (querySpec id).mpr stored, .write reachable enabled edge ?_⟩
      refine ⟨writes.val[source]!, ?_, id, ?_, rfl⟩
      · rw [graphWrites, List.getElem?_eq_getElem (by have := builtSpec.2.1; omega),
          List.Inhabited_getElem_eq_getElem! _ _ (by have := builtSpec.2.1; omega)]
      · have inside : position < writes.val[source]!.val.length := by
          have length := represented.1
          rw [registersAt] at positionBound
          omega
        rw [show id = writes.val[source]!.val[position]! from rfl,
          ←List.Inhabited_getElem_eq_getElem! _ _ inside]
        exact List.getElem_mem inside
  | carry written enabled edge ih =>
      obtain ⟨id, answerEqual, path⟩ := ih queried querySpec
      exact ⟨id, answerEqual, .carry path enabled edge⟩

theorem built_register_query_catches_every_path_write (graph : cache_graph.MayWriteGraph)
    (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey)) (index : register_index.RegisterIndex)
    (writes : alloc.vec.Vec (alloc.vec.Vec Usize))
    (built : register_index.build_writes destinations = .ok (index, writes)) (graphWrites : graph.writes = writes)
    (start pc : Usize) (key : symbolic.RegisterKey) (bit : U32)
    (bounded : GraphBounded graph) (startBound : start.val < graph.reach.val.length)
    (sets : cache_graph.MayWriteSets) (analysis : cache_graph.may_write_sets graph start bit = .ok sets)
    (written : RegisterWrittenBefore graph destinations.val bit start.val pc.val key)
    (register : Option Usize) (lookup : register_index.RegisterIndex.get index key = .ok register)
    (answer : Bool) (queried : cache_graph.may_at sets pc register = .ok answer) : answer = true := by
  obtain ⟨id, registerEqual, densePath⟩ := register_path_maps_to_dense graph destinations index writes built
    graphWrites start.val pc.val key bit written register lookup
  rw [registerEqual] at queried
  exact may_write_sets_query_catches_every_path_write graph start pc id bit bounded startBound sets analysis densePath answer queried

end MemoryWrite.RegisterIndex
