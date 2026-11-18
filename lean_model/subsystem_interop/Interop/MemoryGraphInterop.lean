/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.RegisterIndexInterop

open Aeneas Aeneas.Std Result PtxvIr
open MemoryWrite.MayWrite MemoryWrite.RegisterIndex

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace Instruction

theorem specification_returns {α : Type} (computation : Result α) (post : α → Prop)
    (specification : WP.spec computation post) : ∃ value, computation = .ok value ∧ post value := by
  cases computation with
  | ret value => exact ⟨value, rfl, (WP.spec_ok value).mp specification⟩
  | div => simp only [WP.spec_div] at specification
  | vis effect next => simp only [WP.spec_vis] at specification

theorem instruction_key_injective (left right : Reg) (equal : key left = key right) : left = right := by
  have fileEqual : left.file = right.file := congrArg Key.file equal
  have indexEqual : left.idx = right.idx := UScalar.eq_of_val_eq (congrArg Key.index equal)
  cases left
  cases right
  simp only [Reg.mk.injEq]
  exact ⟨fileEqual, indexEqual⟩

def SourceMemoryRows (code : List Op) (destinations : List (alloc.vec.Vec ProgramKey)) : Prop :=
  ∀ (pc : Nat) (operation : Op), code[pc]? = some operation → ∀ (registers : alloc.vec.Vec Reg),
    register_effect.memory_writes operation = .ok (some registers) →
    ∃ row : alloc.vec.Vec ProgramKey, destinations[pc]? = some row ∧ row.val = registers.val.map programKey

inductive MemoryOpWrittenBefore (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : List Op) (bit : U32) (start : Nat) : Nat → Reg → Prop
  | write {pc successor : Nat} {register : Reg} :
      Reachable (graphView graph bit) start pc → (graphView graph bit).enabled pc →
      (graphView graph bit).edge pc successor →
      (∃ operation, code[pc]? = some operation ∧
        ∃ keys, memoryWrites operation = some keys ∧ key register ∈ keys) →
      MemoryOpWrittenBefore graph code bit start successor register
  | carry {pc successor : Nat} {register : Reg} :
      MemoryOpWrittenBefore graph code bit start pc register →
      (graphView graph bit).enabled pc → (graphView graph bit).edge pc successor →
      MemoryOpWrittenBefore graph code bit start successor register

theorem memory_op_path_uses_actual_program_keys (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : List Op) (destinations : List (alloc.vec.Vec ProgramKey))
    (source : SourceMemoryRows code destinations)
    (bounded : ∀ operation ∈ code, optionBounded (memoryWrites operation))
    (bit : U32) (start pc : Nat) (register : Reg)
    (written : MemoryOpWrittenBefore graph code bit start pc register) :
    RegisterWrittenBefore graph destinations bit start pc (programKey register) := by
  induction written with
  | @write sourcePc successor register reachable enabled edge independent =>
      obtain ⟨operation, operationAt, keys, meaning, member⟩ := independent
      have operationMember : operation ∈ code := List.mem_of_getElem? operationAt
      have specification := memory_writes_spec operation (bounded operation operationMember)
      obtain ⟨returned, returnedOk, returnedMeaning⟩ := specification_returns _ _ specification
      cases returned with
      | none =>
          simp only [registerObservation, meaning, Option.map_none, reduceCtorEq] at returnedMeaning
      | some registers =>
          have observations : registers.val.map key = keys := by
            simpa only [registerObservation, meaning, Option.map_some, Option.some.injEq] using returnedMeaning
          have keyMember : key register ∈ registers.val.map key := by rw [observations]; exact member
          obtain ⟨actual, actualMember, sameKey⟩ := List.mem_map.mp keyMember
          have actualEqual := instruction_key_injective actual register sameKey
          obtain ⟨row, rowAt, rowValues⟩ := source sourcePc operation operationAt registers returnedOk
          refine .write reachable enabled edge ⟨row, rowAt, ?_⟩
          rw [rowValues]
          exact List.mem_map.mpr ⟨actual, actualMember, congrArg programKey actualEqual⟩
  | carry written enabled edge ih => exact .carry ih enabled edge

theorem actual_memory_op_query_catches_every_path_write
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : List Op) (destinations : Slice (alloc.vec.Vec ProgramKey))
    (source : SourceMemoryRows code destinations.val)
    (boundedCode : ∀ operation ∈ code, optionBounded (memoryWrites operation))
    (index : PtxvMemoryWrite.register_index.RegisterIndex)
    (writes : alloc.vec.Vec (alloc.vec.Vec Usize))
    (built : PtxvMemoryWrite.register_index.build_writes destinations = .ok (index, writes))
    (graphWrites : graph.writes = writes) (start pc : Usize) (register : Reg) (bit : U32)
    (boundedGraph : GraphBounded graph) (startBound : start.val < graph.reach.val.length)
    (sets : PtxvMemoryWrite.cache_graph.MayWriteSets)
    (analysis : PtxvMemoryWrite.cache_graph.may_write_sets graph start bit = .ok sets)
    (written : MemoryOpWrittenBefore graph code bit start.val pc.val register)
    (identity : ProgramKey) (identityProduced : programKeyFromSource register = .ok identity)
    (dense : Option Usize) (lookup : PtxvMemoryWrite.register_index.RegisterIndex.get index identity = .ok dense)
    (answer : Bool) (query : PtxvMemoryWrite.cache_graph.may_at sets pc dense = .ok answer) :
    answer = true := by
  have identitySpec := rust_program_key register
  simp only [identityProduced, WP.spec_ok] at identitySpec
  have path := memory_op_path_uses_actual_program_keys graph code destinations.val source boundedCode
    bit start.val pc.val register written
  rw [identitySpec] at lookup
  exact built_register_query_catches_every_path_write graph destinations index writes built graphWrites
    start pc (programKey register) bit boundedGraph startBound sets analysis path dense lookup answer query

end Instruction
