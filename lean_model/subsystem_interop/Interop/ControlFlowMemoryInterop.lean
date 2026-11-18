/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ControlFlowGraphRefinement
import Interop.MemoryGraphInterop

open Aeneas Aeneas.Std Result PtxvIr
open MemoryWrite.MayWrite MemoryWrite.RegisterIndex

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.ControlFlow

def cfgView (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph) (code : List Inst) (cut : U8)
    (fold : List (Option Bool)) (returns : List (alloc.vec.Vec Usize)) (bit : U32) : Graph where
  enabled := (graphView graph bit).enabled
  edge := fun pc destination => ∃ instruction, code[pc]? = some instruction ∧
    Admissible instruction.op instruction.guard (rowContext code.length cut fold returns pc) destination
  writes := fun _ _ => False

theorem actual_cfg_edges (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (space : ∀ pc < code.val.length, (graphRows code.val cut fold.val returns.val)[pc]!.length ≤ Usize.max)
    (produced : control_flow.graph_successors code cut fold returns = .ok graph.succ)
    (bit : U32) (pc destination : Nat) :
    (graphView graph bit).edge pc destination ↔
      (cfgView graph code.val cut fold.val returns.val bit).edge pc destination := by
  have source := rust_graph_successors code cut fold returns space
  simp only [produced, WP.spec_ok] at source
  have atPc := congrArg (fun rows => rows[pc]?) source
  simp only [List.getElem?_map, graphRows, List.getElem?_mapIdx] at atPc
  constructor
  · rintro ⟨row, rowAt, target, targetMember, targetValue⟩
    rw [rowAt, Option.map_some] at atPc
    cases instructionAt : code.val[pc]? with
    | none => simp only [instructionAt, Option.map_none, reduceCtorEq] at atPc
    | some instruction =>
        simp only [instructionAt, Option.map_some, Option.some.injEq] at atPc
        refine ⟨instruction, instructionAt, ?_⟩
        change destination ∈ successors instruction.op instruction.guard
          (rowContext code.val.length cut fold.val returns.val pc)
        rw [←atPc]
        exact List.mem_map.mpr ⟨target, targetMember, targetValue⟩
  · rintro ⟨instruction, instructionAt, member⟩
    rw [instructionAt, Option.map_some] at atPc
    cases rowAt : graph.succ.val[pc]? with
    | none => simp only [rowAt, Option.map_none, reduceCtorEq] at atPc
    | some row =>
        simp only [rowAt, Option.map_some, Option.some.injEq] at atPc
        change destination ∈ successors instruction.op instruction.guard
          (rowContext code.val.length cut fold.val returns.val pc) at member
        rw [←atPc] at member
        obtain ⟨target, targetMember, targetValue⟩ := List.mem_map.mp member
        exact ⟨row, rowAt, target, targetMember, targetValue⟩

theorem actual_cfg_reachable (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (space : ∀ pc < code.val.length, (graphRows code.val cut fold.val returns.val)[pc]!.length ≤ Usize.max)
    (produced : control_flow.graph_successors code cut fold returns = .ok graph.succ)
    (bit : U32) (start pc : Nat)
    (path : Reachable (cfgView graph code.val cut fold.val returns.val bit) start pc) :
    Reachable (graphView graph bit) start pc := by
  induction path with
  | start => exact .start
  | next path enabled edge ih =>
      exact .next ih enabled ((actual_cfg_edges graph code cut fold returns space produced bit _ _).mpr edge)

inductive CfgMemoryWrittenBefore (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : List Inst) (cut : U8) (fold : List (Option Bool)) (returns : List (alloc.vec.Vec Usize))
    (bit : U32) (start : Nat) : Nat → Reg → Prop
  | write {pc successor : Nat} {register : Reg} :
      Reachable (cfgView graph code cut fold returns bit) start pc →
      (cfgView graph code cut fold returns bit).enabled pc →
      (cfgView graph code cut fold returns bit).edge pc successor →
      (∃ instruction, code[pc]? = some instruction ∧
        ∃ keys, memoryWrites instruction.op = some keys ∧ key register ∈ keys) →
      CfgMemoryWrittenBefore graph code cut fold returns bit start successor register
  | carry {pc successor : Nat} {register : Reg} :
      CfgMemoryWrittenBefore graph code cut fold returns bit start pc register →
      (cfgView graph code cut fold returns bit).enabled pc →
      (cfgView graph code cut fold returns bit).edge pc successor →
      CfgMemoryWrittenBefore graph code cut fold returns bit start successor register

theorem actual_cfg_memory_path (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (space : ∀ pc < code.val.length, (graphRows code.val cut fold.val returns.val)[pc]!.length ≤ Usize.max)
    (produced : control_flow.graph_successors code cut fold returns = .ok graph.succ)
    (bit : U32) (start pc : Nat) (register : Reg)
    (path : CfgMemoryWrittenBefore graph code.val cut fold.val returns.val bit start pc register) :
    MemoryOpWrittenBefore graph (code.val.map Inst.op) bit start pc register := by
  induction path with
  | @write sourcePc successor register reachable enabled edge written =>
      obtain ⟨instruction, instructionAt, keys, writes, member⟩ := written
      refine .write (actual_cfg_reachable graph code cut fold returns space produced bit start sourcePc reachable)
        enabled ((actual_cfg_edges graph code cut fold returns space produced bit sourcePc successor).mpr edge) ?_
      exact ⟨instruction.op, by simp only [List.getElem?_map, instructionAt, Option.map_some], keys, writes, member⟩
  | carry path enabled edge ih =>
      exact .carry ih enabled ((actual_cfg_edges graph code cut fold returns space produced bit _ _).mpr edge)


theorem actual_cfg_memory_query_catches_every_path_write
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (rowSpace : ∀ pc < code.val.length, (graphRows code.val cut fold.val returns.val)[pc]!.length ≤ Usize.max)
    (cfgProduced : control_flow.graph_successors code cut fold returns = .ok graph.succ)
    (destinations : Slice (alloc.vec.Vec ProgramKey))
    (source : SourceMemoryRows (code.val.map Inst.op) destinations.val)
    (boundedCode : ∀ operation ∈ code.val.map Inst.op, optionBounded (memoryWrites operation))
    (index : PtxvMemoryWrite.register_index.RegisterIndex)
    (writes : alloc.vec.Vec (alloc.vec.Vec Usize))
    (built : PtxvMemoryWrite.register_index.build_writes destinations = .ok (index, writes))
    (graphWrites : graph.writes = writes) (start pc : Usize) (register : Reg) (bit : U32)
    (boundedGraph : GraphBounded graph) (startBound : start.val < graph.reach.val.length)
    (sets : PtxvMemoryWrite.cache_graph.MayWriteSets)
    (analysis : PtxvMemoryWrite.cache_graph.may_write_sets graph start bit = .ok sets)
    (written : CfgMemoryWrittenBefore graph code.val cut fold.val returns.val bit start.val pc.val register)
    (identity : ProgramKey) (identityProduced : programKeyFromSource register = .ok identity)
    (dense : Option Usize) (lookup : PtxvMemoryWrite.register_index.RegisterIndex.get index identity = .ok dense)
    (answer : Bool) (query : PtxvMemoryWrite.cache_graph.may_at sets pc dense = .ok answer) :
    answer = true := by
  exact actual_memory_op_query_catches_every_path_write graph (code.val.map Inst.op) destinations source
    boundedCode index writes built graphWrites start pc register bit boundedGraph startBound sets analysis
    (actual_cfg_memory_path graph code cut fold returns rowSpace cfgProduced bit start.val pc.val register written)
    identity identityProduced dense lookup answer query

end Instruction.ControlFlow
