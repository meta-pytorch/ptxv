/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.ControlFlowMemoryTotal

open Aeneas Aeneas.Std Result PtxvIr
open MemoryWrite.MayWrite MemoryWrite.RegisterIndex

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

namespace Instruction.ControlFlow

def TargetsBounded (operation : Op) (count : Nat) : Prop :=
  match operation with
  | .Bra target _ _ _ _ _ | .Call target => target.val < count
  | .Brx _ _ targets => ∀ target ∈ targets.val, target.val < count
  | _ => True

theorem next_bounded (context : Context) (destination : Nat) (member : destination ∈ next context) :
    destination < context.count := by
  unfold next at member
  split at member
  · rename_i inside
    simp only [List.mem_singleton] at member
    omega
  · simp only [List.not_mem_nil] at member

theorem return_targets_bounded (context : Context)
    (bounded : ∀ targets, context.returns = some targets → ∀ destination ∈ targets, destination < context.count)
    (destination : Nat) (member : destination ∈ returnTargets context) : destination < context.count := by
  unfold returnTargets at member
  cases found : context.returns with
  | none => simpa only [found, Option.getD_none, List.mem_range] using member
  | some targets =>
      simp only [found, Option.getD_some] at member
      exact bounded targets found destination member

theorem successors_bounded (operation : Op) (guard : Pred) (context : Context)
    (targets : TargetsBounded operation context.count)
    (returns : ∀ row, context.returns = some row → ∀ destination ∈ row, destination < context.count)
    (destination : Nat) (member : Admissible operation guard context destination) :
    destination < context.count := by
  have nextBound := next_bounded context destination
  have returnBound := return_targets_bounded context returns destination
  cases operation <;> simp only [Admissible, successors, TargetsBounded] at member targets
  all_goals try exact nextBound member
  all_goals repeat' split at member
  all_goals try simp only [List.mem_append, List.mem_cons, List.not_mem_nil,
    List.mem_map] at member
  all_goals grind

theorem built_writes_bounded (destinations : Slice (alloc.vec.Vec ProgramKey))
    (index : PtxvMemoryWrite.register_index.RegisterIndex) (writes : alloc.vec.Vec (alloc.vec.Vec Usize))
    (built : PtxvMemoryWrite.register_index.build_writes destinations = .ok (index, writes)) :
    writes.val.length = destinations.val.length ∧
      ∀ pc < writes.val.length, ∀ register ∈ writes.val[pc]!.val, register.val < index.count.val := by
  have source := successful_build_writes destinations index writes built
  have writeLength := source.2.1
  refine ⟨writeLength, ?_⟩
  intro pc inside register member
  obtain ⟨position, positionBound, registerAt⟩ := List.getElem_of_mem member
  have row := source.2.2 pc (by omega)
  have rowLength := row.1
  have destinationBound : position < destinations.val[pc]!.val.length := by omega
  have stored := row.2 position destinationBound
  have bound := source.1.2.1 _ _ stored
  simpa only [←List.Inhabited_getElem_eq_getElem! _ _ positionBound, registerAt] using bound

theorem row_context_returns_bounded (count : Nat) (cut : U8) (fold : List (Option Bool))
    (returns : List (alloc.vec.Vec Usize))
    (bounded : ∀ (pc : Nat) (row : alloc.vec.Vec Usize), returns[pc]? = some row → ∀ destination ∈ row.val, destination.val < count)
    (pc : Nat) (row : List Nat)
    (found : (rowContext count cut fold returns pc).returns = some row) :
    ∀ destination ∈ row, destination < count := by
  change (returns[pc]?.map (fun values => values.val.map UScalar.val)) = some row at found
  cases actual : returns[pc]? with
  | none => simp only [actual, Option.map_none, reduceCtorEq] at found
  | some values =>
      simp only [actual, Option.map_some, Option.some.injEq] at found
      intro destination member
      rw [←found] at member
      obtain ⟨raw, rawMember, rawValue⟩ := List.mem_map.mp member
      rw [←rawValue]
      exact bounded pc values actual raw rawMember

theorem actual_graph_is_bounded
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (rowSpace : ∀ pc < code.val.length, (graphRows code.val cut fold.val returns.val)[pc]!.length ≤ Usize.max)
    (cfgProduced : control_flow.graph_successors code cut fold returns = .ok graph.succ)
    (targets : ∀ (pc : Nat) (instruction : Inst), code.val[pc]? = some instruction → TargetsBounded instruction.op code.val.length)
    (returnTargets : ∀ (pc : Nat) (row : alloc.vec.Vec Usize), returns.val[pc]? = some row →
      ∀ destination ∈ row.val, destination.val < code.val.length)
    (reachLength : graph.reach.val.length = code.val.length)
    (destinations : Slice (alloc.vec.Vec ProgramKey))
    (destinationLength : destinations.val.length = code.val.length)
    (index : PtxvMemoryWrite.register_index.RegisterIndex)
    (writes : alloc.vec.Vec (alloc.vec.Vec Usize))
    (built : PtxvMemoryWrite.register_index.build_writes destinations = .ok (index, writes))
    (graphWrites : graph.writes = writes) (graphCount : graph.register_count = index.count) :
    GraphBounded graph := by
  have produced := rust_graph_successors code cut fold returns rowSpace
  simp only [cfgProduced, WP.spec_ok] at produced
  have successorLength := congrArg List.length produced
  simp only [List.length_map, graph_rows_length] at successorLength
  have writeBounds := built_writes_bounded destinations index writes built
  refine ⟨successorLength.trans reachLength.symm, ?_, ?_, ?_⟩
  · rw [graphWrites, writeBounds.1, destinationLength, reachLength]
  · intro pc inside destination member
    have actualEdge : (graphView graph 0#u32).edge pc destination.val := by
      refine ⟨graph.succ.val[pc]!, ?_, destination, member, rfl⟩
      have rowBound : pc < graph.succ.val.length := by omega
      rw [List.getElem?_eq_getElem rowBound, List.Inhabited_getElem_eq_getElem! _ _ rowBound]
    obtain ⟨instruction, instructionAt, admissible⟩ :=
      (actual_cfg_edges graph code cut fold returns rowSpace cfgProduced 0#u32 pc destination.val).mp actualEdge
    rw [reachLength]
    exact successors_bounded instruction.op instruction.guard
      (rowContext code.val.length cut fold.val returns.val pc)
      (targets pc instruction instructionAt)
      (row_context_returns_bounded code.val.length cut fold.val returns.val returnTargets pc)
      destination.val admissible
  · intro pc inside register member
    rw [graphWrites] at member
    rw [graphCount]
    exact writeBounds.2 pc (by omega) register member

end Instruction.ControlFlow
