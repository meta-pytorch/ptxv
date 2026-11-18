/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.BarrierReachInterop

open Aeneas Aeneas.Std Result PtxvIr
open MemoryWrite.MayWrite MemoryWrite.BarrierReach

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.ControlFlow

inductive RawMemoryWrittenBefore (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : List Inst) (cut : U8) (fold : List (Option Bool)) (returns : List (alloc.vec.Vec Usize))
    (mask : U32) (start : Nat) : Nat → Reg → Prop
  | write {pc successor : Nat} {register : Reg} :
      Reachable (rawCfgView graph code cut fold returns mask) start pc →
      (rawCfgView graph code cut fold returns mask).edge pc successor →
      (∃ instruction, code[pc]? = some instruction ∧
        ∃ keys, memoryWrites instruction.op = some keys ∧ key register ∈ keys) →
      RawMemoryWrittenBefore graph code cut fold returns mask start successor register
  | carry {pc successor : Nat} {register : Reg} :
      RawMemoryWrittenBefore graph code cut fold returns mask start pc register →
      (rawCfgView graph code cut fold returns mask).edge pc successor →
      RawMemoryWrittenBefore graph code cut fold returns mask start successor register

theorem raw_paths_to_barrier_are_enabled
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (publication : ProgramPublication graph code cut fold returns)
    (reachProduced : reachFromSource code (alloc.vec.Vec.deref graph.succ) = .ok graph.reach)
    (mask : U32) (bit : Bit) (requested : HasBit mask bit) (start pc : Nat)
    (path : Reachable (rawCfgView graph code.val cut fold.val returns.val mask) start pc) :
    ToBarrier graph code.val cut fold.val returns.val mask bit pc →
      Reachable (cfgView graph code.val cut fold.val returns.val mask) start pc := by
  induction path with
  | start => intro _; exact .start
  | @next previous current path _ edge ih =>
      intro gate
      have previousGate : ToBarrier graph code.val cut fold.val returns.val mask bit previous := .next edge gate
      exact .next (ih previousGate)
        (actual_reach_enables_every_path_to_barrier graph code cut fold returns publication reachProduced
          mask bit requested previous previousGate) edge

theorem raw_memory_paths_to_barrier_are_enabled
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (publication : ProgramPublication graph code cut fold returns)
    (reachProduced : reachFromSource code (alloc.vec.Vec.deref graph.succ) = .ok graph.reach)
    (mask : U32) (bit : Bit) (requested : HasBit mask bit) (start pc : Nat) (register : Reg)
    (written : RawMemoryWrittenBefore graph code.val cut fold.val returns.val mask start pc register) :
    ToBarrier graph code.val cut fold.val returns.val mask bit pc →
      CfgMemoryWrittenBefore graph code.val cut fold.val returns.val mask start pc register := by
  induction written with
  | @write source successor register reachable edge writes =>
      intro gate
      have sourceGate : ToBarrier graph code.val cut fold.val returns.val mask bit source := .next edge gate
      exact .write
        (raw_paths_to_barrier_are_enabled graph code cut fold returns publication reachProduced mask bit requested
          start source reachable sourceGate)
        (actual_reach_enables_every_path_to_barrier graph code cut fold returns publication reachProduced
          mask bit requested source sourceGate) edge writes
  | @carry source successor register written edge ih =>
      intro gate
      have sourceGate : ToBarrier graph code.val cut fold.val returns.val mask bit source := .next edge gate
      exact .carry (ih sourceGate)
        (actual_reach_enables_every_path_to_barrier graph code cut fold returns publication reachProduced
          mask bit requested source sourceGate) edge

theorem actual_barrier_memory_analysis_and_queries_return_true
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (publication : ProgramPublication graph code cut fold returns)
    (reachProduced : reachFromSource code (alloc.vec.Vec.deref graph.succ) = .ok graph.reach)
    (mask : U32) (bit : Bit) (requested : HasBit mask bit) (start : Usize)
    (startBound : start.val < code.val.length) (fits : AnalysisFits graph) :
    ∃ sets, PtxvMemoryWrite.cache_graph.may_write_sets graph start mask = .ok sets ∧
      ∀ (pc : Usize) (register : Reg),
        RawMemoryWrittenBefore graph code.val cut fold.val returns.val mask start.val pc.val register →
        ToBarrier graph code.val cut fold.val returns.val mask bit pc.val →
        Std.bind (programKeyFromSource register) (fun identity =>
          Std.bind (PtxvMemoryWrite.register_index.RegisterIndex.get publication.index identity)
            (fun dense => PtxvMemoryWrite.cache_graph.may_at sets pc dense)) = .ok true := by
  obtain ⟨sets, analyzed, queries⟩ := published_program_analysis_and_queries_return_true graph code cut fold returns
    publication start mask startBound fits
  refine ⟨sets, analyzed, ?_⟩
  intro pc register written gate
  exact queries pc register (raw_memory_paths_to_barrier_are_enabled graph code cut fold returns publication
    reachProduced mask bit requested start.val pc.val register written gate)

end Instruction.ControlFlow
