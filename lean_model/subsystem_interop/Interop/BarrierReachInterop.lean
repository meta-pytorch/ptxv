/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.BarrierSeedRows
import Interop.ControlFlowPublication
import MemoryWrite.BarrierReachPaths

open Aeneas Aeneas.Std Result PtxvIr
open MemoryWrite.MayWrite MemoryWrite.BarrierReach

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.ControlFlow

def BarrierLabel (code : List Inst) (pc : Nat) (bit : Bit) : Prop :=
  ∃ instruction, code[pc]? = some instruction ∧ BarrierSeed.Labels instruction.op bit

theorem actual_labels_are_seeded (code : List Inst) (flags : List U32)
    (rows : BarrierSeed.Rows code flags) (pc : Nat) (bit : Bit) (label : BarrierLabel code pc bit) :
    HasBitAt flags pc bit := by
  obtain ⟨instruction, instructionAt, labeled⟩ := label
  have pcBound := (List.getElem?_eq_some_iff.mp instructionAt).1
  have instructionEq : code[pc] = instruction := by
    exact Option.some.inj ((List.getElem?_eq_getElem pcBound).symm.trans instructionAt)
  have source := rows.at pc pcBound bit
  rw [instructionEq] at source
  have flagBound : pc < flags.length := by have := rows.length; omega
  apply (bit_at_row flags pc flagBound bit).mpr
  exact source.mpr labeled

def reachFromSource (code : Slice Inst) (successors : Slice (alloc.vec.Vec Usize)) :
    Result (alloc.vec.Vec U32) :=
  Std.bind (control_flow.barrier_seeds code) (fun seeds =>
    PtxvMemoryWrite.barrier_reach.propagate successors seeds)

def ReachSourcePost (code : List Inst) (successors : List (alloc.vec.Vec Usize)) (reach : List U32) : Prop :=
  reach.length = code.length ∧ MemoryWrite.BarrierReach.Closed successors reach ∧
  ∀ bit pc, CanReach (Edge successors) (fun address => BarrierLabel code address bit) pc → HasBitAt reach pc bit

@[step]
theorem rust_reach_from_source (code : Slice Inst) (successors : Slice (alloc.vec.Vec Usize))
    (bounded : EdgesBounded successors.val code.val.length) :
    reachFromSource code successors ⦃ reach => ReachSourcePost code.val successors.val reach.val ⦄ := by
  unfold reachFromSource
  step with BarrierSeed.rust_barrier_seeds code as ⟨seeds, rows⟩
  have seedLength : seeds.val.length = code.val.length := rows.length.symm
  step with rust_propagate_conservative successors seeds
    (by simpa only [seedLength] using bounded) as ⟨reach, propagated, covers⟩
  refine ⟨propagated.1.trans seedLength, propagated.2.2, ?_⟩
  intro bit pc path
  apply covers bit pc
  exact path.map (fun _ _ edge => edge)
    (fun address label => actual_labels_are_seeded code.val seeds.val rows address bit label)

def rawCfgView (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph) (code : List Inst) (cut : U8)
    (fold : List (Option Bool)) (returns : List (alloc.vec.Vec Usize)) (mask : U32) : Graph :=
  { cfgView graph code cut fold returns mask with enabled := fun _ => True }

def ToBarrier (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph) (code : List Inst) (cut : U8)
    (fold : List (Option Bool)) (returns : List (alloc.vec.Vec Usize)) (mask : U32) (bit : Bit) (pc : Nat) : Prop :=
  CanReach (cfgView graph code cut fold returns mask).edge (fun address => BarrierLabel code address bit) pc

theorem published_edges_bounded (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (publication : ProgramPublication graph code cut fold returns) :
    EdgesBounded graph.succ.val code.val.length := by
  have bounded := published_program_bounded graph code cut fold returns publication
  refine ⟨bounded.1.trans publication.reachLength, ?_⟩
  intro pc inside destination member
  rw [←publication.reachLength]
  exact bounded.2.2.1 pc (by rw [publication.reachLength]; exact inside) destination member

theorem actual_reach_enables_every_path_to_barrier
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (publication : ProgramPublication graph code cut fold returns)
    (reachProduced : reachFromSource code (alloc.vec.Vec.deref graph.succ) = .ok graph.reach)
    (mask : U32) (bit : Bit) (requested : HasBit mask bit) (pc : Nat)
    (path : ToBarrier graph code.val cut fold.val returns.val mask bit pc) :
    (cfgView graph code.val cut fold.val returns.val mask).enabled pc := by
  have bounded := published_edges_bounded graph code cut fold returns publication
  have source := rust_reach_from_source code (alloc.vec.Vec.deref graph.succ)
    (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using bounded)
  simp only [reachProduced, WP.spec_ok] at source
  have storage := graph_rows_fit_storage code.val cut fold.val returns.val
    publication.codeStorage publication.targetStorage publication.returnStorage
  have actualPath : CanReach (Edge graph.succ.val) (fun address => BarrierLabel code.val address bit) pc := by
    apply path.map
    · intro address destination edge
      exact (actual_cfg_edges graph code cut fold returns storage publication.cfgProduced mask address destination).mpr edge
    · exact fun _ label => label
  obtain ⟨word, wordAt, present⟩ := source.2.2 bit pc (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using actualPath)
  exact ⟨word, wordAt, shared_bit_intersection_nonzero word mask bit present requested⟩

end Instruction.ControlFlow
