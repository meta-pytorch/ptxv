/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AnalysisReadAdmission
import Instruction.ReadRefinement
import Instruction.AnalysisKnowledge

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction.AnalysisReadAdmission

def PrivateKeys (keys : List Key) : Prop :=
  ∀ observed ∈ keys, RegisterOwner.shared observed.file = false

theorem source_move_policy (destination : Reg) (operand : Val) :
    analysis_knowledge.direct_reads_are_thread_private (.Mov destination operand) =
      .ok (some (OwnerTemporal.valuePrivate operand)) := by
  simp [analysis_knowledge.direct_reads_are_thread_private,
    OwnerTemporal.source_value_thread_private]

theorem actual_move_shared_read_rejected (destination source : Reg) (slots : U8)
    (shared : RegisterOwner.shared source.file = true) :
    analysis_knowledge.direct_reads_are_thread_private
      (.Mov destination (.Reg source slots)) = .ok (some false) := by
  simp [source_move_policy, OwnerTemporal.valuePrivate, shared]

theorem source_constant_load_policy (destination : Reg) (bank : U8) (base : Val)
    (offset : U32) (bytes : U8) (data : alloc.vec.Vec U8) :
    analysis_knowledge.direct_reads_are_thread_private
      (.LdConst destination bank base offset bytes data) =
      .ok (some (OwnerTemporal.valuePrivate base)) := by
  simp [analysis_knowledge.direct_reads_are_thread_private,
    OwnerTemporal.source_value_thread_private]

theorem actual_constant_load_shared_read_rejected (destination source : Reg) (slots : U8)
    (bank : U8) (offset : U32) (bytes : U8) (data : alloc.vec.Vec U8)
    (shared : RegisterOwner.shared source.file = true) :
    analysis_knowledge.direct_reads_are_thread_private
      (.LdConst destination bank (.Reg source slots) offset bytes data) = .ok (some false) := by
  simp [source_constant_load_policy, OwnerTemporal.valuePrivate, shared]

theorem selected_memory_policy (operation : Op)
    (selected : memoryReads operation ≠ none) :
    analysis_knowledge.direct_reads_are_thread_private operation = .ok none := by
  cases operation <;> simp_all [memoryReads,
    analysis_knowledge.direct_reads_are_thread_private]

theorem private_observation (registers : List Reg) :
    PrivateKeys (registers.map key) ↔ PrivateReads registers := by
  constructor
  · intro observed register member
    exact observed (key register) (List.mem_map.mpr ⟨register, member, rfl⟩)
  · intro reads observed member
    obtain ⟨register, inRegs, same⟩ := List.mem_map.mp member
    subst observed
    exact reads register inRegs

def admitMemoryReads (operation : Op) : Result (Option Bool) := do
  let registers ← register_effect.memory_reads operation
  match registers with
  | none => ok none
  | some registers =>
    let admitted ← analysis_knowledge.reads_are_thread_private (alloc.vec.Vec.deref registers)
    ok (some admitted)

@[step]
theorem actual_memory_read_admission (operation : Op)
    (bounded : optionBounded (memoryReads operation)) :
    admitMemoryReads operation ⦃ result =>
      match result with
      | none => memoryReads operation = none
      | some admitted => ∃ keys, memoryReads operation = some keys ∧
          (admitted = true ↔ PrivateKeys keys) ⦄ := by
  unfold admitMemoryReads
  step with memory_reads_spec operation bounded as ⟨registers, observed⟩
  cases registers with
  | none => simpa [registerObservation] using observed.symm
  | some registers =>
    step with rust_reads_are_thread_private (alloc.vec.Vec.deref registers)
      as ⟨admitted, meaning⟩
    refine ⟨registers.val.map key, ?_, ?_⟩
    · simpa [registerObservation] using observed.symm
    · have same : (alloc.vec.Vec.deref registers).val = registers.val := by
        simp [alloc.vec.Vec.deref]
      simpa only [same, private_observation] using meaning

theorem actual_shared_memory_read_is_rejected (operation : Op)
    (bounded : optionBounded (memoryReads operation)) (keys : List Key)
    (reads : memoryReads operation = some keys) (observed : Key)
    (member : observed ∈ keys) (shared : RegisterOwner.shared observed.file = true) :
    admitMemoryReads operation ≠ .ok (some true) := by
  intro accepted
  have source := actual_memory_read_admission operation bounded
  simp only [accepted, WP.spec_ok] at source
  obtain ⟨actual, atActual, allPrivate⟩ := source
  have same : actual = keys := Option.some.inj (atActual.symm.trans reads)
  subst actual
  have contradiction := allPrivate.mp trivial observed member
  simp [shared] at contradiction

end Instruction.AnalysisReadAdmission
