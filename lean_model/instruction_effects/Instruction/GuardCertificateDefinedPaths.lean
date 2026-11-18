/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateDefinedTransfer
import Instruction.GuardCertificatePaths

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 7000000
namespace Instruction.GuardCertificate

open Instruction.ControlFlow

inductive ExecutionDefined {C : Type} (certificate : guard_certificate.Certificate C)
    (bank : U32 → U8 → Option U64) : Nat → (PartialRegisters) → Prop
  | entry (registers : PartialRegisters) : ExecutionDefined certificate bank certificate.entry.val registers
  | peer {pc : Nat} {before after : PartialRegisters} :
      ExecutionDefined certificate bank pc before →
      (∀ register, RegisterOwner.shared register.file = false → after register = before register) →
      ExecutionDefined certificate bank pc after
  | own {pc target : Nat} {before after : PartialRegisters} {instruction : Inst} :
      ExecutionDefined certificate bank pc before →
      certificate.code.val[pc]? = some instruction →
      StepPostDefined certificate.front instruction bank before after →
      (truth : Bool) →
      (∀ observed, partialPredicate certificate.front before instruction.guard = some observed → truth = observed) →
      target ∈ semanticEdges instruction.op instruction.guard truth (nodeContext certificate pc) →
      ExecutionDefined certificate bank target after

def StateSoundDefined {C : Type} (certificate : guard_certificate.Certificate C)
    (pc : Nat) (registers : PartialRegisters) : Prop :=
  ∃ facts, certificate.states.val[pc]? = some (some facts) ∧
    FactsSoundDefined facts.val registers ∧ FactsPrivate facts.val

theorem certificate_execution_facts_defined {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (observations : alloc.vec.Vec (Option (Option Bool)))
    (bank : U32 → U8 → Option U64)
    (checked : CertificateChecked reader certificate observations)
    (rowsFit : RowsFit certificate) (framesFit : FramesFit certificate)
    (constantSound : ConstantsSound reader certificate.constants bank)
    (pc : Nat) (registers : PartialRegisters) (execution : ExecutionDefined certificate bank pc registers) :
    StateSoundDefined certificate pc registers := by
  induction execution with
  | entry registers =>
    obtain ⟨facts, stateAt, empty⟩ := checked.2.1
    refine ⟨facts, stateAt, ?_, ?_⟩ <;> simp [FactsSoundDefined, FactsPrivate, empty]
  | peer prior privateAgree ih =>
    obtain ⟨facts, stateAt, sound, privateFacts⟩ := ih
    refine ⟨facts, stateAt, ?_, privateFacts⟩
    intro fact member
    exact (privateAgree fact.register (privateFacts fact member)).trans (sound fact member)
  | @own pc target before after instruction prior instructionAt stepPost truth choice member ih =>
    obtain ⟨facts, stateAt, sound, privateFacts⟩ := ih
    obtain ⟨observation, observationAt, node⟩ := checked_node reader certificate observations checked pc facts stateAt
    rcases node with ⟨unreached, _⟩ | ⟨nodeFacts, nodeInstruction, sourcePC, known, targets,
      nodeStateAt, nodeInstructionAt, sourcePCAt, observed, decisionAt, successorsAt, targetsChecked⟩
    · rw [stateAt] at unreached
      cases unreached
    have sameFacts : nodeFacts = facts := by simpa [stateAt] using nodeStateAt.symm
    have sameInstruction : nodeInstruction = instruction := by simpa [instructionAt] using nodeInstructionAt.symm
    subst nodeFacts
    subst nodeInstruction
    have compatible : ∀ observed, known = some observed → observed = truth := by
      intro observed present
      have defined := source_decision_defined certificate.front (alloc.vec.Vec.deref facts) before
        (by simpa [alloc.vec.Vec.deref, Slice.from_val] using sound) instruction.guard observed
        (by simpa only [present] using decisionAt)
      exact (choice observed defined).symm
    have checkedMember := checked_edges_cover_truth instruction.op instruction.guard truth known
      (nodeContext certificate pc) target (by
        intro structural
        have parts : instruction.guard.reg = none ∧ instruction.guard.neg = false := by
          simpa [always] using structural
        have defined : partialPredicate certificate.front before instruction.guard = some true := by
          simp [partialPredicate, Instruction.PredicateRead.meaning, parts.1, parts.2]
        exact choice true defined)
      compatible member
    have rowMeaning := rust_successors instruction.op (effectiveGuard instruction.guard known)
      {pc := sourcePC, count := Slice.len certificate.code, cut := 0#u8, fold := known,
        returns := returnsAt certificate.returns pc}
      (by simpa [contextView, checkedEdges, nodeContext, sourcePCAt, Slice.len_val, Slice.length]
        using rowsFit pc instruction instructionAt known)
    simp only [successorsAt, WP.spec_ok] at rowMeaning
    have targetsMeaning : targets.val.map UScalar.val = checkedEdges instruction.op instruction.guard known
        (nodeContext certificate pc) := by
      simpa [RowPost, contextView, checkedEdges, nodeContext, sourcePCAt, Slice.len_val, Slice.length]
        using rowMeaning
    have targetMember : target ∈ targets.val.map UScalar.val := by rw [targetsMeaning]; exact checkedMember
    obtain ⟨sourceTarget, sourceTargetMember, sourceTargetValue⟩ := List.mem_map.mp targetMember
    obtain ⟨position, inside, positionAt⟩ := List.getElem_of_mem sourceTargetMember
    have incoming := targetsChecked position inside
    rw [positionAt] at incoming
    obtain ⟨nextFacts, nextAt, every⟩ := incoming
    refine ⟨nextFacts, by simpa only [sourceTargetValue] using nextAt, ?_, ?_⟩
    · intro fact factMember
      obtain ⟨index, bound, factAt⟩ := List.getElem_of_mem factMember
      have property := every index bound
      rw [factAt] at property
      exact source_post_value_defined reader
        {front := certificate.front, instruction, facts := alloc.vec.Vec.deref facts, constants := certificate.constants}
        fact.register before after bank (framesFit pc instruction instructionAt) property.1 (by simpa [alloc.vec.Vec.deref, Slice.from_val] using sound)
        constantSound stepPost fact.value property.2
    · intro fact factMember
      obtain ⟨index, bound, factAt⟩ := List.getElem_of_mem factMember
      have property := every index bound
      simpa only [factAt] using property.1

theorem accepted_certificate_guard_defined {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (observations : alloc.vec.Vec (Option (Option Bool)))
    (bank : U32 → U8 → Option U64)
    (returned : guard_certificate.validate reader certificate = .ok (some observations))
    (rowsFit : RowsFit certificate) (framesFit : FramesFit certificate)
    (constantSound : ConstantsSound reader certificate.constants bank)
    (pc : Nat) (registers : PartialRegisters) (execution : ExecutionDefined certificate bank pc registers)
    (instruction : Inst) (instructionAt : certificate.code.val[pc]? = some instruction)
    (truth : Bool) (observationAt : observations.val[pc]? = some (some (some truth))) :
    partialPredicate certificate.front registers instruction.guard = some truth := by
  have checked := successful_validate reader certificate observations returned
  obtain ⟨facts, stateAt, sound, _⟩ := certificate_execution_facts_defined reader certificate observations bank checked
    rowsFit framesFit constantSound pc registers execution
  obtain ⟨observation, observedAt, node⟩ := checked_node reader certificate observations checked pc facts stateAt
  rcases node with ⟨unreached, _⟩ | ⟨nodeFacts, nodeInstruction, sourcePC, known, targets,
    nodeStateAt, nodeInstructionAt, sourcePCAt, observed, decisionAt, successorsAt, targetsChecked⟩
  · rw [stateAt] at unreached
    cases unreached
  have sameFacts : nodeFacts = facts := by simpa [stateAt] using nodeStateAt.symm
  have sameInstruction : nodeInstruction = instruction := by simpa [instructionAt] using nodeInstructionAt.symm
  subst nodeFacts
  subst nodeInstruction
  have knownAt : known = some truth := by simpa [observationAt, observed] using observedAt.symm
  exact source_decision_defined certificate.front (alloc.vec.Vec.deref facts) registers (by simpa [alloc.vec.Vec.deref, Slice.from_val] using sound) instruction.guard truth
    (by simpa only [knownAt] using decisionAt)

end Instruction.GuardCertificate
