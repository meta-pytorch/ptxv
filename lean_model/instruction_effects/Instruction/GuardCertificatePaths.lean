/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateTransfer
import Instruction.GuardCertificateControl

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 7000000
namespace Instruction.GuardCertificate

open Instruction.ControlFlow

def nodeContext {C : Type} (certificate : guard_certificate.Certificate C) (pc : Nat) : Context :=
  {pc, count := certificate.code.val.length, cut := 0#u8, fold := none,
    returns := (returnsAt certificate.returns pc).map fun row => row.val.map UScalar.val}

def RowsFit {C : Type} (certificate : guard_certificate.Certificate C) : Prop :=
  ∀ (pc : Nat) (instruction : Inst), certificate.code.val[pc]? = some instruction → ∀ known,
    (checkedEdges instruction.op instruction.guard known (nodeContext certificate pc)).length ≤ Usize.max

def FramesFit {C : Type} (certificate : guard_certificate.Certificate C) : Prop :=
  ∀ (pc : Nat) (instruction : Inst), certificate.code.val[pc]? = some instruction → FrameFits instruction.op

inductive Execution {C : Type} (certificate : guard_certificate.Certificate C)
    (bank : U32 → U8 → Option U64) : Nat → (Reg → U64) → Prop
  | entry (registers : Reg → U64) : Execution certificate bank certificate.entry.val registers
  | peer {pc : Nat} {before after : Reg → U64} :
      Execution certificate bank pc before →
      (∀ register, RegisterOwner.shared register.file = false → after register = before register) →
      Execution certificate bank pc after
  | own {pc target : Nat} {before after : Reg → U64} {instruction : Inst} :
      Execution certificate bank pc before →
      certificate.code.val[pc]? = some instruction →
      StepPost certificate.front instruction bank before after →
      target ∈ semanticEdges instruction.op instruction.guard
        (RegisterOwner.predicateRead certificate.front (fun register => before register != 0#u64) instruction.guard)
        (nodeContext certificate pc) →
      Execution certificate bank target after

def StateSound {C : Type} (certificate : guard_certificate.Certificate C)
    (pc : Nat) (registers : Reg → U64) : Prop :=
  ∃ facts, certificate.states.val[pc]? = some (some facts) ∧
    FactsSound facts.val registers ∧ FactsPrivate facts.val

theorem checked_node {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (observations : alloc.vec.Vec (Option (Option Bool)))
    (checked : CertificateChecked reader certificate observations)
    (pc : Nat) (facts : alloc.vec.Vec Fact)
    (stateAt : certificate.states.val[pc]? = some (some facts)) :
    ∃ observation, observations.val[pc]? = some observation ∧
      NodeChecked reader certificate.front certificate.code certificate.states certificate.returns
        certificate.constants pc observation := by
  have stateBound : pc < certificate.states.val.length := by
    by_contra outside
    have absent : certificate.states.val[pc]? = none := List.getElem?_eq_none (by omega)
    rw [absent] at stateAt
    cases stateAt
  have observationBound : pc < observations.val.length := by rw [checked.2.2.1, checked.1]; exact stateBound
  exact ⟨observations.val[pc], List.getElem?_eq_getElem observationBound,
    checked.2.2.2 pc observationBound⟩

theorem certificate_execution_facts_sound {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (observations : alloc.vec.Vec (Option (Option Bool)))
    (bank : U32 → U8 → Option U64)
    (checked : CertificateChecked reader certificate observations)
    (rowsFit : RowsFit certificate) (framesFit : FramesFit certificate)
    (constantSound : ConstantsSound reader certificate.constants bank)
    (pc : Nat) (registers : Reg → U64) (execution : Execution certificate bank pc registers) :
    StateSound certificate pc registers := by
  induction execution with
  | entry registers =>
    obtain ⟨facts, stateAt, empty⟩ := checked.2.1
    refine ⟨facts, stateAt, ?_, ?_⟩ <;> simp [FactsSound, FactsPrivate, empty]
  | peer prior privateAgree ih =>
    obtain ⟨facts, stateAt, sound, privateFacts⟩ := ih
    refine ⟨facts, stateAt, ?_, privateFacts⟩
    intro fact member
    exact (privateAgree fact.register (privateFacts fact member)).trans (sound fact member)
  | @own pc target before after instruction prior instructionAt stepPost member ih =>
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
    have compatible : ∀ observed, known = some observed → observed =
        RegisterOwner.predicateRead certificate.front (fun register => before register != 0#u64) instruction.guard := by
      intro observed present
      exact source_decision_sound certificate.front (alloc.vec.Vec.deref facts) before (by simpa [alloc.vec.Vec.deref, Slice.from_val] using sound)
        instruction.guard observed (by simpa only [present] using decisionAt)
    have checkedMember := checked_edges_cover_truth instruction.op instruction.guard _ known
      (nodeContext certificate pc) target (structural_guard_true certificate.front before instruction.guard)
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
      exact source_post_value_sound reader
        {front := certificate.front, instruction, facts := alloc.vec.Vec.deref facts, constants := certificate.constants}
        fact.register before after bank (framesFit pc instruction instructionAt) property.1 (by simpa [alloc.vec.Vec.deref, Slice.from_val] using sound)
        constantSound stepPost fact.value property.2
    · intro fact factMember
      obtain ⟨index, bound, factAt⟩ := List.getElem_of_mem factMember
      have property := every index bound
      simpa only [factAt] using property.1

theorem accepted_certificate_guard_truth {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (observations : alloc.vec.Vec (Option (Option Bool)))
    (bank : U32 → U8 → Option U64)
    (returned : guard_certificate.validate reader certificate = .ok (some observations))
    (rowsFit : RowsFit certificate) (framesFit : FramesFit certificate)
    (constantSound : ConstantsSound reader certificate.constants bank)
    (pc : Nat) (registers : Reg → U64) (execution : Execution certificate bank pc registers)
    (instruction : Inst) (instructionAt : certificate.code.val[pc]? = some instruction)
    (truth : Bool) (observationAt : observations.val[pc]? = some (some (some truth))) :
    truth = RegisterOwner.predicateRead certificate.front (fun register => registers register != 0#u64) instruction.guard := by
  have checked := successful_validate reader certificate observations returned
  obtain ⟨facts, stateAt, sound, _⟩ := certificate_execution_facts_sound reader certificate observations bank checked
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
  exact source_decision_sound certificate.front (alloc.vec.Vec.deref facts) registers (by simpa [alloc.vec.Vec.deref, Slice.from_val] using sound) instruction.guard truth
    (by simpa only [knownAt] using decisionAt)

end Instruction.GuardCertificate
