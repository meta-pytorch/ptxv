/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificatePublicationSource
import Instruction.GuardCertificatePaths

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 2000000
namespace Instruction.GuardCertificate

theorem reached_observation_present {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (observations : alloc.vec.Vec Observation)
    (bank : U32 → U8 → Option U64)
    (returned : guard_certificate.validate reader certificate = .ok (some observations))
    (rowsFit : RowsFit certificate) (framesFit : FramesFit certificate)
    (constantSound : ConstantsSound reader certificate.constants bank)
    (pc : Nat) (registers : Reg → U64) (execution : Execution certificate bank pc registers) :
    ∃ current, observations.val[pc]? = some (some current) := by
  have checked := successful_validate reader certificate observations returned
  obtain ⟨facts, stateAt, _, _⟩ := certificate_execution_facts_sound reader certificate observations bank checked
    rowsFit framesFit constantSound pc registers execution
  obtain ⟨observation, observationAt, node⟩ := checked_node reader certificate observations checked pc facts stateAt
  rcases node with ⟨unreached, _⟩ | ⟨_, _, _, known, _, _, _, _, observed, _, _, _⟩
  · rw [stateAt] at unreached
    cases unreached
  · exact ⟨known, by simpa [observed] using observationAt⟩

theorem published_fold_guard_truth {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (observations : alloc.vec.Vec Observation)
    (bank : U32 → U8 → Option U64)
    (returned : guard_certificate.validate reader certificate = .ok (some observations))
    (rowsFit : RowsFit certificate) (framesFit : FramesFit certificate)
    (constantSound : ConstantsSound reader certificate.constants bank)
    (pc : Nat) (registers : Reg → U64) (execution : Execution certificate bank pc registers)
    (instruction : Inst) (instructionAt : certificate.code.val[pc]? = some instruction)
    (gridObservations : List Observation) (currentObservation : Observation)
    (observationAt : observations.val[pc]? = some currentObservation)
    (member : currentObservation ∈ gridObservations) (truth : Bool)
    (published : guard_certificate.branch_fold instruction
      (gridObservations.foldl mergeObservation none) = .ok (some truth)) :
    truth = RegisterOwner.predicateRead certificate.front
      (fun register => registers register != 0#u64) instruction.guard := by
  have meaning := rust_branch_fold instruction (gridObservations.foldl mergeObservation none)
  simp only [published, WP.spec_ok] at meaning
  have accumulated := branch_fold_known instruction _ truth meaning.symm
  obtain ⟨current, currentAt⟩ := reached_observation_present reader certificate observations bank returned
    rowsFit framesFit constantSound pc registers execution
  have currentEq : currentObservation = some current := by simpa [observationAt] using currentAt
  subst currentObservation
  have currentKnown := known_merged_observation gridObservations truth accumulated current member
  apply accepted_certificate_guard_truth reader certificate observations bank returned rowsFit framesFit
    constantSound pc registers execution instruction instructionAt truth
  simpa [currentKnown] using currentAt

end Instruction.GuardCertificate
