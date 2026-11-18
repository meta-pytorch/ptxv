/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateDefinedPaths
import Instruction.GuardCertificateRowsTruth

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 6000000
namespace Instruction.GuardCertificate

theorem reached_defined_observation_present {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (observations : alloc.vec.Vec Observation)
    (bank : U32 → U8 → Option U64)
    (returned : guard_certificate.validate reader certificate = .ok (some observations))
    (rowsFit : RowsFit certificate) (framesFit : FramesFit certificate)
    (constantSound : ConstantsSound reader certificate.constants bank)
    (pc : Nat) (registers : PartialRegisters) (execution : ExecutionDefined certificate bank pc registers) :
    ∃ current, observations.val[pc]? = some (some current) := by
  have checked := successful_validate reader certificate observations returned
  obtain ⟨facts, stateAt, _, _⟩ := certificate_execution_facts_defined reader certificate observations bank checked
    rowsFit framesFit constantSound pc registers execution
  obtain ⟨observation, observationAt, node⟩ := checked_node reader certificate observations checked pc facts stateAt
  rcases node with ⟨unreached, _⟩ | ⟨_, _, _, known, _, _, _, _, observed, _, _, _⟩
  · rw [stateAt] at unreached
    cases unreached
  · exact ⟨known, by simpa [observed] using observationAt⟩

theorem published_rows_guard_defined {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (observations : alloc.vec.Vec Observation)
    (bank : U32 → U8 → Option U64)
    (returned : guard_certificate.validate reader certificate = .ok (some observations))
    (rowsFit : RowsFit certificate) (framesFit : FramesFit certificate)
    (constantSound : ConstantsSound reader certificate.constants bank)
    (rows : List (Slice Observation)) (before after : Slice Observation)
    (calls : MergeRowsCalls rows before after)
    (member : alloc.vec.Vec.deref observations ∈ rows)
    (output : alloc.vec.Vec (Option Bool))
    (published : guard_certificate.publish certificate.code after = .ok (some output))
    (pc : Nat) (initial : before[pc]! = none) (registers : PartialRegisters)
    (execution : ExecutionDefined certificate bank pc registers)
    (instruction : Inst) (instructionAt : certificate.code.val[pc]? = some instruction)
    (truth : Bool) (outputAt : output.val[pc]? = some (some truth)) :
    partialPredicate certificate.front registers instruction.guard = some truth := by
  have codeInside : pc < certificate.code.length := by
    by_contra outside
    have absent : certificate.code.val[pc]? = none := List.getElem?_eq_none (by simpa only [Slice.length] using Nat.le_of_not_lt outside)
    rw [absent] at instructionAt
    cases instructionAt
  have sequence := merge_calls_meaning rows before after calls
  have publication := successful_publish certificate.code after output published
  have sameLength := sequence.1
  have sameCode := publication.1
  have beforeInside : pc < before.length := by omega
  have afterInside : pc < after.length := by omega
  have cell := publication.2.2 pc codeInside
  have finalTruth : output.val[pc]! = some truth := List.getElem!_of_getElem? outputAt
  have classifier : branchFold instruction after[pc]! = some truth := by
    simp only [publicationCell, instructionAt, slice_observation_lookup after pc afterInside] at cell
    exact cell.symm.trans finalTruth
  have accumulated := branch_fold_known instruction after[pc]! truth classifier
  obtain ⟨current, currentAt⟩ := reached_defined_observation_present reader certificate observations bank returned
    rowsFit framesFit constantSound pc registers execution
  have currentValue : (alloc.vec.Vec.deref observations)[pc]! = some current := by
    simpa only [Slice.getElem!_Nat_eq, alloc.vec.Vec.deref, Slice.from_val] using
      (List.getElem!_of_getElem? currentAt)
  have currentTruth := merged_rows_known rows before after calls pc beforeInside initial truth accumulated
    (alloc.vec.Vec.deref observations) member current currentValue
  apply accepted_certificate_guard_defined reader certificate observations bank returned rowsFit framesFit
    constantSound pc registers execution instruction instructionAt truth
  simpa only [currentTruth] using currentAt

end Instruction.GuardCertificate
