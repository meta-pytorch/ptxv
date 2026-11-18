/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificatePublicationSource

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction.GuardCertificate

def publicationCell (code : Slice Inst) (observations : Slice Observation) (pc : Nat) : Option Bool :=
  match code.val[pc]?, observations.val[pc]? with
  | some instruction, some observation => branchFold instruction observation
  | _, _ => none

def RowsPublished (code : Slice Inst) (observations : Slice Observation)
    (output : alloc.vec.Vec (Option Bool)) : Prop :=
  output.length = code.length ∧ ∀ pc < code.length,
    output.val[pc]! = publicationCell code observations pc

def PublishInvariant (code : Slice Inst) (observations : Slice Observation)
    (output : alloc.vec.Vec (Option Bool)) (position : Usize) : Prop :=
  output.length = position.val ∧ position.val ≤ code.length ∧
  ∀ pc < position.val, output.val[pc]! = publicationCell code observations pc

@[step]
theorem rust_publish_loop (code : Slice Inst) (observations : Slice Observation)
    (output : alloc.vec.Vec (Option Bool)) (position : Usize)
    (lengths : code.length = observations.length)
    (initial : PublishInvariant code observations output position) :
    guard_certificate.publish_loop code observations output position
      ⦃ result => RowsPublished code observations result ⦄ := by
  unfold guard_certificate.publish_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec (Option Bool) × Usize => code.length - state.2.val)
    (fun state : alloc.vec.Vec (Option Bool) × Usize => PublishInvariant code observations state.1 state.2)
  · rintro ⟨output, position⟩ invariant
    obtain ⟨outputLength, bound, meaning⟩ := invariant
    unfold guard_certificate.publish_loop.body
    dsimp only at *
    split
    · rename_i test
      have inside : position.val < code.length := by
        simpa only [UScalar.lt_equiv, Slice.len_val] using test
      have observationInside : position.val < observations.length := by omega
      step with Slice.index_usize_spec code position inside as ⟨instruction, instructionAt⟩
      step with Slice.index_usize_spec observations position observationInside as ⟨observation, observationAt⟩
      step with rust_branch_fold instruction observation as ⟨cell, cellAt⟩
      have sourceInstruction : code.val[position.val]? = some instruction := by
        rw [List.getElem?_eq_getElem inside, instructionAt]
      have sourceObservation : observations.val[position.val]? = some observation := by
        rw [List.getElem?_eq_getElem observationInside, observationAt]
      have cellMeaning : cell = publicationCell code observations position.val := by
        simpa [publicationCell, sourceInstruction, sourceObservation] using cellAt
      step with alloc.vec.Vec.push_spec output cell (by scalar_tac) as ⟨after, afterAt⟩
      step with Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac) as ⟨next, nextAt⟩
      have nextValue : next.val = position.val + 1 := by scalar_tac
      refine ⟨?_, by omega⟩
      refine ⟨?_, by omega, ?_⟩
      · simp [alloc.vec.Vec.length, afterAt, ←outputLength, nextValue]
      · intro pc pcInside
        rw [afterAt]
        by_cases earlier : pc < position.val
        · rw [List.getElem!_append_left output.val [cell] pc (by simpa [alloc.vec.Vec.length, outputLength] using earlier)]
          exact meaning pc earlier
        · have same : pc = position.val := by omega
          subst pc
          rw [List.getElem!_append_right output.val [cell] position.val (by simpa [alloc.vec.Vec.length] using outputLength.le)]
          simpa [←outputLength, alloc.vec.Vec.length] using cellMeaning
    · rename_i outside
      have outside' : ¬position.val < code.length := by
        simpa only [UScalar.lt_equiv, Slice.len_val] using outside
      have complete : position.val = code.length := by omega
      simp only [WP.spec_ok, RowsPublished]
      exact ⟨outputLength.trans complete, by simpa [complete] using meaning⟩
  · exact initial

@[step]
theorem rust_publish (code : Slice Inst) (observations : Slice Observation)
    (lengths : code.length = observations.length) :
    guard_certificate.publish code observations
      ⦃ result => ∃ output, result = some output ∧ RowsPublished code observations output ⦄ := by
  unfold guard_certificate.publish
  have same : Slice.len code = Slice.len observations := by
    apply (UScalar.eq_equiv _ _).mpr
    simpa only [Slice.len_val] using lengths
  rw [same]
  simp only [bne_self_eq_false, Bool.false_eq_true, ite_false]
  step with rust_publish_loop code observations
    (alloc.vec.Vec.with_capacity (Option Bool) (Slice.len observations)) 0#usize lengths
    (by simp [PublishInvariant, alloc.vec.Vec.with_capacity, alloc.vec.Vec.new, alloc.vec.Vec.length])
    as ⟨output, outputMeaning⟩
  exact ⟨output, rfl, outputMeaning⟩

end Instruction.GuardCertificate
