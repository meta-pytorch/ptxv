/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AnalysisKnowledge
import Instruction.PartialSuccess

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction.GuardCertificate

open RegisterOwner

abbrev Fact := guard_certificate.Fact

theorem source_same_value (left right : Option U64) :
    guard_certificate.same_value left right = .ok (decide (left = right)) := by
  cases left <;> cases right <;> simp [guard_certificate.same_value]

def EdgeFact {C : Type} (reader : guard_certificate.Constants C)
    (context : guard_certificate.Evaluation C) (fact : Fact) : Prop :=
  shared fact.register.file = false ∧
  guard_certificate.post_value reader context fact.register = .ok (some fact.value)

theorem successful_edge {C : Type} (reader : guard_certificate.Constants C)
    (context : guard_certificate.Evaluation C) (incoming : Slice Fact)
    (success : guard_certificate.edge reader context incoming = .ok true) :
    ∀ index, (inside : index < incoming.val.length) →
      EdgeFact reader context incoming.val[index] := by
  unfold guard_certificate.edge guard_certificate.edge_loop at success
  have execution := successful_loop_has_execution _ _ _ success
  have post := execution.preserves
    (fun position : Usize => position.val ≤ incoming.val.length ∧
      ∀ index, (inside : index < incoming.val.length) → index < position.val →
        EdgeFact reader context incoming.val[index])
    (fun accepted : Bool => accepted = true → ∀ index,
      (inside : index < incoming.val.length) → EdgeFact reader context incoming.val[index])
    (by constructor; scalar_tac; intro index inside before; scalar_tac) ?_
  · exact post rfl
  intro position invariant flow bodyOk
  obtain ⟨positionBound, priorFacts⟩ := invariant
  unfold guard_certificate.edge_loop.body at bodyOk
  dsimp only at bodyOk
  split at bodyOk
  · rename_i test
    have inside : position.val < incoming.val.length := by simpa only [UScalar.lt_equiv, Slice.len_val, Slice.length] using test
    obtain ⟨fact, factOk, privateOk⟩ := bind_success _ _ _ bodyOk
    have factAt := Slice.index_usize_spec incoming position inside
    simp only [factOk, WP.spec_ok] at factAt
    rw [Instruction.OwnerTemporal.source_thread_private] at privateOk
    simp only [bind_ok] at privateOk
    cases sharedAt : shared fact.register.file with
    | true =>
        simp only [sharedAt, Bool.not_true, Bool.false_eq_true, ite_false] at privateOk
        have equal := Result.ok_injective privateOk
        subst flow
        intro accepted
        cases accepted
    | false =>
        simp only [sharedAt, Bool.not_false, ite_true] at privateOk
        obtain ⟨value, valueOk, comparedOk⟩ := bind_success _ _ _ privateOk
        rw [source_same_value] at comparedOk
        simp only [bind_ok] at comparedOk
        by_cases same : value = some fact.value
        · simp only [same, decide_true, ite_true] at comparedOk
          obtain ⟨next, nextOk, returned⟩ := bind_success _ _ _ comparedOk
          have nextAt := Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac)
          simp only [nextOk, WP.spec_ok] at nextAt
          have equal := Result.ok_injective returned
          subst flow
          refine ⟨by scalar_tac, ?_⟩
          intro index indexInside before
          by_cases old : index < position.val
          · exact priorFacts index indexInside old
          · have current : index = position.val := by scalar_tac
            subst index
            rw [←factAt]
            exact ⟨sharedAt, by simpa only [same] using valueOk⟩
        · simp only [same, decide_false, ite_false] at comparedOk
          have equal := Result.ok_injective comparedOk
          subst flow
          intro accepted
          cases accepted
  · rename_i test
    have equal := Result.ok_injective bodyOk
    subst flow
    intro _ index indexInside
    have stopped : ¬position.val < incoming.val.length := by
      simpa only [UScalar.lt_equiv, Slice.len_val, Slice.length] using test
    exact priorFacts index indexInside (by omega)


def TargetChecked {C : Type} (reader : guard_certificate.Constants C)
    (context : guard_certificate.Evaluation C)
    (states : Slice (Option (alloc.vec.Vec Fact))) (target : Usize) : Prop :=
  ∃ incoming, states.val[target.val]? = some (some incoming) ∧
    ∀ index, (inside : index < incoming.val.length) →
      EdgeFact reader context incoming.val[index]

theorem successful_check_successors {C : Type} (reader : guard_certificate.Constants C)
    (context : guard_certificate.Evaluation C)
    (states : Slice (Option (alloc.vec.Vec Fact))) (successors : Slice Usize)
    (success : guard_certificate.check_successors reader context states successors = .ok true) :
    ∀ index, (inside : index < successors.val.length) →
      TargetChecked reader context states successors.val[index] := by
  unfold guard_certificate.check_successors guard_certificate.check_successors_loop at success
  have execution := successful_loop_has_execution _ _ _ success
  have post := execution.preserves
    (fun position : Usize => position.val ≤ successors.val.length ∧
      ∀ index, (inside : index < successors.val.length) → index < position.val →
        TargetChecked reader context states successors.val[index])
    (fun accepted : Bool => accepted = true → ∀ index,
      (inside : index < successors.val.length) → TargetChecked reader context states successors.val[index])
    (by constructor; scalar_tac; intro index inside before; scalar_tac) ?_
  · exact post rfl
  intro position invariant flow bodyOk
  obtain ⟨positionBound, priorTargets⟩ := invariant
  unfold guard_certificate.check_successors_loop.body at bodyOk
  dsimp only at bodyOk
  split at bodyOk
  · rename_i test
    have inside : position.val < successors.val.length := by simpa only [UScalar.lt_equiv, Slice.len_val, Slice.length] using test
    obtain ⟨target, targetOk, boundsOk⟩ := bind_success _ _ _ bodyOk
    have targetAt := Slice.index_usize_spec successors position inside
    simp only [targetOk, WP.spec_ok] at targetAt
    split at boundsOk
    · have equal := Result.ok_injective boundsOk
      subst flow
      intro accepted
      cases accepted
    · rename_i targetBound
      have targetInside : target.val < states.val.length := by
        simp only [ge_iff_le, UScalar.le_equiv, Slice.len_val, Slice.length] at targetBound
        omega
      obtain ⟨present, presentOk, edgeOk⟩ := bind_success _ _ _ boundsOk
      have presentAt := Slice.index_usize_spec states target targetInside
      simp only [presentOk, WP.spec_ok] at presentAt
      cases present with
      | none =>
          simp only at edgeOk
          have equal := Result.ok_injective edgeOk
          subst flow
          intro accepted
          cases accepted
      | some incoming =>
          simp only at edgeOk
          obtain ⟨accepted, acceptedOk, advancedOk⟩ := bind_success _ _ _ edgeOk
          cases accepted with
          | false =>
              simp only [Bool.false_eq_true, ite_false] at advancedOk
              have equal := Result.ok_injective advancedOk
              subst flow
              intro accepted
              cases accepted
          | true =>
              simp only [ite_true] at advancedOk
              obtain ⟨next, nextOk, returned⟩ := bind_success _ _ _ advancedOk
              have nextAt := Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac)
              simp only [nextOk, WP.spec_ok] at nextAt
              have equal := Result.ok_injective returned
              subst flow
              refine ⟨by scalar_tac, ?_⟩
              intro index indexInside before
              by_cases old : index < position.val
              · exact priorTargets index indexInside old
              · have current : index = position.val := by scalar_tac
                subst index
                rw [←targetAt]
                refine ⟨incoming, ?_, ?_⟩
                · rw [List.getElem?_eq_getElem targetInside, ←presentAt]
                · simpa only [alloc.vec.Vec.deref, Slice.from_val] using
                    successful_edge reader context (alloc.vec.Vec.deref incoming) acceptedOk
  · rename_i test
    have equal := Result.ok_injective bodyOk
    subst flow
    intro _ index indexInside
    exact priorTargets index indexInside (by
      simp only [UScalar.lt_equiv, Slice.len_val, Slice.length] at test
      omega)

end Instruction.GuardCertificate
