/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicLaneScan
import MemoryWrite.RustAtomicScalar

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.AtomicLaneScan

abbrev Claim := U32 × U8 × Usize × U64

def pairModel (ptx used continued shared : Bool) (left right : Claim) :
    AtomicLaneCoverage.Pair :=
  ⟨ptx,used,continued,decide (left.1 = right.1),
    decide (left.2.1.val ≠ 0 ∧ left.2.1 = right.2.1 ∧
      left.2.2.1 = right.2.2.1 ∧ left.2.2.2 = right.2.2.2),shared⟩

def Reports (ptx used continued shared : Bool) (left right : Claim) : Prop :=
  AtomicLaneCoverage.needsDiagnostic (pairModel ptx used continued shared left right) = true

@[step] theorem rust_claim_pair (ptx used continued shared : Bool) (left right : Claim) :
    atomic_scalar.claim_pair_unmodeled ptx used continued shared left right
      ⦃ result => result = AtomicLaneCoverage.needsDiagnostic
        (pairModel ptx used continued shared left right) ⦄ := by
  rcases left with ⟨lw,lr,la,lo⟩
  rcases right with ⟨rw,rr,ra,ro⟩
  by_cases hz : lr.val = 0 <;> by_cases hr : lr = rr <;>
    by_cases ha : la = ra <;> by_cases ho : lo = ro <;>
    simp_all [atomic_scalar.claim_pair_unmodeled, pairModel] <;>
    apply AtomicScalar.rust_bundled_order_unmodeled

@[step] theorem rust_unmodeled_with (ptx used continued shared : Bool) (left : Claim)
    (claims : Slice Claim) (start : Usize) :
    atomic_scalar.unmodeled_with ptx used continued shared left claims start
      ⦃ result => result = true ↔ AnySuffix (Reports ptx used continued shared)
        left claims.val start.val ⦄ := by
  unfold atomic_scalar.unmodeled_with atomic_scalar.unmodeled_with_loop
  apply loop.spec_decr_nat
    (fun index : Usize => claims.val.length - index.val)
    (fun index : Usize => AnySuffix (Reports ptx used continued shared) left claims.val index.val ↔
      AnySuffix (Reports ptx used continued shared) left claims.val start.val)
  · intro index remaining
    unfold atomic_scalar.unmodeled_with_loop.body
    by_cases inside : index.val < claims.val.length
    · have test : index < Slice.len claims := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec claims index inside as ⟨right,rightEq⟩
      have atClaim : right = claims.val[index.val]! := by
        rw [rightEq,List.Inhabited_getElem_eq_getElem! _ _ inside]
      step with rust_claim_pair ptx used continued shared left right as ⟨found,foundEq⟩
      split
      · rename_i foundTrue
        simp only [WP.spec_ok,eq_self,true_iff]
        apply remaining.mp
        exact ⟨index.val,by omega,inside,by simpa [Reports,atClaim] using foundEq.symm.trans foundTrue⟩
      · rename_i foundFalse
        have absent : ¬Reports ptx used continued shared left claims.val[index.val]! := by
          intro reports
          apply foundFalse
          rw [foundEq,atClaim]
          exact reports
        step as ⟨next,nextEq⟩
        refine ⟨?_,by scalar_tac⟩
        have nextVal : next.val = index.val + 1 := by scalar_tac
        rw [nextVal]
        exact (suffix_advance _ _ _ _ absent).symm.trans remaining
    · have test : ¬index < Slice.len claims := by scalar_tac
      simp only [test, ite_false,WP.spec_ok,Bool.false_eq_true,false_iff]
      exact fun found => no_suffix_at_the_end _ _ _ _ (by omega) (remaining.mpr found)
  · rfl

@[step] theorem rust_unmodeled_claims (ptx used continued shared : Bool) (claims : Slice Claim) :
    atomic_scalar.unmodeled_claims ptx used continued shared claims
      ⦃ result => result = true ↔ AnyPair (Reports ptx used continued shared) claims.val 0 ⦄ := by
  unfold atomic_scalar.unmodeled_claims atomic_scalar.unmodeled_claims_loop
  apply loop.spec_decr_nat
    (fun index : Usize => claims.val.length - index.val)
    (fun index : Usize => AnyPair (Reports ptx used continued shared) claims.val index.val ↔
      AnyPair (Reports ptx used continued shared) claims.val 0)
  · intro index remaining
    unfold atomic_scalar.unmodeled_claims_loop.body
    by_cases inside : index.val < claims.val.length
    · have test : index < Slice.len claims := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec claims index inside as ⟨left,leftEq⟩
      have atClaim : left = claims.val[index.val]! := by
        rw [leftEq,List.Inhabited_getElem_eq_getElem! _ _ inside]
      step as ⟨start,startEq⟩
      have startVal : start.val = index.val + 1 := by scalar_tac
      step with rust_unmodeled_with ptx used continued shared left claims start as ⟨found,foundEq⟩
      split
      · rename_i foundTrue
        simp only [WP.spec_ok,eq_self,true_iff]
        apply remaining.mp
        apply a_pair_is_found_in_its_first_suffix
        simpa [atClaim,startVal] using foundEq.mp foundTrue
      · rename_i foundFalse
        have absent : ¬AnySuffix (Reports ptx used continued shared) claims.val[index.val]!
            claims.val (index.val + 1) := by
          intro reports
          apply foundFalse
          apply foundEq.mpr
          simpa [atClaim,startVal] using reports
        simp only [WP.spec_ok]
        refine ⟨?_,by scalar_tac⟩
        rw [startVal]
        exact (pair_advance _ _ _ absent).symm.trans remaining
    · have test : ¬index < Slice.len claims := by scalar_tac
      simp only [test, ite_false,WP.spec_ok,Bool.false_eq_true,false_iff]
      exact fun found => no_pair_at_the_end _ _ _ (by omega) (remaining.mpr found)
  · rfl

@[step] theorem source_an_undiagnosed_claim_list_covers_every_interfering_pair
    (continued shared : Bool) (claims : Slice Claim) :
    atomic_scalar.unmodeled_claims true true continued shared claims
      ⦃ result => result = false → ∀ first second, first < second → second < claims.val.length →
        AtomicLaneCoverage.interferes
          (pairModel true true continued shared claims.val[first]! claims.val[second]!) = true →
        continued = true ⦄ := by
  apply WP.spec_mono (rust_unmodeled_claims true true continued shared claims)
  intro result resultSpec absent first second ordered inside conflicting
  cases continuation : continued
  · have reported : Reports true true continued shared claims.val[first]! claims.val[second]! := by
      apply AtomicLaneCoverage.an_uncontinued_returning_interfering_pair_is_diagnosed
      · rfl
      · rfl
      · exact continuation
      · exact conflicting
    have found := resultSpec.mpr ⟨first,second,by omega,ordered,inside,reported⟩
    simp [absent] at found
  · rfl

end MemoryWrite.AtomicLaneScan
