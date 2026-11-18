/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ControlFlowPrimitives

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 2000000
namespace Instruction.GuardCertificate

abbrev Observation := Option (Option Bool)

def mergeObservation (previous incoming : Observation) : Observation :=
  match incoming with
  | none => previous
  | some value => some (match previous with
    | none => value
    | some old => if old = value then value else none)

def Compatible (truth : Bool) (observation : Observation) : Prop :=
  observation = none ∨ observation = some (some truth)

theorem merge_compatible (truth : Bool) (previous incoming : Observation)
    (compatible : Compatible truth (mergeObservation previous incoming)) :
    Compatible truth previous ∧ Compatible truth incoming := by
  cases previous with
  | none => cases incoming <;> simp_all [Compatible, mergeObservation]
  | some previous =>
    cases previous with
    | none => cases incoming <;> simp_all [Compatible, mergeObservation]
    | some previous =>
      cases incoming with
      | none => simp_all [Compatible, mergeObservation]
      | some incoming =>
        cases incoming with
        | none => simp_all [Compatible, mergeObservation]
        | some incoming =>
          by_cases same : previous = incoming <;> simp_all [Compatible, mergeObservation]

theorem folded_observations_known (observations : List Observation) (initial : Observation)
    (truth : Bool)
    (returned : observations.foldl mergeObservation initial = some (some truth)) :
    Compatible truth initial ∧ ∀ observation ∈ observations, Compatible truth observation := by
  induction observations generalizing initial with
  | nil => simp_all [Compatible]
  | cons observation rest ih =>
    have checked := ih (mergeObservation initial observation) (by simpa using returned)
    have first := merge_compatible truth initial observation checked.1
    exact ⟨first.1, by simpa only [List.mem_cons, forall_eq_or_imp] using And.intro first.2 checked.2⟩

theorem known_merged_observation (observations : List Observation) (truth : Bool)
    (returned : observations.foldl mergeObservation none = some (some truth))
    (observation : Option Bool) (member : some observation ∈ observations) :
    observation = some truth := by
  have compatible := (folded_observations_known observations none truth returned).2 _ member
  simpa [Compatible] using compatible

@[step]
theorem rust_merge_grid (previous incoming : Observation) :
    guard_certificate.merge_grid previous incoming
      ⦃ result => result = mergeObservation previous incoming ⦄ := by
  cases incoming with
  | none => simp [guard_certificate.merge_grid, mergeObservation]
  | some incoming =>
    cases previous with
    | none => simp [guard_certificate.merge_grid, mergeObservation]
    | some previous =>
      cases incoming <;> cases previous <;>
        simp [guard_certificate.merge_grid, mergeObservation]
      split <;> simp_all

def branchFold (instruction : Inst) (observation : Observation) : Option Bool :=
  match instruction.op, observation with
  | .Bra _ false false false none none, some (some truth) =>
      if Instruction.ControlFlow.always instruction.guard then none else some truth
  | _, _ => none

@[step]
theorem rust_branch_fold (instruction : Inst) (observation : Observation) :
    guard_certificate.branch_fold instruction observation
      ⦃ result => result = branchFold instruction observation ⦄ := by
  rcases instruction with ⟨guard, operation, uniform, loc, text⟩
  cases operation <;> try simp [guard_certificate.branch_fold, branchFold]
  case Bra target any tid all condition divergent =>
    cases any <;> cases tid <;> cases all <;> cases condition <;> cases divergent <;> cases observation <;>
      try simp [guard_certificate.branch_fold, branchFold]
    rename_i observed
    cases observed <;> simp [guard_certificate.branch_fold, branchFold,
      Instruction.ControlFlow.rust_pred_always]
    split <;> simp_all

theorem branch_fold_known (instruction : Inst) (observation : Observation) (truth : Bool)
    (published : branchFold instruction observation = some truth) :
    observation = some (some truth) := by
  rcases instruction with ⟨guard, operation, uniform, loc, text⟩
  cases operation <;> try simp [branchFold] at published
  case Bra target any tid all condition divergent =>
    cases any <;> cases tid <;> cases all <;> cases condition <;> cases divergent <;> cases observation <;>
      try simp [branchFold] at published
    rename_i observed
    cases observed <;> simp [branchFold] at published
    simp_all

end Instruction.GuardCertificate
