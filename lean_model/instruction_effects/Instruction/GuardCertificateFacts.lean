/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateCheck

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction.GuardCertificate

theorem successful_same_register (left right : Reg)
    (success : guard_certificate.same_register left right = .ok true) : left = right := by
  cases left with
  | mk lf li =>
    cases right with
    | mk rf ri =>
      cases lf <;> cases rf <;>
        simp_all [guard_certificate.same_register, read_discriminant, File.read_discriminant, lift]

def FactsSound (facts : List Fact) (registers : Reg → U64) : Prop :=
  ∀ fact ∈ facts, registers fact.register = fact.value

def FactsPrivate (facts : List Fact) : Prop :=
  ∀ fact ∈ facts, RegisterOwner.shared fact.register.file = false

theorem successful_lookup (facts : Slice Fact) (register : Reg) (value : U64)
    (success : guard_certificate.lookup facts register = .ok (some value)) :
    ∃ fact ∈ facts.val, fact.register = register ∧ fact.value = value := by
  unfold guard_certificate.lookup guard_certificate.lookup_loop at success
  have execution := successful_loop_has_execution _ _ _ success
  have post := execution.preserves
    (fun position : Usize => position.val ≤ facts.val.length)
    (fun observed : Option U64 => ∀ value, observed = some value →
      ∃ fact ∈ facts.val, fact.register = register ∧ fact.value = value)
    (by scalar_tac) ?_
  · exact post value rfl
  intro position bound flow bodyOk
  unfold guard_certificate.lookup_loop.body at bodyOk
  dsimp only at bodyOk
  split at bodyOk
  · rename_i test
    have inside : position.val < facts.val.length := by
      simpa only [UScalar.lt_equiv, Slice.len_val, Slice.length] using test
    obtain ⟨fact, factOk, sameOk⟩ := bind_success _ _ _ bodyOk
    have factAt := Slice.index_usize_spec facts position inside
    simp only [factOk, WP.spec_ok] at factAt
    obtain ⟨same, sameRead, restOk⟩ := bind_success _ _ _ sameOk
    cases same with
    | true =>
        simp only [ite_true] at restOk
        have equal := Result.ok_injective restOk
        subst flow
        intro found equal
        have valueSame := Option.some.inj equal
        refine ⟨fact, ?_, successful_same_register fact.register register sameRead, valueSame⟩
        rw [factAt]
        exact List.getElem_mem inside
    | false =>
        simp only [Bool.false_eq_true, ite_false] at restOk
        obtain ⟨next, nextOk, returned⟩ := bind_success _ _ _ restOk
        have nextAt := Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac)
        simp only [nextOk, WP.spec_ok] at nextAt
        have equal := Result.ok_injective returned
        subst flow
        scalar_tac
  · have equal := Result.ok_injective bodyOk
    subst flow
    intro value impossible
    cases impossible

theorem lookup_sound (facts : Slice Fact) (registers : Reg → U64)
    (sound : FactsSound facts.val registers) (register : Reg) (value : U64)
    (success : guard_certificate.lookup facts register = .ok (some value)) :
    registers register = value := by
  obtain ⟨fact, member, same, assigned⟩ := successful_lookup facts register value success
  simpa only [same, assigned] using sound fact member

end Instruction.GuardCertificate
