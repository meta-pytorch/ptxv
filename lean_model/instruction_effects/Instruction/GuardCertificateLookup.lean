/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateFacts
import Instruction.Model

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction.GuardCertificate

deriving instance DecidableEq for Reg

def lookup (facts : List Fact) (register : Reg) : Option U64 :=
  (facts.find? fun fact => decide (fact.register = register)).map guard_certificate.Fact.value

theorem source_same_register (left right : Reg) :
    guard_certificate.same_register left right = .ok (decide (left = right)) := by
  cases left with
  | mk lf li =>
    cases right with
    | mk rf ri =>
      cases lf <;> cases rf <;>
        simp [guard_certificate.same_register, read_discriminant, File.read_discriminant, lift]

@[simp] theorem lookup_nil (register : Reg) : lookup [] register = none := by simp [lookup]

@[simp] theorem lookup_cons (fact : Fact) (facts : List Fact) (register : Reg) :
    lookup (fact :: facts) register =
      if fact.register = register then some fact.value else lookup facts register := by
  by_cases same : fact.register = register <;> simp [lookup, same]

theorem lookup_drop_step (facts : List Fact) (register : Reg) (index : Nat)
    (inside : index < facts.length) :
    lookup (facts.drop index) register =
      if facts[index].register = register then some facts[index].value
      else lookup (facts.drop (index + 1)) register := by
  induction facts generalizing index with
  | nil => simp at inside
  | cons first rest ih =>
      cases index with
      | zero => simp
      | succ index => simpa using ih index (by simpa using inside)

@[step]
theorem rust_lookup (facts : Slice Fact) (register : Reg) :
    guard_certificate.lookup facts register ⦃ result => result = lookup facts.val register ⦄ := by
  unfold guard_certificate.lookup guard_certificate.lookup_loop
  apply loop.spec_decr_nat
    (fun position : Usize => facts.val.length - position.val)
    (fun position : Usize => position.val ≤ facts.val.length ∧
      lookup facts.val register = lookup (facts.val.drop position.val) register)
  · intro position invariant
    obtain ⟨bound, meaning⟩ := invariant
    unfold guard_certificate.lookup_loop.body
    dsimp only
    split
    · rename_i test
      have inside : position.val < facts.val.length := by
        simpa only [UScalar.lt_equiv, Slice.len_val, Slice.length] using test
      step with Slice.index_usize_spec facts position inside as ⟨fact, factAt⟩
      rw [source_same_register]
      simp only [bind_ok]
      split
      · rename_i same
        simp only [decide_eq_true_eq] at same
        simp only [WP.spec_ok]
        rw [meaning, lookup_drop_step facts.val register position.val inside, ←factAt, if_pos same]
      · rename_i different
        simp only [decide_eq_true_eq] at different
        step with Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac)
          as ⟨next, nextAt⟩
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        have nextValue : next.val = position.val + 1 := by scalar_tac
        rw [nextValue, meaning, lookup_drop_step facts.val register position.val inside, ←factAt, if_neg different]
    · rename_i test
      have outside : ¬position.val < facts.val.length := by
        simpa only [UScalar.lt_equiv, Slice.len_val, Slice.length] using test
      simp only [WP.spec_ok]
      have complete : position.val = facts.val.length := by omega
      rw [meaning, complete, List.drop_length, lookup_nil]
  · exact ⟨by scalar_tac, by simp⟩

end Instruction.GuardCertificate
