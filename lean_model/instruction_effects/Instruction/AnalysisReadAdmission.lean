/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.OwnerModel
import Instruction.Code.Funs

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction.AnalysisReadAdmission

def PrivateReads (registers : List Reg) : Prop :=
  ∀ register ∈ registers, RegisterOwner.shared register.file = false

@[step]
theorem source_private (register : Reg) :
    register_owner.is_thread_private register
      ⦃ result => result = !RegisterOwner.shared register.file ⦄ := by
  rcases register with ⟨file, index⟩
  cases file <;> simp [register_owner.is_thread_private, RegisterOwner.shared, WP.spec_ok]

@[step]
theorem rust_loop (registers : Slice Reg) (index : Usize)
    (inside : index.val ≤ registers.length)
    (prior : ∀ position (bound : position < registers.length), position < index.val →
      RegisterOwner.shared registers.val[position].file = false) :
    analysis_knowledge.reads_are_thread_private_loop registers index
      ⦃ result => result = true ↔ PrivateReads registers.val ⦄ := by
  unfold analysis_knowledge.reads_are_thread_private_loop
  apply loop.spec_decr_nat (fun current : Usize => registers.length - current.val)
    (fun current => current.val ≤ registers.length ∧
      ∀ position (bound : position < registers.length), position < current.val →
        RegisterOwner.shared registers.val[position].file = false)
  · intro current invariant
    obtain ⟨bound, checked⟩ := invariant
    unfold analysis_knowledge.reads_are_thread_private_loop.body
    dsimp only
    split
    · rename_i remaining
      have currentBound : current.val < registers.length := by
        simpa only [UScalar.lt_equiv, Slice.len_val] using remaining
      step with Slice.index_usize_spec registers current currentBound as ⟨register, atRegister⟩
      step with source_private register as ⟨isPrivate, observed⟩
      cases isPrivate with
      | false =>
        simp only [Bool.false_eq_true, ite_false, WP.spec_ok]
        constructor
        · intro impossible; cases impossible
        · intro all
          have atPrivate := all register (by rw [atRegister]; exact List.getElem_mem currentBound)
          simp [atPrivate] at observed
      | true =>
        simp only [ite_true]
        step with Usize.add_spec (x := current) (y := 1#usize) (by scalar_tac)
          as ⟨next, nextEq⟩
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        intro position beforeEnd beforeNext
        by_cases previous : position < current.val
        · exact checked position beforeEnd previous
        · have same : position = current.val := by scalar_tac
          subst position
          rw [←atRegister]
          simpa using observed.symm
    · rename_i completed
      have complete : current.val = registers.length := by
        have outside : ¬current.val < registers.length := by
          simpa only [UScalar.lt_equiv, Slice.len_val] using completed
        omega
      simp only [WP.spec_ok, true_iff]
      intro register member
      obtain ⟨position, atBound, atRegister⟩ := List.mem_iff_getElem.mp member
      rw [←atRegister]
      exact checked position atBound (by rw [complete]; exact atBound)
  · exact ⟨inside, prior⟩

@[step]
theorem rust_reads_are_thread_private (registers : Slice Reg) :
    analysis_knowledge.reads_are_thread_private registers
      ⦃ result => result = true ↔ PrivateReads registers.val ⦄ := by
  unfold analysis_knowledge.reads_are_thread_private
  exact rust_loop registers 0#usize (by scalar_tac) (by simp)

theorem admitted_reads_are_private (registers : Slice Reg)
    (accepted : analysis_knowledge.reads_are_thread_private registers = .ok true) :
    PrivateReads registers.val := by
  have source := rust_reads_are_thread_private registers
  simpa only [accepted, WP.spec_ok, true_iff] using source

theorem shared_read_rejected (registers : Slice Reg) (register : Reg)
    (member : register ∈ registers.val)
    (shared : RegisterOwner.shared register.file = true) :
    analysis_knowledge.reads_are_thread_private registers ≠ .ok true := by
  intro accepted
  have allPrivate := admitted_reads_are_private registers accepted register member
  simp [shared] at allPrivate

end Instruction.AnalysisReadAdmission
