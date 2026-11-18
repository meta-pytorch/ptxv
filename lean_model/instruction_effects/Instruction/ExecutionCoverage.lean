/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ReadRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

def executionAtomicValueReads (operation : AtomOp) (bytes : U8) (value : Val) : List Key :=
  match operation with
  | .Exch => operandReads value
  | _ => atomicValueReads operation bytes value

def executionReplacementReads (operation : AtomOp) (bytes : U8)
    (compare : Val) (replacement : Option Val) : List Key :=
  match operation with
  | .Cas | .CasSpin => casReplacement bytes compare replacement
  | _ => []

def executionReads (operation : Op) : Option (List Key) :=
  match operation with
  | .Ld _ _ _ _ address _ _ _ | .Ld256 _ _ address _ _ _ => some (addressReads address)
  | .St _ bytes address value _ => some (addressReads address ++ storeValueReads false bytes value)
  | .StAsync address barrier value bytes =>
      some (operandReads address ++ operandReads barrier ++ storeValueReads true bytes value)
  | .St256 lo hi address _ => some (addressReads address ++ consecutive lo 4 ++ consecutive hi 4)
  | .Atom operation _ bytes address value _ _ _ replacement =>
      some (addressReads address ++ executionAtomicValueReads operation bytes value ++
        executionReplacementReads operation bytes value replacement)
  | _ => none

theorem scalar_reads_covered_by_vector (value : Val) (reg : Key)
    (hread : reg ∈ operandReads value) : reg ∈ vectorReads value := by
  cases value with
  | Reg first elements =>
    by_cases hf : first.file = .V
    · simp only [operandReads, scalarSlots, hf, ite_true, consecutive_mem] at hread
      simp only [vectorReads, hf, ite_true, consecutive_mem]
      exact ⟨hread.1, hread.2.1, by have hcount : 1 ≤ max elements.val 1 := Nat.le_max_right _ _; omega⟩
    · simpa [operandReads, vectorReads, hf] using hread
  | Imm _ | Const _ _ | SReg _ => simp [operandReads] at hread

theorem atomic_value_reads_covered (operation : AtomOp) (bytes : U8) (value : Val) (reg : Key)
    (hread : reg ∈ executionAtomicValueReads operation bytes value) :
    reg ∈ atomicValueReads operation bytes value := by
  cases operation with
  | Exch => exact scalar_reads_covered_by_vector value reg hread
  | _ => exact hread

theorem replacement_reads_covered (operation : AtomOp) (bytes : U8)
    (compare : Val) (replacement : Option Val) (reg : Key)
    (hread : reg ∈ executionReplacementReads operation bytes compare replacement) :
    reg ∈ replacementReads operation bytes compare replacement := by
  cases operation <;> cases replacement <;>
    simp_all [executionReplacementReads, replacementReads, casReplacement]

theorem execution_reads_covered (operation : Op) (reg : Key)
    (hread : reg ∈ (executionReads operation).getD []) :
    reg ∈ (memoryReads operation).getD [] := by
  cases operation <;> try (solve | simpa [executionReads, memoryReads] using hread)
  case Atom operation space bytes address value destination sync predicate replacement =>
    simp only [executionReads, memoryReads, Option.getD_some, List.mem_append] at hread ⊢
    rcases hread with (haddress | hvalue) | hreplacement
    · exact Or.inl (Or.inl haddress)
    · exact Or.inl (Or.inr (atomic_value_reads_covered operation bytes value reg hvalue))
    · exact Or.inr (replacement_reads_covered operation bytes value replacement reg hreplacement)

theorem actual_reads_covered_spec (operation : Op)
    (hindex : optionBounded (memoryReads operation)) :
    register_effect.memory_reads operation ⦃ result =>
      ∀ reg, reg ∈ (executionReads operation).getD [] →
        reg ∈ (registerObservation result).getD [] ⦄ := by
  step with memory_reads_spec operation hindex as ⟨result, hobservation⟩
  rw [hobservation]
  apply execution_reads_covered operation
  assumption

end Instruction
