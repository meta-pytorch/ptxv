/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AtomicSourcesRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

def registerObservation (registers : Option (alloc.vec.Vec Reg)) : Option (List Key) :=
  registers.map fun registers => registers.val.map key

theorem address_control_spec (address : Addr) (hindex : Bounded (addressReads address)) :
    (do let registers ← register_effect.address (alloc.vec.Vec.new Reg) address
        ok (some registers)) ⦃ result => registerObservation result = some (addressReads address) ⦄ := by
  have ha := addressReads_length address
  step with address_spec (alloc.vec.Vec.new Reg) address hindex
    (by simp only [empty_register_vector]; scalar_tac) as ⟨registers, hmap, hlength⟩
  simp [registerObservation, hmap, empty_register_vector]

@[step] theorem control_reads_spec (operation : Op)
    (hindex : optionBounded (controlReads operation)) :
    register_effect.memory_control_reads operation
      ⦃ result => registerObservation result = controlReads operation ⦄ := by
  cases operation <;>
    try simp [register_effect.memory_control_reads, controlReads, registerObservation]
  case Ld destination elements space bytes address sync signed nc =>
    simpa [registerObservation] using address_control_spec address hindex
  case Ld256 hi lo address sync mask nc =>
    simpa [registerObservation] using address_control_spec address hindex
  case St space bytes address value sync =>
    simpa [registerObservation] using address_control_spec address hindex
  case St256 lo hi address sync =>
    simpa [registerObservation] using address_control_spec address hindex
  case Atom operation space bytes address value destination sync predicate replacement =>
    simpa [registerObservation] using address_control_spec address hindex
  case StAsync address barrier value bytes =>
    have hparts : Bounded (operandReads address) ∧ Bounded (operandReads barrier) := by
      simpa only [optionBounded, controlReads, bounded_append] using hindex
    have ha := operandReads_length address
    have hb := operandReads_length barrier
    step with slots_spec (alloc.vec.Vec.new Reg) address hparts.1
      (by simp only [empty_register_vector]; scalar_tac) as ⟨middle, hmmap, hmlen⟩
    simp only [empty_register_vector, List.length_nil, Nat.zero_add] at hmlen
    step with slots_spec middle barrier hparts.2 (by scalar_tac) as ⟨registers, hmap, hlength⟩
    simp [registerObservation, hmap, hmmap, empty_register_vector, controlReads]

@[step] theorem data_reads_spec (operation : Op)
    (hindex : optionBounded (dataReads operation)) :
    register_effect.memory_data_reads operation
      ⦃ result => registerObservation result = dataReads operation ⦄ := by
  cases operation <;>
    try simp [register_effect.memory_data_reads, registerObservation, dataReads, empty_register_vector]
  case St space bytes address value sync =>
    have hv := storeValueReads_length false bytes value
    step with store_sources_spec (alloc.vec.Vec.new Reg) value bytes false hindex
      (by simp only [empty_register_vector]; scalar_tac) as ⟨registers, hmap, hlength⟩
    simp [registerObservation, hmap, empty_register_vector, dataReads]
  case StAsync address barrier value bytes =>
    have hv := storeValueReads_length true bytes value
    step with store_sources_spec (alloc.vec.Vec.new Reg) value bytes true hindex
      (by simp only [empty_register_vector]; scalar_tac) as ⟨registers, hmap, hlength⟩
    simp [registerObservation, hmap, empty_register_vector, dataReads]
  case St256 lo hi address sync =>
    have hparts : Bounded (consecutive lo 4) ∧ Bounded (consecutive hi 4) := by
      simpa only [optionBounded, dataReads, bounded_append] using hindex
    step with append_spec (alloc.vec.Vec.new Reg) lo 4#u16
      (consecutive_bounded lo 4 hparts.1)
      (by simp only [empty_register_vector]; scalar_tac) as ⟨middle, hmmap, hmlen⟩
    simp only [empty_register_vector, List.length_nil, Nat.zero_add] at hmlen
    step with append_spec middle hi 4#u16 (consecutive_bounded hi 4 hparts.2)
      (by scalar_tac) as ⟨registers, hmap, hlength⟩
    simp [registerObservation, hmap, hmmap, empty_register_vector, dataReads]
  case Atom operation space bytes address value destination sync predicate replacement =>
    have hv := atomicValueReads_length operation bytes value
    have hn := replacementReads_length operation bytes value replacement
    step with atomic_sources_spec (alloc.vec.Vec.new Reg) operation bytes value replacement hindex
      (by simp only [empty_register_vector, List.length_append]; scalar_tac)
      as ⟨registers, hmap, hlength⟩
    simp [registerObservation, hmap, empty_register_vector, dataReads]

end Instruction
