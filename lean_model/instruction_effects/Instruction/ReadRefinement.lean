/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MemoryRefinement
import Instruction.ControlDataRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

theorem memory_reads_partition (operation : Op) :
    memoryReads operation = (do
      let control ← controlReads operation
      let data ← dataReads operation
      pure (control ++ data)) := by
  cases operation <;> simp [memoryReads, controlReads, dataReads, List.append_assoc]

@[step] theorem memory_reads_spec (operation : Op)
    (hindex : optionBounded (memoryReads operation)) :
    register_effect.memory_reads operation
      ⦃ result => registerObservation result = memoryReads operation ⦄ := by
  cases operation <;> try (solve | simp [register_effect.memory_reads,
    memoryReads, registerObservation])
  case Ld destination elements space bytes address sync signed nc =>
    step with load_reads_spec destination elements bytes space address sync signed nc hindex
      as ⟨result, registers, heq, hmeaning, hlength⟩
    simp [registerObservation, heq, memoryReads, hmeaning]
  case Ld256 hi lo address sync mask nc =>
    have ha := addressReads_length address
    simp only [register_effect.memory_reads]
    step with address_spec (alloc.vec.Vec.new Reg) address hindex
      (by simp only [empty_register_vector]; scalar_tac) as ⟨registers, hmap, hlength⟩
    simp [registerObservation, hmap, empty_register_vector, memoryReads]
  case St space bytes address value sync =>
    step with store_reads_spec space bytes address value sync hindex
      as ⟨result, registers, heq, hmeaning, hlength⟩
    simp [registerObservation, heq, memoryReads, hmeaning]
  case StAsync address barrier value bytes =>
    step with async_store_reads_spec address barrier value bytes hindex
      as ⟨result, registers, heq, hmeaning⟩
    simp [registerObservation, heq, memoryReads, hmeaning]
  case St256 lo hi address sync =>
    have hparts : (Bounded (addressReads address) ∧ Bounded (consecutive lo 4)) ∧
        Bounded (consecutive hi 4) := by
      simpa only [optionBounded, memoryReads, bounded_append] using hindex
    have ha := addressReads_length address
    simp only [register_effect.memory_reads]
    step with address_spec (alloc.vec.Vec.new Reg) address hparts.1.1
      (by simp only [empty_register_vector]; scalar_tac) as ⟨first, hfmap, hflen⟩
    simp only [empty_register_vector, List.length_nil, Nat.zero_add] at hflen
    step with append_spec first lo 4#u16 (consecutive_bounded lo 4 hparts.1.2)
      (by scalar_tac) as ⟨middle, hmmap, hmlen⟩
    step with append_spec middle hi 4#u16 (consecutive_bounded hi 4 hparts.2)
      (by scalar_tac) as ⟨registers, hmap, hlength⟩
    simp [registerObservation, hmap, hmmap, hfmap, empty_register_vector, memoryReads]
  case Atom operation space bytes address value destination sync predicate replacement =>
    have hparts : (Bounded (addressReads address) ∧
        Bounded (atomicValueReads operation bytes value)) ∧
        Bounded (replacementReads operation bytes value replacement) := by
      simpa only [optionBounded, memoryReads, bounded_append] using hindex
    have ha := addressReads_length address
    have hv := atomicValueReads_length operation bytes value
    have hn := replacementReads_length operation bytes value replacement
    simp only [register_effect.memory_reads]
    step with address_spec (alloc.vec.Vec.new Reg) address hparts.1.1
      (by simp only [empty_register_vector]; scalar_tac) as ⟨middle, hmmap, hmlen⟩
    simp only [empty_register_vector, List.length_nil, Nat.zero_add] at hmlen
    step with atomic_sources_spec middle operation bytes value replacement
      (by simpa only [bounded_append] using And.intro hparts.1.2 hparts.2)
      (by simp only [List.length_append]; scalar_tac) as ⟨registers, hmap, hlength⟩
    simp [registerObservation, hmap, hmmap, empty_register_vector, memoryReads, List.append_assoc]

end Instruction
