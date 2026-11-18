/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.StoreSourceRefinement
import Instruction.AddressRefinement
import Instruction.Bounds

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

@[step] theorem load_reads_spec (destination : Reg) (elements bytes : U8)
    (space : Space) (address : Addr) (sync : Option MemSync) (signed nc : Bool)
    (hindex : Bounded (addressReads address)) :
    register_effect.memory_reads (.Ld destination elements space bytes address sync signed nc)
      ⦃ result => ∃ registers, result = some registers ∧
        registers.val.map key = addressReads address ∧
        registers.val.length = (addressReads address).length ⦄ := by
  simp only [register_effect.memory_reads]
  step with address_spec (alloc.vec.Vec.new Reg) address hindex
    (by have h := addressReads_length address
        simp only [empty_register_vector]
        scalar_tac) as ⟨registers, hmap, hlength⟩
  refine ⟨registers, rfl, ?_, ?_⟩
  · simpa [empty_register_vector] using hmap
  · simpa [empty_register_vector] using hlength

@[step] theorem store_reads_spec (space : Space) (bytes : U8)
    (address : Addr) (value : Val) (sync : Option MemSync)
    (hindex : Bounded (addressReads address ++ storeValueReads false bytes value)) :
    register_effect.memory_reads (.St space bytes address value sync)
      ⦃ result => ∃ registers, result = some registers ∧
        registers.val.map key = addressReads address ++ storeValueReads false bytes value ∧
        registers.val.length = (addressReads address ++ storeValueReads false bytes value).length ⦄ := by
  have hparts := (bounded_append _ _).mp hindex
  have ha := addressReads_length address
  have hv := storeValueReads_length false bytes value
  simp only [register_effect.memory_reads]
  step with address_spec (alloc.vec.Vec.new Reg) address hparts.1
    (by simp only [empty_register_vector]
        scalar_tac) as ⟨middle, hmmap, hmlen⟩
  simp only [empty_register_vector,
    List.length_nil, Nat.zero_add] at hmlen
  step with store_sources_spec middle value bytes false hparts.2 (by scalar_tac) as ⟨registers, hmap, hlength⟩
  refine ⟨registers, rfl, ?_, ?_⟩
  · simpa [hmmap, empty_register_vector]
      using hmap
  · simpa [hmlen] using hlength

@[step] theorem async_store_reads_spec (address barrier value : Val) (bytes : U8)
    (hindex : Bounded (operandReads address ++ operandReads barrier ++ storeValueReads true bytes value)) :
    register_effect.memory_reads (.StAsync address barrier value bytes)
      ⦃ result => ∃ registers, result = some registers ∧
        registers.val.map key = operandReads address ++ operandReads barrier ++ storeValueReads true bytes value ⦄ := by
  have hparts := (bounded_append _ _).mp hindex
  have hfirst := (bounded_append _ _).mp hparts.1
  have ha := operandReads_length address
  have hb := operandReads_length barrier
  have hv := storeValueReads_length true bytes value
  simp only [register_effect.memory_reads]
  step with slots_spec (alloc.vec.Vec.new Reg) address hfirst.1
    (by simp only [empty_register_vector]
        scalar_tac) as ⟨first, hfmap, hflen⟩
  simp only [empty_register_vector,
    List.length_nil, Nat.zero_add] at hflen
  step with slots_spec first barrier hfirst.2 (by scalar_tac) as ⟨middle, hmmap, hmlen⟩
  step with store_sources_spec middle value bytes true hparts.2 (by scalar_tac) as ⟨registers, hmap, hlength⟩
  refine ⟨registers, rfl, ?_⟩
  simpa [hmmap, hfmap, empty_register_vector]
    using hmap

end Instruction
