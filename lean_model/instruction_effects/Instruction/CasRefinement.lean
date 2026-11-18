/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AtomicRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

@[step] theorem cas_sources_spec (initial : alloc.vec.Vec Reg) (bytes : U8)
    (compare : Val) (replacement : Option Val)
    (hindex : Bounded (operandReads compare ++ casReplacement bytes compare replacement))
    (hsize : initial.val.length +
      (operandReads compare ++ casReplacement bytes compare replacement).length ≤ Usize.max) :
    register_effect.atomic_sources initial .Cas bytes compare replacement ⦃ result =>
      result.val.map key = initial.val.map key ++ operandReads compare ++
        casReplacement bytes compare replacement ∧
      result.val.length = initial.val.length +
        (operandReads compare ++ casReplacement bytes compare replacement).length ⦄ := by
  have hparts := (bounded_append _ _).mp hindex
  simp only [register_effect.atomic_sources, register_effect.atomic_value_sources]
  step with slots_spec initial compare hparts.1
    (by simp only [List.length_append] at hsize; omega) as ⟨middle, hmmap, hmlen⟩
  cases replacement with
  | none =>
    step with implicit_cas_spec middle bytes compare hparts.2
      (by simp only [List.length_append] at hsize; omega) as ⟨result, hmap, hlength⟩
    simpa [hmmap, hmlen, List.length_append, Nat.add_assoc] using And.intro hmap hlength
  | some value =>
    step with slots_spec middle value
      (by simpa [casReplacement] using hparts.2)
      (by simp only [List.length_append, casReplacement] at hsize; omega)
      as ⟨result, hmap, hlength⟩
    simpa [hmmap, hmlen, casReplacement, List.length_append, Nat.add_assoc]
      using And.intro hmap hlength

@[step] theorem cas_reads_spec (space : Space) (bytes : U8) (address : Addr)
    (compare : Val) (destination : Option Reg) (sync : MemSync)
    (predicate : Option Reg) (replacement : Option Val)
    (hindex : Bounded (addressReads address ++ operandReads compare ++
      casReplacement bytes compare replacement)) :
    register_effect.memory_reads
      (.Atom .Cas space bytes address compare destination sync predicate replacement)
      ⦃ result => ∃ registers, result = some registers ∧
        registers.val.map key = addressReads address ++ operandReads compare ++
          casReplacement bytes compare replacement ⦄ := by
  have hparts := (bounded_append _ _).mp hindex
  have hfirst := (bounded_append _ _).mp hparts.1
  have hdata : Bounded (operandReads compare ++ casReplacement bytes compare replacement) :=
    (bounded_append _ _).mpr ⟨hfirst.2, hparts.2⟩
  have ha := addressReads_length address
  have hc := operandReads_length compare
  have hn := casReplacement_length bytes compare replacement
  simp only [register_effect.memory_reads]
  step with address_spec (alloc.vec.Vec.new Reg) address hfirst.1
    (by simp only [empty_register_vector]; scalar_tac) as ⟨middle, hmmap, hmlen⟩
  simp only [empty_register_vector, List.length_nil, Nat.zero_add] at hmlen
  step with cas_sources_spec middle bytes compare replacement hdata
    (by simp only [List.length_append]; scalar_tac) as ⟨registers, hmap, hlength⟩
  refine ⟨registers, rfl, ?_⟩
  simpa [hmmap, empty_register_vector] using hmap

end Instruction
