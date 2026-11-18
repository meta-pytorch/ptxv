/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AtomicRefinement
import Instruction.ProducerModel

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

@[step] theorem atomic_value_sources_spec (initial : alloc.vec.Vec Reg)
    (operation : AtomOp) (bytes : U8) (value : Val)
    (hindex : Bounded (atomicValueReads operation bytes value))
    (hsize : initial.val.length + (atomicValueReads operation bytes value).length ≤ Usize.max) :
    register_effect.atomic_value_sources initial operation bytes value ⦃ result =>
      result.val.map key = initial.val.map key ++ atomicValueReads operation bytes value ∧
      result.val.length = initial.val.length + (atomicValueReads operation bytes value).length ⦄ := by
  cases operation <;> first
  | simpa only [register_effect.atomic_value_sources, atomicValueReads] using
      vector_spec initial value hindex hsize
  | simpa only [register_effect.atomic_value_sources] using
      floating_sources_spec initial value bytes hindex hsize
  | simpa only [register_effect.atomic_value_sources, atomicValueReads] using
      slots_spec initial value hindex hsize

@[step] theorem atomic_sources_spec (initial : alloc.vec.Vec Reg)
    (operation : AtomOp) (bytes : U8) (value : Val) (replacement : Option Val)
    (hindex : Bounded (atomicValueReads operation bytes value ++
      replacementReads operation bytes value replacement))
    (hsize : initial.val.length + (atomicValueReads operation bytes value ++
      replacementReads operation bytes value replacement).length ≤ Usize.max) :
    register_effect.atomic_sources initial operation bytes value replacement ⦃ result =>
      result.val.map key = initial.val.map key ++ atomicValueReads operation bytes value ++
        replacementReads operation bytes value replacement ∧
      result.val.length = initial.val.length + (atomicValueReads operation bytes value ++
        replacementReads operation bytes value replacement).length ⦄ := by
  have hparts := (bounded_append _ _).mp hindex
  unfold register_effect.atomic_sources
  step with atomic_value_sources_spec initial operation bytes value hparts.1
    (by simp only [List.length_append] at hsize; omega) as ⟨middle, hmmap, hmlen⟩
  cases replacement with
  | some replacement =>
    step with slots_spec middle replacement hparts.2
      (by simp only [List.length_append, replacementReads] at hsize; omega)
      as ⟨result, hmap, hlength⟩
    simpa [hmmap, hmlen, replacementReads, List.length_append, Nat.add_assoc]
      using And.intro hmap hlength
  | none =>
    cases operation <;> first
    | (step with implicit_cas_spec middle bytes value hparts.2
        (by simp only [List.length_append, replacementReads] at hsize; omega)
        as ⟨result, hmap, hlength⟩
       simpa [hmmap, hmlen, replacementReads, List.length_append, Nat.add_assoc]
        using And.intro hmap hlength)
    | simpa [replacementReads, hmmap, hmlen] using And.intro hmmap hmlen

end Instruction
