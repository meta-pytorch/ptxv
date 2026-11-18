/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SpanRefinement

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction

@[step] theorem address_spec (initial : alloc.vec.Vec Reg) (address : Addr)
    (hindex : Bounded (addressReads address))
    (hsize : initial.val.length + (addressReads address).length ≤ Usize.max) :
    register_effect.address initial address ⦃ result =>
      result.val.map key = initial.val.map key ++ addressReads address ∧
      result.val.length = initial.val.length + (addressReads address).length ⦄ := by
  rcases address with ⟨base, base2, offset⟩
  cases base with
  | none =>
    cases base2 with
    | none => simp [register_effect.address, addressReads]
    | some value =>
      simp only [register_effect.address, bind_ok]
      step with slots_spec initial value
        (by simpa [addressReads] using hindex)
        (by simpa [addressReads] using hsize) as ⟨result, hmap, hlength⟩
      simpa [addressReads] using And.intro hmap hlength
  | some left =>
    cases base2 with
    | none =>
      simp only [register_effect.address]
      step with slots_spec initial left
        (by simpa [addressReads] using hindex)
        (by simpa [addressReads] using hsize) as ⟨result, hmap, hlength⟩
      simpa [addressReads] using And.intro hmap hlength
    | some right =>
      have hparts : Bounded (operandReads left) ∧ Bounded (operandReads right) := by
        simpa [addressReads] using hindex
      have hlength : initial.val.length + (operandReads left).length +
          (operandReads right).length ≤ Usize.max := by
        simpa [addressReads, Nat.add_assoc] using hsize
      simp only [register_effect.address]
      step with slots_spec initial left hparts.1 (by omega) as ⟨middle, hmmap, hmlen⟩
      step with slots_spec middle right hparts.2 (by omega) as ⟨result, hmap, hrlen⟩
      simpa [addressReads, hmmap, hmlen, List.append_assoc, Nat.add_assoc]
        using And.intro hmap hrlen

end Instruction
