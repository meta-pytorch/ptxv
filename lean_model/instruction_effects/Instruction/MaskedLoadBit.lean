/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Funs

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

theorem masked_load_low_bit (mask shifted : U8) (source : Nat)
    (hshift : shifted.bv = mask.bv >>> source) :
    shifted &&& 1#u8 = if mask.bv.getLsbD source then 1#u8 else 0#u8 := by
  apply UScalar.eq_of_val_eq
  change (shifted.bv &&& 1#8).toNat = _
  rw [BitVec.and_one_eq_setWidth_ofBool_getLsbD, hshift,
    BitVec.getLsbD_ushiftRight]
  cases hb : mask.bv.getLsbD source <;> simp

theorem masked_load_bit_selected (mask shifted : U8) (source : Nat)
    (hshift : shifted.bv = mask.bv >>> source) :
    (shifted &&& 1#u8 != 0#u8) = mask.bv.getLsbD source := by
  rw [masked_load_low_bit mask shifted source hshift]
  cases hb : mask.bv.getLsbD source <;> simp

end Instruction
