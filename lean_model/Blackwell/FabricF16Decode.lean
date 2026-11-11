/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricF16V2
import FabricF16ExactBounds
namespace FabricF16Decode
open FloatLib.Floats.Formats.BinaryInterchange

theorem units_value (word:Nat) (value:Int) (hu:FabricF16V2.units word=some value) :
    (Model.ofNatBits word:Model FloatFormat.binary16).toReal=(value:ℝ)/2^24 :=
  FabricF16ExactBounds.units_value word value hu

theorem word_frac (word : Nat) :
    (Model.ofNatBits word : Model FloatFormat.binary16).fracField = word%1024 := by
  rw [Model.fracField_eq_fracFieldImpl_apply]
  change (word % 65536) &&& (2^10-1) = word%1024
  rw [Nat.and_two_pow_sub_one_eq_mod]
  omega

theorem word_sign (word : Nat) (hw : word < 65536) :
    (Model.ofNatBits word : Model FloatFormat.binary16).signBit = decide (32768 ≤ word) := by
  rw [Model.signBit_eq_signBitImpl_apply]
  change (word%65536).testBit 15 = decide (32768≤word)
  rw [Nat.mod_eq_of_lt hw, Nat.testBit_eq_decide_div_mod_eq]
  congr 1
  apply propext
  change word/32768%2=1 ↔32768≤word
  omega

theorem units_canonical (word : Nat) (hw : word<65536) (hm : word%32768<31744) :
    FabricF16V2.units word=some (if 32768≤word then -(FabricF16V2.unit (word%32768) : Int) else (FabricF16V2.unit (word%32768) : Int)) := by
  have hs := word_sign word hw
  have hf := FabricF16Bounds.exp_word word
  have hfrac : word%1024=(word%32768)%1024 := by omega
  have he : word%32768/1024 ≠ 31 := by omega
  simp only [FabricF16V2.units,hw,↓reduceIte]
  simp only [Model.toDyadic?,show FloatFormat.binary16.isIEEE=true from rfl,↓reduceIte,
    Model.ieeeToDyadic?,Model.IEEE.isNaN,Model.IEEE.isInf]
  simp only [hf,word_frac,hfrac,hs,show FloatFormat.binary16.expAllOnesNat=31 from rfl,
    show FloatFormat.binary16.ieeeMinSubnormalExponent=(-24:Int) from rfl,
    show FloatFormat.binary16.ieeeNormalMantissaExpOffset=25 from rfl,
    show Model.pow2 FloatFormat.binary16.fracWidth=1024 from rfl,
    beq_eq_false_iff_ne.mpr he,Bool.false_and,Bool.false_or,Bool.false_eq_true,↓reduceIte]
  by_cases hz : word%32768/1024=0
  · by_cases hfz : word%32768%1024=0
    · have hwz : word%32768=0 := by omega
      simp [FabricF16V2.unit,hwz]
    · have hfz1 : word%1024 ≠ 0 := by simpa using hfz
      simp [hz,hfz1,FabricF16V2.unit]
  · have hepos : 1≤word%32768/1024 := by omega
    have hex : ((word%32768/1024:Int)-25)+24=(word%32768/1024:Int)-1 := by omega
    have heok : (word%32768/1024:Int)-25 ≥ -24 := by omega
    have hnat : ((word%32768/1024:Int)-1).toNat=word%32768/1024-1 := by omega
    simp [hz,heok,hex,hnat,FabricF16V2.unit,add_comm]


#print axioms units_value
#print axioms word_frac
#print axioms word_sign
#print axioms units_canonical
end FabricF16Decode
