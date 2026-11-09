import MufuRcp64Classification
set_option maxRecDepth 12000
namespace GB300Math.Rcp64Bounds
open FloatLib.Floats.Formats.BinaryInterchange

theorem high_zero_exponent (w : Nat) (hw : w < 2^32) (he : w/2^20%2048=0) :
    Fp64Seed.highWord false w = (w &&& 0x80000000) ||| 0x7ff00000 := by
  unfold Fp64Seed.highWord
  rw [Nat.mod_eq_of_lt hw]
  dsimp only
  rw [he]
  simp

theorem high_nan (w : Nat) (hw : w < 2^32) (he : w/2^20%2048=2047)
    (hf : w%2^20 ≠ 0) : Fp64Seed.highWord false w = 0x7fffffff := by
  unfold Fp64Seed.highWord
  rw [Nat.mod_eq_of_lt hw]
  dsimp only
  rw [he]
  simp only [hf,ne_eq,not_false_eq_true,and_self,ite_true]

theorem high_infinity (w : Nat) (hw : w < 2^32) (he : w/2^20%2048=2047)
    (hf : w%2^20=0) : Fp64Seed.highWord false w = w &&& 0x80000000 := by
  unfold Fp64Seed.highWord
  rw [Nat.mod_eq_of_lt hw]
  dsimp only
  rw [he,hf]
  simp

theorem zero_exponent_full_results (high : Fin (2^20)) (low : Fin (2^32)) :
    Fp64Seed.evaluate false (high.val*2^32+low.val) = 0x7ff0000000000000 ∧
    Fp64Seed.evaluate false ((2^31+high.val)*2^32+low.val) = 0xfff0000000000000 := by
  have hh:=high.isLt
  have hl:=low.isLt
  have hsp : high.val &&& 0x80000000 = 0 := RcpSigned.sign_zero high.val (by omega)
  have hsn : (2^31+high.val) &&& 0x80000000 = 2^31 := RcpSigned.sign_set high.val (by omega)
  constructor
  · rw [Fp64Seed.ignores_low_word false _ _ hl,high_zero_exponent _ (by omega) (by omega),hsp]
    decide +kernel
  · rw [Fp64Seed.ignores_low_word false _ _ hl,high_zero_exponent _ (by omega) (by omega),hsn]
    decide +kernel

theorem infinity_highword_full_results (low : Fin (2^32)) :
    Fp64Seed.evaluate false (0x7ff00000*2^32+low.val) = 0 ∧
    Fp64Seed.evaluate false (0xfff00000*2^32+low.val) = 2^63 := by
  have hl:=low.isLt
  constructor
  · rw [Fp64Seed.ignores_low_word false _ _ hl,high_infinity _ (by decide) (by decide) (by decide)]
    decide +kernel
  · rw [Fp64Seed.ignores_low_word false _ _ hl,high_infinity _ (by decide) (by decide) (by decide)]
    decide +kernel

theorem nan_highword_full_results (high : Fin (2^20)) (low : Fin (2^32)) (nonzero : high.val ≠ 0) :
    Fp64Seed.evaluate false ((0x7ff00000+high.val)*2^32+low.val) = 0x7fffffff00000000 ∧
    Fp64Seed.evaluate false ((0xfff00000+high.val)*2^32+low.val) = 0x7fffffff00000000 := by
  have hh:=high.isLt
  have hl:=low.isLt
  have hfp : (0x7ff00000+high.val)%2^20 ≠ 0 := by omega
  have hfn : (0xfff00000+high.val)%2^20 ≠ 0 := by omega
  constructor
  · rw [Fp64Seed.ignores_low_word false _ _ hl,high_nan _ (by omega) (by omega) hfp]
  · rw [Fp64Seed.ignores_low_word false _ _ hl,high_nan _ (by omega) (by omega) hfn]

theorem canonical_nan_floatlib :
    (Model.ofNatBits 0x7fffffff00000000 : Model FloatFormat.binary64).toRat? = none := by
  decide +kernel

theorem low_payload_nan_maps_to_zero :
    (Model.ofNatBits 0x7ff0000000000001 : Model FloatFormat.binary64).toRat? = none ∧
    Fp64Seed.evaluate false 0x7ff0000000000001 = 0 := by decide +kernel

theorem complete_encoding_count :
    2*2046*2^52+2*2^52+2*2^52=2^64 := by decide +kernel

#print axioms high_zero_exponent
#print axioms high_nan
#print axioms high_infinity
#print axioms zero_exponent_full_results
#print axioms infinity_highword_full_results
#print axioms nan_highword_full_results
#print axioms canonical_nan_floatlib
#print axioms low_payload_nan_maps_to_zero
#print axioms complete_encoding_count
end GB300Math.Rcp64Bounds
