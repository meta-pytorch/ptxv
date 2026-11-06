import Fp64RsqFloatLib
import MufuRcp32Signed
set_option maxRecDepth 12000
namespace GB300Math.Rsq64Classification
open FloatLib.Floats.Formats.BinaryInterchange

theorem high_zero_exponent (w : Nat) (hw : w < 2^32) (he : w/2^20%2048=0) :
    Fp64Seed.highWord true w = (w &&& 0x80000000) ||| 0x7ff00000 := by
  unfold Fp64Seed.highWord
  rw [Nat.mod_eq_of_lt hw]
  dsimp only
  rw [he]
  simp

theorem high_nan (w : Nat) (hw : w < 2^32) (he : w/2^20%2048=2047)
    (hf : w%2^20 ≠ 0) : Fp64Seed.highWord true w = 0x7fffffff := by
  unfold Fp64Seed.highWord
  rw [Nat.mod_eq_of_lt hw]
  dsimp only
  rw [he]
  simp only [hf,ne_eq,not_false_eq_true,and_self,ite_true]

theorem high_infinity (w : Nat) (hw : w < 2^32) (he : w/2^20%2048=2047)
    (hf : w%2^20=0) : Fp64Seed.highWord true w = if w &&& 0x80000000 = 0 then 0 else 0x7fffffff := by
  unfold Fp64Seed.highWord
  rw [Nat.mod_eq_of_lt hw]
  dsimp only
  rw [he,hf]
  split_ifs <;> simp_all

theorem zero_exponent_full_results (high : Fin (2^20)) (low : Fin (2^32)) :
    Fp64Seed.evaluate true (high.val*2^32+low.val) = 0x7ff0000000000000 ∧
    Fp64Seed.evaluate true ((2^31+high.val)*2^32+low.val) = 0xfff0000000000000 := by
  have hh:=high.isLt
  have hl:=low.isLt
  have hsp : high.val &&& 0x80000000 = 0 := RcpSigned.sign_zero high.val (by omega)
  have hsn : (2^31+high.val) &&& 0x80000000 = 2^31 := RcpSigned.sign_set high.val (by omega)
  constructor
  · rw [Fp64Seed.ignores_low_word true _ _ hl,high_zero_exponent _ (by omega) (by omega),hsp]
    decide +kernel
  · rw [Fp64Seed.ignores_low_word true _ _ hl,high_zero_exponent _ (by omega) (by omega),hsn]
    decide +kernel

theorem infinity_highword_full_results (low : Fin (2^32)) :
    Fp64Seed.evaluate true (0x7ff00000*2^32+low.val) = 0 ∧
    Fp64Seed.evaluate true (0xfff00000*2^32+low.val) = 0x7fffffff00000000 := by
  have hl:=low.isLt
  constructor
  · rw [Fp64Seed.ignores_low_word true _ _ hl,high_infinity _ (by decide) (by decide) (by decide)]
    decide +kernel
  · rw [Fp64Seed.ignores_low_word true _ _ hl,high_infinity _ (by decide) (by decide) (by decide)]
    decide +kernel

theorem nan_highword_full_results (high : Fin (2^20)) (low : Fin (2^32)) (nonzero : high.val ≠ 0) :
    Fp64Seed.evaluate true ((0x7ff00000+high.val)*2^32+low.val) = 0x7fffffff00000000 ∧
    Fp64Seed.evaluate true ((0xfff00000+high.val)*2^32+low.val) = 0x7fffffff00000000 := by
  have hh:=high.isLt
  have hl:=low.isLt
  have hfp : (0x7ff00000+high.val)%2^20 ≠ 0 := by omega
  have hfn : (0xfff00000+high.val)%2^20 ≠ 0 := by omega
  constructor
  · rw [Fp64Seed.ignores_low_word true _ _ hl,high_nan _ (by omega) (by omega) hfp]
  · rw [Fp64Seed.ignores_low_word true _ _ hl,high_nan _ (by omega) (by omega) hfn]

theorem canonical_nan_floatlib :
    (Model.ofNatBits 0x7fffffff00000000 : Model FloatFormat.binary64).toRat? = none := by
  decide +kernel

theorem low_payload_nan_maps_to_zero :
    (Model.ofNatBits 0x7ff0000000000001 : Model FloatFormat.binary64).toRat? = none ∧
    Fp64Seed.evaluate true 0x7ff0000000000001 = 0 := by decide +kernel

theorem negative_nonzero_exponent (w : Nat) (hw : w < 2^32)
    (he : w/2^20%2048 ≠ 0) (hs : w &&& 0x80000000 ≠ 0) :
    Fp64Seed.highWord true w = 0x7fffffff := by
  unfold Fp64Seed.highWord
  rw [Nat.mod_eq_of_lt hw]
  dsimp only
  norm_num only [Nat.reducePow] at he hs ⊢
  by_cases hn : w/1048576%2048 = 2047 ∧ w%1048576 ≠ 0
  · rw [ite_eq_left hn]
  · rw [ite_eq_right hn,ite_eq_right he]
    simp only [true_and]
    rw [ite_eq_left hs]

theorem negative_normal_bits (e : Fin 2046) (f : Fin (2^52)) :
    Fp64Seed.evaluate true (2^63+(1+e.val)*2^52+f.val) = 0x7fffffff00000000 := by
  let high := 2^31+(1+e.val)*2^20+f.val/2^32
  have he := e.isLt
  have hf := f.isLt
  have wh : (2^63+(1+e.val)*2^52+f.val)/2^32 = high := by dsimp [high]; omega
  have hs : high &&& 0x80000000 ≠ 0 := by
    have h := RcpSigned.sign_set ((1+e.val)*2^20+f.val/2^32) (by omega)
    have hh : high &&& 0x80000000 = 2^31 := by
      simpa only [high,Nat.add_assoc,Mufu.sign] using h
    rw [hh]
    decide
  rw [Fp64Seed.evaluate,wh,negative_nonzero_exponent high (by dsimp [high]; omega)
    (by dsimp [high]; omega) hs]

theorem negative_normal_decode (e : Fin 2046) (f : Fin (2^52)) :
    (Model.ofNatBits (Fp64Seed.evaluate true (2^63+(1+e.val)*2^52+f.val)) :
      Model FloatFormat.binary64).toRat? = none := by
  rw [negative_normal_bits]
  exact canonical_nan_floatlib

#print axioms negative_nonzero_exponent
#print axioms negative_normal_bits
#print axioms negative_normal_decode

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
end GB300Math.Rsq64Classification
