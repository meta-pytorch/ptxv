import MufuRcp32Signed
import FloatLib.Floats.Formats.BinaryInterchange.Algebraic.PowerProof
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace GB300Math.CycleIntegerTail
open FloatLib.Floats.Formats.BinaryInterchange

def tailWord (e : Fin 105) (f : Fin (2^23)) : Nat := (150+e.val)*2^23+f.val

theorem tail_rat (e : Fin 105) (f : Fin (2^23)) :
    (Model.ofNatBits (tailWord e f) : Model FloatFormat.binary32).toRat? =
      some (((2^23+f.val)*2^e.val : Nat) : Rat) := by
  let ei : Fin 254 := ⟨149+e.val, by have := e.isLt; omega⟩
  have h := MufuRcp32Bounds.decode_normal ei f
  have he : (ei.val:Int)-149 = (e.val:Int) := by dsimp [ei]; omega
  have hw : (1+ei.val)*2^23+f.val = tailWord e f := by simp [ei, tailWord]; omega
  rw [hw,he,zpow_natCast] at h
  simpa using h

theorem negative_tail_rat (e : Fin 105) (f : Fin (2^23)) :
    (Model.ofNatBits (2^31+tailWord e f) : Model FloatFormat.binary32).toRat? =
      some (-(((2^23+f.val)*2^e.val : Nat) : Rat)) := by
  rw [Nat.add_comm]
  exact RcpSigned.signed_decode _ _ (tail_rat e f)

theorem tail_bits (word : Nat)
    (he : 150 ≤ Mufu.exponent word ∧ Mufu.exponent word < 255) :
    Mufu.sincos word false = (if Mufu.sign word ≠ 0 then 2^31 else 0) ∧
      Mufu.sincos word true = 0x3f800000 := by
  have hn : Mufu.exponent word ≠ 255 := by omega
  have hh : (Mufu.exponent word : Int)-125 ≥ 25 := by omega
  have hz : Mufu.interpolate MufuTables.sin 0 = 0 := by decide +kernel
  constructor
  · simp [Mufu.sincos,hn,hh,hz]
  · simp only [Mufu.sincos,hn,hh,or_true,ite_false,ite_true]
    decide +kernel

theorem parameter_fields (negative : Bool) (e : Fin 105) (f : Fin (2^23)) :
    let word := (if negative then 2^31 else 0)+tailWord e f
    Mufu.exponent word = 150+e.val ∧
      Mufu.sign word = (if negative then 2^31 else 0) := by
  have he := e.isLt
  have hf := f.isLt
  have hw : tailWord e f < 2^31 := by unfold tailWord; omega
  cases negative
  · simp only [Bool.false_eq_true,ite_false,Nat.zero_add]
    exact ⟨by unfold Mufu.exponent tailWord; omega, RcpSigned.sign_zero _ hw⟩
  · simp only [ite_true]
    exact ⟨by unfold Mufu.exponent tailWord; omega, RcpSigned.sign_set _ hw⟩

theorem signed_bits (negative : Bool) (e : Fin 105) (f : Fin (2^23)) :
    let word := (if negative then 2^31 else 0)+tailWord e f
    Mufu.sincos word false = (if negative then 2^31 else 0) ∧
      Mufu.sincos word true = 0x3f800000 := by
  have hfields := parameter_fields negative e f
  have he := e.isLt
  have h := tail_bits ((if negative then 2^31 else 0)+tailWord e f) (by rw [hfields.1]; omega)
  rw [hfields.2] at h
  cases negative <;> simpa using h

theorem integer_ideal (n : Int) :
    Real.sin (2*Real.pi*n) = 0 ∧ Real.cos (2*Real.pi*n) = 1 := by
  constructor
  · have h := Real.sin_int_mul_pi (2*n)
    simpa only [Int.cast_mul,Int.cast_ofNat,mul_assoc,mul_comm,mul_left_comm] using h
  · simpa only [mul_assoc,mul_comm,mul_left_comm] using Real.cos_int_mul_two_pi n

theorem decoded_exact (negative : Bool) (e : Fin 105) (f : Fin (2^23)) :
    let word := (if negative then 2^31 else 0)+tailWord e f
    ∃ x : Rat,
      (Model.ofNatBits word : Model FloatFormat.binary32).toRat? = some x ∧
      (Model.ofNatBits (Mufu.sincos word false) : Model FloatFormat.binary32).toRat? = some 0 ∧
      (Model.ofNatBits (Mufu.sincos word true) : Model FloatFormat.binary32).toRat? = some 1 ∧
      Real.sin (2*Real.pi*(x:ℝ)) = 0 ∧ Real.cos (2*Real.pi*(x:ℝ)) = 1 := by
  have hbits := signed_bits negative e f
  have pos := tail_rat e f
  have neg := negative_tail_rat e f
  have zero : (Model.ofNatBits 0 : Model FloatFormat.binary32).toRat? = some 0 := by decide +kernel
  have nzero : (Model.ofNatBits (2^31) : Model FloatFormat.binary32).toRat? = some 0 := by decide +kernel
  have one : (Model.ofNatBits 0x3f800000 : Model FloatFormat.binary32).toRat? = some 1 := by decide +kernel
  cases negative
  · simp only [Bool.false_eq_true,ite_false,Nat.zero_add] at hbits ⊢
    refine ⟨_,pos,by rw [hbits.1]; exact zero,by rw [hbits.2]; exact one,?_⟩
    exact_mod_cast integer_ideal (((2^23+f.val)*2^e.val : Nat) : Int)
  · simp only [ite_true] at hbits ⊢
    refine ⟨_,neg,by rw [hbits.1]; exact nzero,by rw [hbits.2]; exact one,?_⟩
    exact_mod_cast integer_ideal (-(((2^23+f.val)*2^e.val : Nat) : Int))

theorem input_count : 2*105*2^23 = 1761607680 := by decide +kernel

#print axioms tail_rat
#print axioms negative_tail_rat
#print axioms tail_bits
#print axioms parameter_fields
#print axioms signed_bits
#print axioms integer_ideal
#print axioms decoded_exact
#print axioms input_count
end GB300Math.CycleIntegerTail
