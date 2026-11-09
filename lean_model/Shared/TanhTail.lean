import RcpBounds
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.Complex.ExponentialBounds

set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace GB300Math.TanhTail

theorem remainder (x : ℝ) : 1-Real.tanh x = 2/(Real.exp (2*x)+1) := by
  rw [Real.tanh_eq,Real.exp_neg,show 2*x=x+x by ring,Real.exp_add]
  have hp := Real.exp_pos x
  have hn : Real.exp x+(Real.exp x)⁻¹ ≠ 0 := ne_of_gt (by positivity)
  have hd : Real.exp x*Real.exp x+1 ≠ 0 := ne_of_gt (by positivity)
  field_simp
  ring

theorem exp_sixteen_lower : (20000000000 : ℝ)/2251 < Real.exp 16+1 := by
  have he : (27182818283/10000000000 : ℝ) ≤ Real.exp 1 := by linarith [Real.exp_one_gt_d9]
  have hp := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 27182818283/10000000000) he 16
  have heq : Real.exp 16 = (Real.exp 1)^16 := by norm_num [Real.exp_nat_mul]
  rw [←heq] at hp
  norm_num at hp ⊢
  linarith

theorem positive_tail (x : ℝ) (hx : 8 ≤ x) :
    |1-Real.tanh x| < 2251/10000000000 := by
  rw [remainder,abs_of_pos (by positivity)]
  have he := Real.exp_le_exp.mpr (show (16 : ℝ) ≤ 2*x by linarith)
  have hd : 0 < Real.exp (2*x)+1 := by positivity
  apply (div_lt_iff₀ hd).mpr
  nlinarith [exp_sixteen_lower]

theorem negative_tail (x : ℝ) (hx : x ≤ -8) :
    |(-1)-Real.tanh x| < 2251/10000000000 := by
  have h := positive_tail (-x) (by linarith)
  rw [Real.tanh_neg] at h
  have he : (-1)-Real.tanh x = -(1- -Real.tanh x) := by ring
  simpa only [he,abs_neg] using h


def inputWord (negative : Bool) (exponent fraction : Nat) : Nat :=
  (if negative then 2^31 else 0)+exponent*2^23+fraction

theorem tail_bits (negative : Bool) (exponent fraction : Nat)
    (he : 130 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    Mufu.tanh (inputWord negative exponent fraction) =
      some ((if negative then 2^31 else 0)+0x3f800000) := by
  have hex : Mufu.exponent (inputWord negative exponent fraction) = exponent := by
    cases negative <;> dsimp [inputWord,Mufu.exponent] <;> omega
  have hmag : Mufu.magnitude (inputWord negative exponent fraction) ≤ 0x7f800000 := by
    cases negative <;> dsimp [inputWord,Mufu.magnitude] <;> omega
  have hs : Mufu.sign (inputWord negative exponent fraction) = (if negative then 2^31 else 0) := by
    cases negative
    · simp only [inputWord,Bool.false_eq_true,ite_false,Nat.zero_add]
      exact RcpBounds.small_sign _ (by omega)
    · simp only [inputWord,ite_true,Nat.add_assoc]
      exact RcpBounds.high_sign _ (by omega)
  unfold Mufu.tanh
  split
  · omega
  · split
    · omega
    · split
      · rw [hs]
      · omega

theorem positive_input_lower (exponent fraction : Nat)
    (he : 130 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    (8 : ℝ) ≤ (RcpBounds.positiveNormalValue (exponent*2^23+fraction) : ℝ) := by
  have hs : (8 : ℚ) ≤ RcpBounds.exponentScale exponent := by
    have h := zpow_le_zpow_right₀ (by norm_num : (1 : ℚ) ≤ 2)
      (show (3 : Int) ≤ (exponent : Int)-127 by omega)
    norm_num [RcpBounds.exponentScale] at h ⊢
    exact h
  have hm : (1 : ℚ) ≤ RcpBounds.normalizedInput fraction := by
    dsimp [RcpBounds.normalizedInput]
    have h : (0 : ℚ) ≤ fraction := by positivity
    norm_num
    linarith
  have hp := mul_le_mul hm hs (by norm_num : (0 : ℚ) ≤ 8) (by linarith : 0 ≤ RcpBounds.normalizedInput fraction)
  rw [RcpBounds.input_decode_scaled exponent fraction (by omega) hf]
  exact_mod_cast (by simpa using hp : (8 : ℚ) ≤ RcpBounds.normalizedInput fraction*RcpBounds.exponentScale exponent)

theorem tail_bits_ideal_error (negative : Bool) (exponent fraction : Nat)
    (he : 130 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    let word := inputWord negative exponent fraction
    |(RcpBounds.normalValue ((Mufu.tanh word).getD 0) : ℝ)-
      Real.tanh (RcpBounds.normalValue word : ℝ)| < 2251/10000000000 := by
  dsimp only
  rw [tail_bits negative exponent fraction he hf]
  have hx := RcpBounds.value_sign (exponent*2^23+fraction) (by omega)
  have ho := RcpBounds.value_sign 0x3f800000 (by decide)
  have hone : RcpBounds.positiveNormalValue 0x3f800000 = 1 := by decide +kernel
  have hlo := positive_input_lower exponent fraction he hf
  cases negative
  · simp only [inputWord,Bool.false_eq_true,ite_false,Nat.zero_add,Option.getD_some,hx.1,ho.1,hone,Rat.cast_one]
    exact positive_tail _ hlo
  · simp only [inputWord,ite_true,Nat.add_assoc,Option.getD_some,hx.2,ho.2,hone,Rat.cast_neg,Rat.cast_one]
    exact negative_tail _ (by linarith)

theorem tail_error_at_eight : (225 : ℝ)/1000000000 < |1-Real.tanh 8| := by
  have he : Real.exp 1 ≤ (27182818286/10000000000 : ℝ) := by linarith [Real.exp_one_lt_d9]
  have hp := pow_le_pow_left₀ (le_of_lt (Real.exp_pos 1)) he 16
  have heq : Real.exp 16 = (Real.exp 1)^16 := by norm_num [Real.exp_nat_mul]
  rw [←heq] at hp
  have hu : Real.exp 16+1 < (2000000000 : ℝ)/225 := by norm_num at hp ⊢; linarith
  rw [remainder,abs_of_pos (by positivity)]
  norm_num only
  apply (lt_div_iff₀ (by positivity : 0 < Real.exp 16+1)).mpr
  nlinarith

#print axioms tail_error_at_eight
#print axioms positive_tail
#print axioms negative_tail
#print axioms tail_bits
#print axioms tail_bits_ideal_error
end GB300Math.TanhTail
