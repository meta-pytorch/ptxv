import RsqPtxBounds

set_option maxRecDepth 1000000
namespace GB300Math.RsqPtxBounds
open SqrtBounds (positiveNormalValue scaleExponent mantissaIndex)

theorem output_normal_range (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    63*2^23 ≤ Mufu.rsq (exponent*2^23+fraction) ∧
    Mufu.rsq (exponent*2^23+fraction) < 191*2^23 := by
  have ht : -63 ≤ scaleExponent exponent ∧ scaleExponent exponent ≤ 63 := by
    dsimp [scaleExponent];omega
  rw [RsqBounds.output_word_scaled exponent fraction he hf]
  split
  · omega
  · dsimp only
    split <;> omega

theorem subnormal_scaled_value (f : Nat) (hf : 0 < f ∧ f < 2^23) :
    positiveNormalValue ((f.log2+2)*2^23+(f*2^(23-f.log2)-2^23)) =
      (f : ℝ)*(2 : ℝ)^(-125 : Int) := by
  have hl : f.log2 ≤ 22 := by
    have h := (Nat.log2_lt (by omega : f ≠ 0)).mpr (show f < 2^23 by omega)
    omega
  have hlogs := (Nat.log2_eq_iff (by omega : f ≠ 0)).mp (show f.log2 = f.log2 by rfl)
  have hp : 2^f.log2 ≤ f := hlogs.1
  have hq : f < 2^(f.log2+1) := hlogs.2
  have hm : 2^23 ≤ f*2^(23-f.log2) ∧ f*2^(23-f.log2) < 2^24 := by
    have hpow : 2^f.log2*2^(23-f.log2) = 2^23 := by rw [← pow_add];congr;omega
    have hpow' : 2^(f.log2+1)*2^(23-f.log2) = 2^24 := by rw [← pow_add];congr;omega
    constructor
    · rw [← hpow]
      exact Nat.mul_le_mul_right _ hp
    · rw [← hpow']
      exact Nat.mul_lt_mul_of_pos_right hq (by positivity)
  unfold positiveNormalValue
  rw [RcpBounds.normal_decode _ _ (by omega) (by omega)]
  have hc : ((f*2^(23-f.log2)-2^23 : Nat) : ℚ) = (f : ℚ)*2^(23-f.log2)-2^23 := by
    rw [Nat.cast_sub hm.1]
    push_cast
    rfl
  rw [hc]
  push_cast
  rw [show (2^23+((f : ℝ)*2^(23-f.log2)-2^23)) = (f : ℝ)*2^(23-f.log2) by ring]
  rw [mul_assoc,← zpow_natCast,← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
  congr 2
  omega

theorem shift_output_value (e f : Nat) (he : 1 ≤ e ∧ e ≤ 242) (hf : f < 2^23) :
    positiveNormalValue ((e+12)*2^23+f) = positiveNormalValue (e*2^23+f)*2^12 := by
  unfold positiveNormalValue
  rw [RcpBounds.normal_decode _ _ (by omega) hf,RcpBounds.normal_decode _ _ (by omega) hf]
  push_cast
  have hp : (2 : ℝ)^((e : Int)+12-150) = (2 : ℝ)^((e : Int)-150)*2^12 := by
    rw [show (2 : ℝ)^12 = (2 : ℝ)^(12 : Int) by norm_num,← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    congr 1
    omega
  rw [hp]
  ring

theorem relative_rescale (x y e : ℝ) (hx : 0 < x)
    (hb : |y-1/Real.sqrt (x*2^24)| ≤ e/Real.sqrt (x*2^24)) :
    |y*2^12-1/Real.sqrt x| ≤ e/Real.sqrt x := by
  have hs : 0 < Real.sqrt x := Real.sqrt_pos.mpr hx
  have h24 : (2 : ℝ)^24 = ((2 : ℝ)^12)^2 := by norm_num
  rw [Real.sqrt_mul hx.le,h24,Real.sqrt_sq (by positivity)] at hb
  have hp := mul_le_mul_of_nonneg_right hb (show (0 : ℝ) ≤ 2^12 by positivity)
  have ha : |(y-1/(Real.sqrt x*2^12))*2^12| = |y-1/(Real.sqrt x*2^12)| * 2^12 := by rw [abs_mul,abs_of_pos (by positivity : (0 : ℝ) < 2^12)]
  rw [← ha] at hp
  have hid : (y-1/(Real.sqrt x*2^12))*2^12 = y*2^12-1/Real.sqrt x := by field_simp
  have hid' : e/(Real.sqrt x*2^12)*2^12 = e/Real.sqrt x := by field_simp
  rw [hid,hid'] at hp
  exact hp

#print axioms output_normal_range
#print axioms subnormal_scaled_value
#print axioms shift_output_value
#print axioms relative_rescale
end GB300Math.RsqPtxBounds
