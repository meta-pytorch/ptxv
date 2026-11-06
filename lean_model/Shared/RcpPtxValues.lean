import RcpPtxBounds
set_option maxRecDepth 10000

namespace GB300Math.RcpPtxBounds
open RcpBounds

theorem reciprocal_absolute_error (fraction : Nat) (hf : fraction < 2^23) :
    |normalizedOutput fraction - 1/normalizedInput fraction| ≤ 1/2^24 := by
  have h := interpolate_error fraction hf
  dsimp only at h
  have hlo : -(1 : ℚ)*(2^23+fraction) ≤
      1*((2^23+(fraction : ℚ))*(Mufu.interpolate MufuTables.rcp fraction : ℚ)-2^47) := by
    simp only [one_mul, neg_mul]
    exact_mod_cast h.1
  have hhi : 1*((2^23+(fraction : ℚ))*(Mufu.interpolate MufuTables.rcp fraction : ℚ)-2^47) ≤
      1*(2^23+(fraction : ℚ)) := by
    simp only [one_mul]
    exact_mod_cast h.2
  have hm : (0 : ℚ) < 2^23+(fraction : ℚ) := by positivity
  have he : normalizedOutput fraction - 1/normalizedInput fraction =
      (1*((2^23+(fraction : ℚ))*(Mufu.interpolate MufuTables.rcp fraction : ℚ)-2^47))/
      ((2^23+(fraction : ℚ))*2^24) := by
    dsimp [normalizedOutput,normalizedInput]
    field_simp
  rw [he,abs_div,abs_of_pos (mul_pos hm (by norm_num))]
  calc
    _ ≤ (1*(2^23+(fraction : ℚ)))/((2^23+(fraction : ℚ))*2^24) :=
      div_le_div_of_nonneg_right (abs_le.mpr ⟨by linarith,hhi⟩) (by positivity)
    _ = _ := by field_simp

theorem rcp_bits_absolute_error (fraction : Nat) (hf : fraction < 2^23) :
    |positiveNormalValue (Mufu.rcp (0x3f800000+fraction)) -
      1/positiveNormalValue (0x3f800000+fraction)| ≤ 1/2^24 := by
  rw [output_decode fraction hf,input_decode fraction hf]
  exact reciprocal_absolute_error fraction hf

theorem normalized_relative_error (fraction : Nat) (hf : fraction < 2^23) :
    |normalizedOutput fraction*normalizedInput fraction-1| ≤ 1/2^23 := by
  have habs := reciprocal_absolute_error fraction hf
  have hx : 1 ≤ normalizedInput fraction ∧ normalizedInput fraction < 2 := by
    dsimp [normalizedInput]
    have hfn : (0 : ℚ) ≤ fraction := by positivity
    have hfu : (fraction : ℚ) < 8388608 := by exact_mod_cast hf
    norm_num
    constructor <;> linarith
  have hne : normalizedInput fraction ≠ 0 := by linarith [hx.1]
  have he : normalizedOutput fraction*normalizedInput fraction-1 =
      (normalizedOutput fraction-1/normalizedInput fraction)*normalizedInput fraction := by
    field_simp
  rw [he,abs_mul,abs_of_pos (show 0 < normalizedInput fraction by linarith [hx.1])]
  have hm := mul_le_mul_of_nonneg_right habs (show 0 ≤ normalizedInput fraction by linarith [hx.1])
  norm_num at hm ⊢
  linarith [hx.2]

theorem rcp_bits_relative_error (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    |positiveNormalValue (Mufu.rcp (exponent*2^23+fraction))*
      positiveNormalValue (exponent*2^23+fraction)-1| ≤ 1/2^23 := by
  have ho := output_decode_scaled exponent fraction he hf
  rw [input_decode_scaled exponent fraction (by omega) hf]
  have hid : positiveNormalValue (Mufu.rcp (exponent*2^23+fraction))*
      (normalizedInput fraction*exponentScale exponent) =
      normalizedOutput fraction*normalizedInput fraction := by
    calc
      _ = (positiveNormalValue (Mufu.rcp (exponent*2^23+fraction))*exponentScale exponent)*normalizedInput fraction := by ring
      _ = _ := by rw [ho]
  rw [hid]
  exact normalized_relative_error fraction hf

theorem rcp_bits_scaled_absolute_error (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    |positiveNormalValue (Mufu.rcp (exponent*2^23+fraction))-
      1/positiveNormalValue (exponent*2^23+fraction)| ≤ (1/2^24)/exponentScale exponent := by
  have ho := output_decode_scaled exponent fraction he hf
  have hp : 0 < exponentScale exponent := by unfold exponentScale; positivity
  have hout := (eq_div_iff (ne_of_gt hp)).mpr ho
  rw [hout,input_decode_scaled exponent fraction (by omega) hf]
  have hid : normalizedOutput fraction/exponentScale exponent-
      1/(normalizedInput fraction*exponentScale exponent) =
      (normalizedOutput fraction-1/normalizedInput fraction)/exponentScale exponent := by ring
  rw [hid,abs_div,abs_of_pos hp]
  exact div_le_div_of_nonneg_right (reciprocal_absolute_error fraction hf) (le_of_lt hp)

theorem rcp_signed_relative_error (negative : Bool) (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    let word := (if negative then 2^31 else 0)+exponent*2^23+fraction
    |normalValue (Mufu.rcp word)*normalValue word-1| ≤ 1/2^23 := by
  let x := exponent*2^23+fraction
  have hx : x < 2^31 := by dsimp [x];omega
  have hy : Mufu.rcp x < 2^31 := by
    change Mufu.rcp (exponent*2^23+fraction) < 2^31
    rw [output_word_scaled exponent fraction he hf]
    generalize (Mufu.interpolate MufuTables.rcp fraction).toNat = z
    clear negative hx x hf fraction
    split <;> omega
  have vx := value_sign x hx
  have vy := value_sign (Mufu.rcp x) hy
  cases negative
  · simp only [Bool.false_eq_true,↓reduceIte,Nat.zero_add]
    change |normalValue (Mufu.rcp x)*normalValue x-1| ≤ _
    rw [vx.1,vy.1]
    exact rcp_bits_relative_error exponent fraction he hf
  · simp only [↓reduceIte]
    have hw : 2^31+exponent*2^23+fraction = 2^31+x := by dsimp [x];omega
    rw [hw,reciprocal_sign x (by dsimp [x];omega),vx.2,vy.2]
    simp only [neg_mul_neg]
    exact rcp_bits_relative_error exponent fraction he hf

theorem rcp_signed_scaled_absolute_error (negative : Bool) (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    let word := (if negative then 2^31 else 0)+exponent*2^23+fraction
    |normalValue (Mufu.rcp word)-1/normalValue word| ≤ (1/2^24)/exponentScale exponent := by
  let x := exponent*2^23+fraction
  have hx : x < 2^31 := by dsimp [x];omega
  have hy : Mufu.rcp x < 2^31 := by
    change Mufu.rcp (exponent*2^23+fraction) < 2^31
    rw [output_word_scaled exponent fraction he hf]
    generalize (Mufu.interpolate MufuTables.rcp fraction).toNat = z
    clear negative hx x hf fraction
    split <;> omega
  have vx := value_sign x hx
  have vy := value_sign (Mufu.rcp x) hy
  cases negative
  · simp only [Bool.false_eq_true,↓reduceIte,Nat.zero_add]
    change |normalValue (Mufu.rcp x)-1/normalValue x| ≤ _
    rw [vx.1,vy.1]
    exact rcp_bits_scaled_absolute_error exponent fraction he hf
  · simp only [↓reduceIte]
    have hw : 2^31+exponent*2^23+fraction = 2^31+x := by dsimp [x];omega
    rw [hw,reciprocal_sign x (by dsimp [x];omega),vx.2,vy.2]
    rw [one_div,inv_neg,sub_neg_eq_add]
    have hn : -positiveNormalValue (Mufu.rcp x)+(positiveNormalValue x)⁻¹ =
        -(positiveNormalValue (Mufu.rcp x)-(positiveNormalValue x)⁻¹) := by ring
    rw [hn,abs_neg]
    simpa only [one_div] using rcp_bits_scaled_absolute_error exponent fraction he hf

def reciprocalUlp (exponent fraction : Nat) : ℚ :=
  (if fraction = 0 then 2 else 1)*(1/2^24)/exponentScale exponent

theorem rcp_signed_one_ulp (negative : Bool) (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    let word := (if negative then 2^31 else 0)+exponent*2^23+fraction
    |normalValue (Mufu.rcp word)-1/normalValue word| ≤ reciprocalUlp exponent fraction := by
  have h := rcp_signed_scaled_absolute_error negative exponent fraction he hf
  apply le_trans h
  unfold reciprocalUlp
  have hp : 0 ≤ exponentScale exponent := by unfold exponentScale; positivity
  apply div_le_div_of_nonneg_right _ hp
  split_ifs <;> norm_num

theorem rcp_signed_one_ulp_real (negative : Bool) (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    let word := (if negative then 2^31 else 0)+exponent*2^23+fraction
    |(normalValue (Mufu.rcp word) : ℝ)-1/(normalValue word : ℝ)| ≤
      (reciprocalUlp exponent fraction : ℝ) := by
  exact_mod_cast rcp_signed_one_ulp negative exponent fraction he hf

#print axioms rcp_signed_one_ulp
#print axioms rcp_signed_one_ulp_real
#print axioms reciprocal_absolute_error
#print axioms rcp_bits_absolute_error
#print axioms normalized_relative_error
#print axioms rcp_bits_relative_error
#print axioms rcp_bits_scaled_absolute_error
#print axioms rcp_signed_relative_error
#print axioms rcp_signed_scaled_absolute_error
end GB300Math.RcpPtxBounds
