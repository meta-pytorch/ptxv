import RsqBounds
import Fp64SeedBounds

set_option maxRecDepth 1000000
set_option maxHeartbeats 10000000

namespace GB300Math.Fp64RsqBounds

theorem rsq_coefficients : SeedTables.rsq = MufuTables.rsq.coefficients := by decide +kernel

theorem rsq_interpolate (index : Nat) :
    Fp64Seed.interpolate true index = (Mufu.interpolate MufuTables.rsq index).toNat := by
  simp [Fp64Seed.interpolate,Mufu.interpolate,MufuTables.rsq,rsq_coefficients]

def mantissaIndex (exponent fraction : Nat) : Nat := (exponent+1)%2*2^23+fraction*8

def scaleExponent (exponent : Nat) : Int := ((exponent : Int)-1023-((exponent+1)%2 : Nat))/2

noncomputable def outputScale (exponent : Nat) : ℝ := (2 : ℝ)^scaleExponent exponent

def mantissa (index : Nat) : Nat := if index = 0 then 2^24 else Fp64Seed.interpolate true index

noncomputable def normalizedOutput (index : Nat) : ℝ := ((mantissa index/8 : Nat) : ℝ)/2^21

noncomputable def normalizedInput (exponent fraction low : Nat) : ℝ :=
  (1+(fraction : ℝ)/2^20+(low : ℝ)/2^52)*2^((exponent+1)%2)

noncomputable def positiveNormalValue (word : Nat) : ℝ :=
  (Fp64SeedBounds.positiveNormalValue word : ℝ)

theorem index_properties (exponent fraction : Nat) (hf : fraction < 2^20) :
    mantissaIndex exponent fraction < 2^24 ∧
    mantissaIndex exponent fraction%2^23 = fraction*8 ∧
    mantissaIndex exponent fraction/2^23 = (exponent+1)%2 := by
  dsimp [mantissaIndex]
  omega

theorem mantissa_range (index : Nat) (hi : index < 2^24) :
    2^23 ≤ mantissa index ∧ mantissa index ≤ 2^24 := by
  by_cases hz : index = 0
  · norm_num [mantissa,hz]
  · have h := RsqBounds.interpolate_error index hi
    simp only [mantissa,ite_eq_right hz,rsq_interpolate]
    omega

theorem normalized_rounding (index : Nat) (hi : index < 2^24) :
    0 ≤ normalizedOutput index ∧ normalizedOutput index ≤ 1 ∧
    0 ≤ RsqBounds.normalizedOutput index-normalizedOutput index ∧
    RsqBounds.normalizedOutput index-normalizedOutput index ≤ 7/2^24 := by
  have hb := mantissa_range index hi
  let z := mantissa index
  have hq : z/8 ≤ 2^21 := by dsimp [z];omega
  have he : (z/8 : Nat)*8+z%8 = z := by omega
  have heq : ((z/8 : Nat) : ℝ)*8+(z%8 : Nat) = z := by exact_mod_cast he
  have hbq : ((z%8 : Nat) : ℝ) ≤ 7 := by exact_mod_cast (show z%8 ≤ 7 by omega)
  have hqq : ((z/8 : Nat) : ℝ) ≤ 2^21 := by exact_mod_cast hq
  have hy0 : RsqBounds.normalizedOutput index = (z : ℝ)/2^24 := by
    by_cases hz : index = 0
    · norm_num [RsqBounds.normalizedOutput,z,mantissa,hz]
    · have hr := RsqBounds.interpolate_error index hi
      have he : ((Mufu.interpolate MufuTables.rsq index).toNat : Int) = Mufu.interpolate MufuTables.rsq index := by omega
      have he' : ((Mufu.interpolate MufuTables.rsq index).toNat : ℝ) = (Mufu.interpolate MufuTables.rsq index : ℝ) := by exact_mod_cast he
      simp only [RsqBounds.normalizedOutput,ite_eq_right hz,z,mantissa,rsq_interpolate,he']
  rw [hy0]
  dsimp only [normalizedOutput]
  change 0 ≤ ((z/8 : Nat) : ℝ)/2^21 ∧ ((z/8 : Nat) : ℝ)/2^21 ≤ 1 ∧ _
  norm_num only [Nat.cast_mul, Nat.cast_ofNat, OfNat.ofNat] at heq hbq hqq
  norm_num
  refine ⟨by positivity,by linarith,?_,?_⟩
  · have h0 : (0 : ℝ) ≤ (z%8 : Nat) := by positivity
    linarith
  · linarith

theorem normalized_input_bounds (exponent fraction low : Nat)
    (hf : fraction < 2^20) (hl : low < 2^32) :
    let x0 := SqrtBounds.normalizedInput (mantissaIndex exponent fraction)
    let x := normalizedInput exponent fraction low
    1 ≤ x0 ∧ x0 ≤ 4 ∧ x0 ≤ x ∧ x ≤ x0*(1+1/2^20) := by
  have hp := index_properties exponent fraction hf
  have hfq : (fraction : ℝ) ≤ 1048576 := by exact_mod_cast (show fraction ≤ 2^20 by omega)
  have hlq : (low : ℝ) ≤ 4294967296 := by exact_mod_cast (show low ≤ 2^32 by omega)
  have hf0 : (0 : ℝ) ≤ fraction := by positivity
  have hl0 : (0 : ℝ) ≤ low := by positivity
  dsimp [SqrtBounds.normalizedInput,normalizedInput]
  norm_num only [Nat.reducePow] at hp
  rw [hp.2.1,hp.2.2]
  have hpar : (exponent+1)%2 = 0 ∨ (exponent+1)%2 = 1 := by omega
  rcases hpar with hpar | hpar <;> rw [hpar] <;> push_cast <;> norm_num <;>
    refine ⟨?_,?_,?_,?_⟩ <;> nlinarith

theorem normalized_relative_error (exponent fraction low : Nat)
    (hf : fraction < 2^20) (hl : low < 2^32) :
    |normalizedOutput (mantissaIndex exponent fraction)*Real.sqrt (normalizedInput exponent fraction low)-1| ≤ 9/2^23 := by
  let index := mantissaIndex exponent fraction
  let y := normalizedOutput index
  let y0 := RsqBounds.normalizedOutput index
  let x0 := SqrtBounds.normalizedInput index
  let x := normalizedInput exponent fraction low
  have hi := (index_properties exponent fraction hf).1
  have hr := normalized_rounding index hi
  have hb := RsqBounds.rsq_absolute_error index hi
  have hx := normalized_input_bounds exponent fraction low hf hl
  change 1 ≤ x0 ∧ x0 ≤ 4 ∧ x0 ≤ x ∧ x ≤ x0*(1+1/2^20) at hx
  have hs0 : 0 ≤ Real.sqrt x0 := Real.sqrt_nonneg _
  have hs : 0 ≤ Real.sqrt x := Real.sqrt_nonneg _
  have hsq0 := Real.sq_sqrt (show 0 ≤ x0 by linarith)
  have hsq := Real.sq_sqrt (show 0 ≤ x by linarith)
  have hslo : 1 ≤ Real.sqrt x0 := by nlinarith only [hs0,hsq0,hx.1]
  have hshi : Real.sqrt x0 ≤ 2 := by nlinarith only [hs0,hsq0,hx.2.1]
  have hsm : Real.sqrt x0 ≤ Real.sqrt x := Real.sqrt_le_sqrt hx.2.2.1
  have hideal : (1/Real.sqrt x0)*Real.sqrt x0 = 1 := by field_simp
  have hb' := abs_le.mp hb
  have hyl : -(9/2^24) ≤ y-1/Real.sqrt x0 := by
    dsimp only [y,y0,x0] at *
    linarith [hr.2.2.2]
  have hyh : y-1/Real.sqrt x0 ≤ 1/2^23 := by
    dsimp only [y,y0,x0] at *
    linarith [hr.2.2.1]
  have hlo0 := mul_le_mul_of_nonneg_right hyl hs0
  have hlo1 := mul_le_mul_of_nonneg_left hshi (by norm_num : (0 : ℝ) ≤ 9/2^24)
  have hhi0 := mul_le_mul_of_nonneg_right hyh hs0
  have hhi1 := mul_le_mul_of_nonneg_left hshi (by norm_num : (0 : ℝ) ≤ 1/2^23)
  have ha : 1-9/2^23 ≤ y*Real.sqrt x0 ∧ y*Real.sqrt x0 ≤ 1+1/2^22 := by
    constructor <;> nlinarith only [hlo0,hlo1,hhi0,hhi1,hideal]
  have hym : y*Real.sqrt x0 ≤ y*Real.sqrt x := mul_le_mul_of_nonneg_left hsm hr.1
  have hyp := mul_nonneg hr.1 hs0
  have hyps := mul_nonneg hr.1 hs
  have hscaled := mul_le_mul_of_nonneg_left hx.2.2.2 (sq_nonneg y)
  have hsqBound : (y*Real.sqrt x0)^2 ≤ (1+1/2^22 : ℝ)^2 := by nlinarith only [ha.2,hyp]
  have hproduct := mul_le_mul_of_nonneg_right hsqBound (by norm_num : (0 : ℝ) ≤ 1+1/2^20)
  have hident0 : (y*Real.sqrt x0)^2 = y^2*x0 := by rw [mul_pow,hsq0]
  have hident : (y*Real.sqrt x)^2 = y^2*x := by rw [mul_pow,hsq]
  have hupper : y*Real.sqrt x ≤ 1+9/2^23 := by
    nlinarith only [hscaled,hproduct,hident0,hident,hyps]
  exact abs_le.mpr ⟨by linarith [ha.1],by linarith⟩

#print axioms rsq_coefficients
#print axioms normalized_relative_error

theorem input_decode_scaled (exponent fraction low : Nat)
    (he : exponent < 2048) (hf : fraction < 2^20) (hl : low < 2^32) :
    positiveNormalValue (exponent*2^52+fraction*2^32+low) =
      normalizedInput exponent fraction low*outputScale exponent^2 := by
  unfold positiveNormalValue
  rw [Nat.add_assoc,Fp64SeedBounds.normal_decode _ _ he (by omega)]
  push_cast
  let parity := (exponent+1)%2
  have heq : (exponent : Int)-1075 = -52+(parity : Int)+2*scaleExponent exponent := by
    dsimp [parity,scaleExponent]
    omega
  have hpow : (2 : ℝ)^((exponent : Int)-1075) =
      (2 : ℝ)^(-52 : Int)*(2 : ℝ)^parity*((2 : ℝ)^scaleExponent exponent)^2 := by
    rw [← zpow_natCast (2 : ℝ) parity,← zpow_natCast ((2 : ℝ)^scaleExponent exponent) 2,
      ← zpow_mul,← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0),
      ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    congr 1
    rw [heq]
    ring
  rw [hpow]
  dsimp [normalizedInput,outputScale,parity]
  norm_num
  ring

theorem high_word (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2046) (hf : fraction < 2^20) :
    Fp64Seed.highWord true (exponent*2^20+fraction) =
      let z := mantissa (mantissaIndex exponent fraction)
      (1023-scaleExponent exponent-(if z = 2^24 then 0 else 1)).toNat*2^20+z%2^23/8 := by
  let w := exponent*2^20+fraction
  have hw : w < 2^31 := by dsimp [w];omega
  have hm : w%2^32 = w := by omega
  have hs := RcpBounds.small_sign w hw
  change w &&& 2147483648 = 0 at hs
  have heq : w/2^20%2048 = exponent := by dsimp [w];omega
  have hfq : w%2^20 = fraction := by dsimp [w];omega
  have ht : -511 ≤ scaleExponent exponent ∧ scaleExponent exponent ≤ 511 := by
    dsimp [scaleExponent]
    omega
  change Fp64Seed.highWord true w = _
  unfold Fp64Seed.highWord
  rw [hm]
  simp only [hs,heq,hfq,show exponent ≠ 2047 by omega,show exponent ≠ 0 by omega,
    true_and,false_and,ite_false,ite_true,ne_eq,not_true_eq_false]
  change (if 1023-scaleExponent exponent-(if mantissa (mantissaIndex exponent fraction) = 2^24 then 0 else 1) ≤ 0 then 0 else _) = _
  have hn : ¬(1023-scaleExponent exponent-(if mantissa (mantissaIndex exponent fraction) = 2^24 then 0 else 1) : Int) ≤ 0 := by
    split_ifs <;> omega
  rw [ite_eq_right hn]
  simp only [Nat.zero_or]
  change ((1023-scaleExponent exponent-(if mantissa (mantissaIndex exponent fraction) = 2^24 then 0 else 1)).toNat*2^20 ||| mantissa (mantissaIndex exponent fraction)%2^23/8) = _
  rw [Nat.mul_comm _ (2^20),← Nat.two_pow_add_eq_or_of_lt (show mantissa (mantissaIndex exponent fraction)%2^23/8 < 2^20 by omega)]

theorem output_word (exponent fraction low : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2046) (hf : fraction < 2^20) (hl : low < 2^32) :
    Fp64Seed.evaluate true (exponent*2^52+fraction*2^32+low) =
      let z := mantissa (mantissaIndex exponent fraction)
      (1023-scaleExponent exponent-(if z = 2^24 then 0 else 1)).toNat*2^52+(z%2^23/8)*2^32 := by
  have hd : (exponent*2^52+fraction*2^32+low)/2^32 = exponent*2^20+fraction := by omega
  rw [Fp64Seed.evaluate,hd,high_word exponent fraction he hf]
  ring

theorem output_is_positive_normal (exponent fraction low : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2046) (hf : fraction < 2^20) (hl : low < 2^32) :
    let word := Fp64Seed.evaluate true (exponent*2^52+fraction*2^32+low)
    1 ≤ word/2^52 ∧ word/2^52 ≤ 2046 := by
  have ht : -511 ≤ scaleExponent exponent ∧ scaleExponent exponent ≤ 511 := by
    dsimp [scaleExponent]
    omega
  rw [output_word exponent fraction low he hf hl]
  dsimp only
  generalize mantissa (mantissaIndex exponent fraction) = z
  split_ifs <;> omega

#print axioms output_is_positive_normal

theorem output_decode_scaled (exponent fraction low : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2046) (hf : fraction < 2^20) (hl : low < 2^32) :
    positiveNormalValue (Fp64Seed.evaluate true (exponent*2^52+fraction*2^32+low))*outputScale exponent =
      normalizedOutput (mantissaIndex exponent fraction) := by
  rw [output_word exponent fraction low he hf hl]
  let z := mantissa (mantissaIndex exponent fraction)
  have hb : 2^23 ≤ z ∧ z ≤ 2^24 := mantissa_range _ (index_properties exponent fraction hf).1
  have ht : -511 ≤ scaleExponent exponent ∧ scaleExponent exponent ≤ 511 := by
    dsimp [scaleExponent]
    omega
  change positiveNormalValue ((1023-scaleExponent exponent-(if z = 2^24 then 0 else 1)).toNat*2^52+(z%2^23/8)*2^32)*_ = _
  by_cases hz : z = 2^24
  · rw [ite_eq_left hz,hz]
    norm_num only [Nat.reducePow,Nat.reduceMod,Nat.reduceDiv,Nat.zero_mul,Nat.add_zero,sub_zero]
    have hdec := Fp64SeedBounds.normal_decode (1023-scaleExponent exponent).toNat 0 (by omega) (by norm_num)
    simp only [Nat.add_zero] at hdec
    norm_num only [Nat.reducePow] at hdec
    unfold positiveNormalValue
    rw [hdec]
    push_cast
    have hc : (((1023-scaleExponent exponent).toNat : Nat) : Int) = 1023-scaleExponent exponent := by omega
    rw [hc]
    dsimp [outputScale]
    rw [mul_assoc,← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    rw [show 1023-scaleExponent exponent-1075+scaleExponent exponent = (-52 : Int) by omega]
    change _ = ((z/8 : Nat) : ℝ)/2^21
    rw [hz]
    norm_num
  · rw [ite_eq_right hz]
    unfold positiveNormalValue
    rw [Fp64SeedBounds.normal_decode _ _ (by omega) (by omega)]
    have hm : z%2^23/8 = z/8-2^20 := by omega
    have hsub : ((z/8-2^20 : Nat) : ℝ) = ((z/8 : Nat) : ℝ)-2^20 := by
      rw [Nat.cast_sub (by omega)];norm_num
    rw [hm]
    push_cast
    norm_num only [Nat.reducePow] at hsub
    rw [hsub]
    have hc : (((1023-scaleExponent exponent-1).toNat : Nat) : Int) = 1023-scaleExponent exponent-1 := by omega
    rw [hc]
    dsimp [outputScale]
    rw [mul_assoc,← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    rw [show 1023-scaleExponent exponent-1-1075+scaleExponent exponent = (-53 : Int) by omega]
    change _ = ((z/8 : Nat) : ℝ)/2^21
    norm_num
    ring

theorem rsq_bits_relative_error (exponent fraction low : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2046) (hf : fraction < 2^20) (hl : low < 2^32) :
    let word := exponent*2^52+fraction*2^32+low
    |positiveNormalValue (Fp64Seed.evaluate true word)*Real.sqrt (positiveNormalValue word)-1| ≤ 9/2^23 := by
  dsimp only
  rw [input_decode_scaled exponent fraction low (by omega) hf hl,
    Real.sqrt_mul (show 0 ≤ normalizedInput exponent fraction low by unfold normalizedInput;positivity),
    Real.sqrt_sq (show 0 ≤ outputScale exponent by unfold outputScale;positivity)]
  have ho := output_decode_scaled exponent fraction low he hf hl
  have hid : positiveNormalValue (Fp64Seed.evaluate true (exponent*2^52+fraction*2^32+low))*
      (Real.sqrt (normalizedInput exponent fraction low)*outputScale exponent) =
      normalizedOutput (mantissaIndex exponent fraction)*Real.sqrt (normalizedInput exponent fraction low) := by
    calc
      _ = (positiveNormalValue (Fp64Seed.evaluate true (exponent*2^52+fraction*2^32+low))*outputScale exponent)*Real.sqrt (normalizedInput exponent fraction low) := by ring
      _ = _ := by rw [ho]
  rw [hid]
  exact normalized_relative_error exponent fraction low hf hl

theorem rsq_bits_ideal_error (exponent fraction low : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2046) (hf : fraction < 2^20) (hl : low < 2^32) :
    let word := exponent*2^52+fraction*2^32+low
    |positiveNormalValue (Fp64Seed.evaluate true word)-1/Real.sqrt (positiveNormalValue word)| ≤
      (9/2^23)/Real.sqrt (positiveNormalValue word) := by
  let word := exponent*2^52+fraction*2^32+low
  have hx : 0 < positiveNormalValue word := by
    change 0 < positiveNormalValue (exponent*2^52+fraction*2^32+low)
    rw [input_decode_scaled exponent fraction low (by omega) hf hl]
    unfold normalizedInput outputScale
    positivity
  have hs := Real.sqrt_pos.mpr hx
  have h := rsq_bits_relative_error exponent fraction low he hf hl
  have hid : positiveNormalValue (Fp64Seed.evaluate true word)-1/Real.sqrt (positiveNormalValue word) =
      (positiveNormalValue (Fp64Seed.evaluate true word)*Real.sqrt (positiveNormalValue word)-1)/Real.sqrt (positiveNormalValue word) := by
    field_simp
  change |positiveNormalValue (Fp64Seed.evaluate true word)-1/Real.sqrt (positiveNormalValue word)| ≤ _
  rw [hid,abs_div,abs_of_pos hs]
  exact div_le_div_of_nonneg_right h hs.le

theorem all_positive_normal_words (word : Nat)
    (he : 1 ≤ word/2^52 ∧ word/2^52 ≤ 2046) :
    |positiveNormalValue (Fp64Seed.evaluate true word)-1/Real.sqrt (positiveNormalValue word)| ≤
      (9/2^23)/Real.sqrt (positiveNormalValue word) := by
  have hword : word/2^52*2^52+(word/2^32%2^20)*2^32+word%2^32 = word := by omega
  have result := rsq_bits_ideal_error (word/2^52) (word/2^32%2^20) (word%2^32)
    he (by omega) (by omega)
  simpa only [hword] using result

#print axioms all_positive_normal_words

#print axioms rsq_bits_relative_error
#print axioms rsq_bits_ideal_error

end GB300Math.Fp64RsqBounds
