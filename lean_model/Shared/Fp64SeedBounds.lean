import RcpBounds

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000

namespace GB300Math.Fp64SeedBounds

theorem reciprocal_coefficients : SeedTables.rcp = MufuTables.rcp.coefficients := by decide +kernel

theorem reciprocal_interpolate (index : Nat) :
    Fp64Seed.interpolate false index = (Mufu.interpolate MufuTables.rcp index).toNat := by
  simp [Fp64Seed.interpolate,Mufu.interpolate,MufuTables.rcp,reciprocal_coefficients]

def inputMantissa (fraction low : Nat) : ℚ := 1+(fraction : ℚ)/2^20+(low : ℚ)/2^52
def outputMantissa (fraction : Nat) : ℚ := (Fp64Seed.interpolate false (fraction*8)/8 : Nat)/2^21

theorem normalized_rounding (fraction : Nat) (hf : fraction < 2^20) :
    let y := outputMantissa fraction
    let y0 := RcpBounds.normalizedOutput (fraction*8)
    0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ y0-y ∧ y0-y ≤ 7/2^24 := by
  have hi : fraction*8 < 2^23 := by omega
  have hr := RcpBounds.interpolate_range (fraction*8) hi
  let z := (Mufu.interpolate MufuTables.rcp (fraction*8)).toNat
  have hz : (z : Int) = Mufu.interpolate MufuTables.rcp (fraction*8) := by dsimp [z];omega
  have hzb : 2^23 ≤ z ∧ z ≤ 2^24 := by dsimp [z];omega
  have hq : z/8 ≤ 2^21 := by omega
  have he : (z/8 : Nat)*8+z%8 = z := by omega
  have heq : ((z/8 : Nat) : ℚ)*8+(z%8 : Nat) = z := by exact_mod_cast he
  have hb : (z%8 : Nat) ≤ 7 := by omega
  have hbq : ((z%8 : Nat) : ℚ) ≤ 7 := by exact_mod_cast hb
  have hqq : ((z/8 : Nat) : ℚ) ≤ 2^21 := by exact_mod_cast hq
  dsimp [outputMantissa,RcpBounds.normalizedOutput]
  rw [reciprocal_interpolate]
  have hzq : (Mufu.interpolate MufuTables.rcp (fraction*8) : ℚ) = z := by exact_mod_cast hz.symm
  rw [hzq]
  change (0 : ℚ) ≤ ((z/8 : Nat) : ℚ)/2^21 ∧ ((z/8 : Nat) : ℚ)/2^21 ≤ 1 ∧ _
  norm_num only [Nat.cast_mul, Nat.cast_ofNat, OfNat.ofNat] at heq hbq hqq
  norm_num
  refine ⟨by positivity,by linarith,?_,?_⟩
  · have h0 : (0 : ℚ) ≤ (z%8 : Nat) := by positivity
    linarith
  · linarith

theorem reciprocal_normalized_relative_error (fraction low : Nat)
    (hf : fraction < 2^20) (hl : low < 2^32) :
    |outputMantissa fraction*inputMantissa fraction low-1| ≤ 37/2^25 := by
  let y := outputMantissa fraction
  let y0 := RcpBounds.normalizedOutput (fraction*8)
  let x0 := RcpBounds.normalizedInput (fraction*8)
  let delta : ℚ := (low : ℚ)/2^52
  have hr := normalized_rounding fraction hf
  have hb := RcpBounds.normalized_relative_error (fraction*8) (by omega)
  have hb' := abs_le.mp hb
  have hx0 : 1 ≤ x0 ∧ x0 ≤ 2 := by
    dsimp [x0,RcpBounds.normalizedInput]
    have hq : (fraction : ℚ) ≤ 1048576 := by exact_mod_cast (show fraction ≤ 2^20 by omega)
    norm_num
    have h0 : (0 : ℚ) ≤ fraction := by positivity
    constructor <;> nlinarith
  have hd : 0 ≤ delta ∧ delta ≤ 1/2^20 := by
    dsimp [delta]
    have hq : (low : ℚ) ≤ 4294967296 := by exact_mod_cast (show low ≤ 2^32 by omega)
    norm_num
    exact ⟨by positivity,by linarith⟩
  have hxi : inputMantissa fraction low = x0+delta := by
    dsimp [inputMantissa,x0,delta,RcpBounds.normalizedInput]
    push_cast
    ring
  have hround0 := mul_nonneg hr.2.2.1 (show 0 ≤ x0 by linarith [hx0.1])
  have hround1 := mul_le_mul hr.2.2.2 hx0.2 (show 0 ≤ x0 by linarith [hx0.1]) (by norm_num : (0 : ℚ) ≤ 7/2^24)
  have hextra0 := mul_nonneg hr.1 hd.1
  have hextra1 := mul_le_mul hr.2.1 hd.2 hd.1 (by norm_num : (0 : ℚ) ≤ 1)
  rw [hxi]
  apply abs_le.mpr
  dsimp only [y,y0,x0] at *
  constructor <;> nlinarith only [hb'.1,hb'.2,hround0,hround1,hextra0,hextra1]

def positiveNormalValue (word : Nat) : ℚ :=
  (2^52+(word%2^52 : Nat))*(2 : ℚ)^((word/2^52%2048 : Nat)-1075 : Int)

def exponentScale (exponent : Nat) : ℚ := (2 : ℚ)^((exponent : Int)-1023)

theorem normal_decode (exponent fraction : Nat) (he : exponent < 2048) (hf : fraction < 2^52) :
    positiveNormalValue (exponent*2^52+fraction) =
      (2^52+(fraction : ℚ))*(2 : ℚ)^((exponent : Int)-1075) := by
  have heq : (exponent*2^52+fraction)/2^52%2048 = exponent := by omega
  have hfq : (exponent*2^52+fraction)%2^52 = fraction := by omega
  simp only [positiveNormalValue,heq,hfq]

theorem input_decode (exponent fraction low : Nat)
    (he : exponent < 2048) (hf : fraction < 2^20) (hl : low < 2^32) :
    positiveNormalValue (exponent*2^52+fraction*2^32+low) =
      inputMantissa fraction low*exponentScale exponent := by
  rw [Nat.add_assoc,normal_decode _ _ he (by omega)]
  have hpow : (2 : ℚ)^((exponent : Int)-1075) =
      (2 : ℚ)^(-52 : Int)*(2 : ℚ)^((exponent : Int)-1023) := by
    rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
    congr 1
    omega
  rw [hpow]
  dsimp [inputMantissa,exponentScale]
  push_cast
  norm_num
  ring

theorem high_word (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2044) (hf : fraction < 2^20) :
    Fp64Seed.highWord false (exponent*2^20+fraction) =
      let z := Fp64Seed.interpolate false (fraction*8)
      (if z = 2^24 then 2046-exponent else 2045-exponent)*2^20+z%2^23/8 := by
  let w := exponent*2^20+fraction
  have hw : w < 2^31 := by dsimp [w];omega
  have hm : w%2^32 = w := by omega
  have hs := RcpBounds.small_sign w hw
  change w &&& 2147483648 = 0 at hs
  have heq : w/2^20%2048 = exponent := by dsimp [w];omega
  have hfq : w%2^20 = fraction := by dsimp [w];omega
  change Fp64Seed.highWord false w = _
  unfold Fp64Seed.highWord
  rw [hm]
  change (let sign := Mufu.sign w; let exponent := w/2^20%2048; let fraction := w%2^20; _) = _
  simp only [hs,heq,hfq]
  simp only [show exponent ≠ 2047 by omega,show exponent ≠ 0 by omega,
    Bool.false_eq_true,false_and,ite_false,Nat.zero_mul,Nat.zero_add]
  generalize Fp64Seed.interpolate false (fraction*8) = z
  by_cases hz : z = 2^24
  · rw [ite_eq_left hz]
    have hn : (0 : Int) < 2046-exponent := by omega
    have hc : (2046-(exponent : Int)).toNat = 2046-exponent := by omega
    simp only [ite_eq_left hz,Int.sub_zero,not_le.mpr hn,ite_false,hc,Nat.zero_or]
    rw [Nat.mul_comm (2046-exponent),← Nat.two_pow_add_eq_or_of_lt (show z%2^23/8 < 2^20 by omega)]
  · have hn : (0 : Int) < 2046-exponent-1 := by omega
    have hc : (2046-(exponent : Int)-1).toNat = 2045-exponent := by omega
    simp only [ite_eq_right hz,not_le.mpr hn,ite_false,hc,Nat.zero_or]
    rw [Nat.mul_comm (2045-exponent),← Nat.two_pow_add_eq_or_of_lt (show z%2^23/8 < 2^20 by omega)]

theorem output_word (exponent fraction low : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2044) (hf : fraction < 2^20) (hl : low < 2^32) :
    Fp64Seed.evaluate false (exponent*2^52+fraction*2^32+low) =
      let z := Fp64Seed.interpolate false (fraction*8)
      (if z = 2^24 then 2046-exponent else 2045-exponent)*2^52+(z%2^23/8)*2^32 := by
  have hd : (exponent*2^52+fraction*2^32+low)/2^32 = exponent*2^20+fraction := by omega
  rw [Fp64Seed.evaluate,hd,high_word exponent fraction he hf]
  ring

theorem output_decode (exponent fraction low : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2044) (hf : fraction < 2^20) (hl : low < 2^32) :
    positiveNormalValue (Fp64Seed.evaluate false (exponent*2^52+fraction*2^32+low))*exponentScale exponent =
      outputMantissa fraction := by
  rw [output_word exponent fraction low he hf hl]
  let z := Fp64Seed.interpolate false (fraction*8)
  have hr := RcpBounds.interpolate_range (fraction*8) (by omega)
  have hb : 2^23 ≤ z ∧ z ≤ 2^24 := by dsimp [z];rw [reciprocal_interpolate];omega
  change positiveNormalValue ((if z = 2^24 then _ else _)*2^52+(z%2^23/8)*2^32)*_ = _
  by_cases hz : z = 2^24
  · rw [ite_eq_left hz,hz]
    norm_num only [Nat.pow_succ,Nat.pow_zero,Nat.mul_one,Nat.reducePow,Nat.reduceMod,Nat.reduceDiv,Nat.zero_mul,Nat.add_zero]
    have hdec := normal_decode (2046-exponent) 0 (by omega) (by norm_num)
    simp only [Nat.add_zero] at hdec
    norm_num only [Nat.reducePow] at hdec
    rw [hdec]
    dsimp [exponentScale,outputMantissa]
    have hz' : Fp64Seed.interpolate false (fraction*8) = 16777216 := hz
    rw [hz']
    have hc : ((2046-exponent : Nat) : Int)-1075+((exponent : Int)-1023) = -52 := by omega
    rw [mul_assoc,← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0),hc]
    norm_num
  · rw [ite_eq_right hz,normal_decode _ _ (by omega) (by omega)]
    have hm : z%2^23/8 = z/8-2^20 := by omega
    have hsub : ((z/8-2^20 : Nat) : ℚ) = ((z/8 : Nat) : ℚ)-2^20 := by
      rw [Nat.cast_sub (by omega)];norm_num
    rw [hm]
    push_cast
    norm_num only [Nat.reducePow] at hsub
    rw [hsub]
    dsimp [exponentScale]
    have hc : ((2045-exponent : Nat) : Int)-1075+((exponent : Int)-1023) = -53 := by omega
    rw [mul_assoc,← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0),hc]
    change _ = ((z/8 : Nat) : ℚ)/2^21
    norm_num
    ring

theorem reciprocal_bits_relative_error (exponent fraction low : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2044) (hf : fraction < 2^20) (hl : low < 2^32) :
    let word := exponent*2^52+fraction*2^32+low
    |positiveNormalValue (Fp64Seed.evaluate false word)*positiveNormalValue word-1| ≤ 37/2^25 := by
  dsimp only
  rw [input_decode exponent fraction low (by omega) hf hl]
  have ho := output_decode exponent fraction low he hf hl
  have hx : positiveNormalValue (Fp64Seed.evaluate false (exponent*2^52+fraction*2^32+low))*
      (inputMantissa fraction low*exponentScale exponent) = outputMantissa fraction*inputMantissa fraction low := by
    calc
      _ = (positiveNormalValue (Fp64Seed.evaluate false (exponent*2^52+fraction*2^32+low))*exponentScale exponent)*inputMantissa fraction low := by ring
      _ = _ := by rw [ho]
  rw [hx]
  exact reciprocal_normalized_relative_error fraction low hf hl

theorem high_word_negative (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2044) (hf : fraction < 2^20) :
    Fp64Seed.highWord false (2^31+(exponent*2^20+fraction)) =
      2^31+Fp64Seed.highWord false (exponent*2^20+fraction) := by
  let w := exponent*2^20+fraction
  have hw : w < 2^31 := by dsimp [w];omega
  have hm0 : w%2^32 = w := by omega
  have hm1 : (2^31+w)%2^32 = 2^31+w := by omega
  have hs0 := RcpBounds.small_sign w hw
  have hs1 := RcpBounds.high_sign w hw
  change w &&& 2147483648 = 0 at hs0
  change (2^31+w) &&& 2147483648 = 2^31 at hs1
  have he0 : w/2^20%2048 = exponent := by dsimp [w];omega
  have he1 : (2^31+w)/2^20%2048 = exponent := by dsimp [w];omega
  have hf0 : w%2^20 = fraction := by dsimp [w];omega
  have hf1 : (2^31+w)%2^20 = fraction := by dsimp [w];omega
  change Fp64Seed.highWord false (2^31+w) = 2^31+Fp64Seed.highWord false w
  unfold Fp64Seed.highWord
  simp only [hm0,hm1,hs0,hs1,he0,he1,hf0,hf1,
    show exponent ≠ 2047 by omega,show exponent ≠ 0 by omega,
    Bool.false_eq_true,false_and,ite_false,Nat.zero_mul,Nat.zero_add]
  generalize Fp64Seed.interpolate false (fraction*8) = z
  by_cases hz : z = 2^24
  all_goals
    have hn : (0 : Int) < 2046-exponent-(if z = 2^24 then 0 else 1) := by split <;> omega
    simp only [hz,↓reduceIte] at hn ⊢
    rw [ite_eq_right (not_le.mpr hn),ite_eq_right (not_le.mpr hn)]
    simp only [Nat.zero_or]
    rw [Nat.or_assoc]
    have hb : ((2046-(exponent : Int)-(if z = 2^24 then 0 else 1)).toNat*2^20 ||| z%2^23/8) < 2^31 := by
      apply Nat.or_lt_two_pow
      · have hle : (2046-(exponent : Int)-(if z = 2^24 then 0 else 1)).toNat ≤ 2046 := by
          split_ifs; omega
        exact Nat.mul_lt_mul_of_pos_right (Nat.lt_of_le_of_lt hle (by decide : 2046 < 2048)) (by decide : 0 < 2^20)
      · clear hn hz hs0 hs1
        omega
    simp only [hz,↓reduceIte] at hb
    rw [Nat.or_comm (2^31),Nat.or_two_pow_eq_add_of_lt hb,Nat.add_comm]

def normalValue (word : Nat) : ℚ :=
  if word < 2^63 then positiveNormalValue word else -positiveNormalValue word

theorem value_sign (word : Nat) (hw : word < 2^63) :
    normalValue word = positiveNormalValue word ∧
    normalValue (2^63+word) = -positiveNormalValue word := by
  have he : (2^63+word)/2^52%2048 = word/2^52%2048 := by omega
  have hf : (2^63+word)%2^52 = word%2^52 := by omega
  simp only [normalValue,ite_eq_left hw,ite_eq_right (show ¬2^63+word < 2^63 by omega)]
  unfold positiveNormalValue
  rw [he,hf]
  exact ⟨trivial,rfl⟩

theorem reciprocal_bits_signed_relative_error (negative : Bool) (exponent fraction low : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 2044) (hf : fraction < 2^20) (hl : low < 2^32) :
    let word := (if negative then 2^63 else 0)+exponent*2^52+fraction*2^32+low
    |normalValue (Fp64Seed.evaluate false word)*normalValue word-1| ≤ 37/2^25 := by
  let x := exponent*2^52+fraction*2^32+low
  have hx : x < 2^63 := by dsimp [x];omega
  have hy : Fp64Seed.evaluate false x < 2^63 := by
    change Fp64Seed.evaluate false (exponent*2^52+fraction*2^32+low) < 2^63
    rw [output_word exponent fraction low he hf hl]
    generalize Fp64Seed.interpolate false (fraction*8) = z
    dsimp only
    clear hx x
    split <;> omega
  have hneg : Fp64Seed.evaluate false (2^63+x) = 2^63+Fp64Seed.evaluate false x := by
    have hd0 : x/2^32 = exponent*2^20+fraction := by dsimp [x];omega
    have hd1 : (2^63+x)/2^32 = 2^31+(exponent*2^20+fraction) := by dsimp [x];omega
    rw [Fp64Seed.evaluate,hd1,high_word_negative exponent fraction he hf,Fp64Seed.evaluate,hd0]
    ring
  have vx := value_sign x hx
  have vy := value_sign (Fp64Seed.evaluate false x) hy
  cases negative
  · simp only [Bool.false_eq_true,↓reduceIte,Nat.zero_add]
    change |normalValue (Fp64Seed.evaluate false x)*normalValue x-1| ≤ _
    rw [vx.1,vy.1]
    exact reciprocal_bits_relative_error exponent fraction low he hf hl
  · simp only [↓reduceIte]
    have hw : 2^63+exponent*2^52+fraction*2^32+low = 2^63+x := by dsimp [x];omega
    rw [hw,hneg,vx.2,vy.2,neg_mul_neg]
    exact reciprocal_bits_relative_error exponent fraction low he hf hl

#print axioms reciprocal_coefficients
#print axioms reciprocal_normalized_relative_error
#print axioms reciprocal_bits_relative_error
#print axioms reciprocal_bits_signed_relative_error
end GB300Math.Fp64SeedBounds
