import GB300Math.StochasticNormal
namespace GB300Math.StochasticFinite
open StochasticSm103 StochasticValue StochasticProbability StochasticNormal
theorem positive_finite_bin (e p b j maximum q : Nat) (hb : 0 < j)
    (hmaxcode : maximum < 2^(e+p)) (hm : j*2^p+q ≤ maximum) (hq : q ≤ 2^p) :
    rationalValue e p b maximum (j*2^p+q) = some ((1+(q:Rat)/(2:Rat)^p) * (2:Rat)^((j:Int)-b)) := by
  have hp := Nat.two_pow_pos p
  have hc : j*2^p+q ≤ (j+1)*2^p := by nlinarith
  have he : j*2^p+q < 2^(e+p) := by omega
  have hsmall : ¬ j*2^p+q ≥ 2^(e+p+1) := by
    rw [Nat.pow_succ]
    have := Nat.two_pow_pos (e+p)
    omega
  have hmax : ¬ j*2^p+q > maximum := by omega
  have hmod := Nat.mod_eq_of_lt he
  have hs := Nat.div_eq_of_lt he
  simp only [rationalValue, hmod, hsmall, hmax, decide_false, Bool.false_or,
    Bool.false_eq_true, ↓reduceIte, hs]
  by_cases heq : q = 2^p
  · subst q
    have hc : j*2^p+2^p = (j+1)*2^p := by ring
    rw [hc, Nat.mul_div_cancel _ hp, Nat.mul_mod_left]
    simp only [Nat.add_eq_zero_iff, one_ne_zero, and_false, ↓reduceIte, zero_add]
    have hmaxb : max (j+1) 1 = j+1 := by omega
    rw [hmaxb]
    have hs : ((j+1:Nat):Int) - b - p = ((j:Int)-b) + (1 - (p:Int)) := by omega
    rw [hs, zpow_add₀ (by norm_num : (2:Rat) ≠ 0)]
    have hpw : (2:Rat)^(1-(p:Int)) = 2/(2:Rat)^p := by
      rw [zpow_sub₀ (by norm_num : (2:Rat) ≠ 0)]
      norm_num
    rw [hpw]
    norm_num
    field_simp
  · have hlt : q < 2^p := by omega
    have hd : (j*2^p+q)/2^p = j := by
      rw [Nat.mul_comm j, Nat.mul_add_div hp, Nat.div_eq_of_lt hlt, Nat.add_zero]
    have hfrac : (j*2^p+q)%2^p = q := by simp [Nat.add_mod, Nat.mod_eq_of_lt hlt]
    rw [hd, hfrac]
    have hmaxb : max j 1 = j := by omega
    simp only [Nat.ne_of_gt hb, ↓reduceIte, hmaxb]
    have hs : (j:Int) - b - p = ((j:Int)-b) - (p:Int) := by omega
    rw [hs, zpow_sub₀ (by norm_num : (2:Rat) ≠ 0)]
    norm_num
    field_simp
    ring





theorem round_cap (m r d b cap : Nat) (hb : b ≤ d) (hm : m ≤ cap*2^d) :
    roundMantissa m r d b ≤ cap := by
  have he := (strict_one_step_error m r d b hb).1
  have hlt : roundMantissa m r d b * 2^d < (cap+1)*2^d := by nlinarith
  have h := (Nat.mul_lt_mul_right (Nat.two_pow_pos d)).mp hlt
  omega

theorem normal_round_cap (f : Format) (j t r : Nat)
    (hj : j*2^(fractionBits f) ≤ maximum f)
    (ht : t ≤ (maximum f-j*2^(fractionBits f))*2^(23-fractionBits f)) :
    j*2^(fractionBits f)+roundMantissa t r (23-fractionBits f) (entropyBits f) ≤ maximum f := by
  have hw : entropyBits f ≤ 23-fractionBits f := by cases f <;> decide
  have h := round_cap t r _ _ _ hw ht
  omega

theorem finite_bitvalue (f : Format) (relu satfinite : Bool) (e j t r : Nat)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hb : e+bias f = j+127)
    (hm : j*2^(fractionBits f)+roundMantissa t r (23-fractionBits f) (entropyBits f) ≤ maximum f)
    (ht : t < 8388608) :
    lane f relu satfinite (e*8388608+t) r =
      j*2^(fractionBits f)+roundMantissa t r (23-fractionBits f) (entropyBits f) := by
  have hx : (e*8388608+t)%4294967296 = e*8388608+t := by omega
  have hxe : (e*8388608+t)/8388608%256 = e := by omega
  have hf : (e*8388608+t)%8388608 = t := by omega
  have hs : (e*8388608+t)/2147483648 = 0 := by omega
  have hmaxe : max e 1 = e := by omega
  have htexp : (e:Int)-127+bias f = (j:Int) := by omega
  have hdrop : max 0 (1-(j:Int)) = 0 := by omega
  have hbase : max 0 ((j:Int)-1) = ((j-1:Nat):Int) := by omega
  have hne : e ≠ 0 := by omega
  have hne255 : e ≠ 255 := by omega
  cases f <;> cases relu <;> cases satfinite <;>
    norm_num only [bias, maximum, fractionBits, parameters, entropyBits, isPair] at hb hm htexp ⊢ <;>
    norm_num only [lane, parameters, isPair] <;>
    simp only [hx, hxe, hf, hs, hmaxe, htexp, hdrop, hbase, hne, hne255] <;>
    clear hx hxe hf hs hmaxe htexp hdrop hbase hne hne255 hb he he' <;>
    norm_num [roundMantissa] at hm ⊢ <;> omega

theorem finite_code_value (f : Format) (j q : Nat) (hj : 0 < j)
    (hm : j*2^(fractionBits f)+q ≤ maximum f) (hq : q ≤ 2^(fractionBits f)) :
    finiteValue f (j*2^(fractionBits f)+q) =
      some ((1+(q:Rat)/(2:Rat)^(fractionBits f)) * (2:Rat)^((j:Int)-bias f)) := by
  change rationalValue (exponentBits f) (fractionBits f) (bias f) (maximum f) _ = _
  apply positive_finite_bin _ _ _ _ _ _ hj _ hm hq
  cases f <;> decide

theorem fp32_input (e t : Nat) (he : 1 ≤ e) (he' : e ≤ 254) (ht : t < 8388608) :
    rationalValue 8 23 127 0x7f7fffff (e*8388608+t) =
      some ((1+(t:Rat)/8388608)*(2:Rat)^((e:Int)-127)) := by
  have h := positive_finite_bin 8 23 127 e 0x7f7fffff t he (by decide)
    (by norm_num; omega) (by norm_num; omega)
  norm_num at h
  exact h

theorem finite_full_error (f : Format) (relu satfinite : Bool) (e j t r : Nat)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hb : e+bias f = j+127)
    (hbase : j*2^(fractionBits f) ≤ maximum f)
    (hcap : t ≤ (maximum f-j*2^(fractionBits f))*2^(23-fractionBits f))
    (ht : t < 8388608) :
    ∃ input : Rat,
      rationalValue 8 23 127 0x7f7fffff (e*8388608+t) = some input ∧
      finiteValue f (lane f relu satfinite (e*8388608+t) r) = some (normalOutput f j t r) ∧
      |normalOutput f j t r-input| < (2:Rat)^((j:Int)-bias f)/(2:Rat)^(fractionBits f) := by
  have hm := normal_round_cap f j t r hbase hcap
  have hi := fp32_input e t he he' ht
  have hs : (e:Int)-127 = (j:Int)-bias f := by omega
  rw [hs] at hi
  refine ⟨_, hi, ?_, normal_conversion_error f j t r⟩
  rw [finite_bitvalue f relu satfinite e j t r he he' hj hb hm ht]
  exact finite_code_value f j _ hj hm (unit_round_bound f t r ht)

#print axioms positive_finite_bin
#print axioms round_cap
#print axioms normal_round_cap
#print axioms finite_bitvalue
#print axioms finite_code_value
#print axioms fp32_input
#print axioms finite_full_error


theorem normal_mean_scale (f : Format) (j t : Nat) :
    normalMean f j t = unitMean f t*(2:Rat)^((j:Int)-bias f) := by
  unfold normalMean normalOutput unitMean
  rw [← Finset.sum_mul]
  ring

theorem normal_mean_error (f : Format) (j t : Nat) :
    |normalMean f j t-(1+(t:Rat)/8388608)*(2:Rat)^((j:Int)-bias f)| =
      ((t%quantum f:Nat):Rat)/8388608*(2:Rat)^((j:Int)-bias f) := by
  have hp : 0 < (2:Rat)^((j:Int)-bias f) := by positivity
  rw [normal_mean_scale, ← sub_mul, abs_mul, abs_of_pos hp, unit_mean_error]

theorem normal_mean_error_bound (f : Format) (j t : Nat) :
    |normalMean f j t-(1+(t:Rat)/8388608)*(2:Rat)^((j:Int)-bias f)| <
      ((2:Rat)^((j:Int)-bias f)/(2:Rat)^(fractionBits f))/(2:Rat)^(entropyBits f) := by
  have hp : 0 < (2:Rat)^((j:Int)-bias f) := by positivity
  have h := mul_lt_mul_of_pos_right (unit_mean_error_bound f t) hp
  rw [normal_mean_scale, ← sub_mul, abs_mul, abs_of_pos hp]
  convert h using 1 <;> ring

#print axioms normal_mean_scale
#print axioms normal_mean_error
#print axioms normal_mean_error_bound
end GB300Math.StochasticFinite
