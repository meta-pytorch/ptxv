import GB300Math.StochasticValue
namespace GB300Math.StochasticNormal
open StochasticSm103 StochasticProbability StochasticValue
theorem positive_normal_bin (e p b j maximum q : Nat) (hb : 0 < j)
    (hbe : j+1 < 2^e) (hm : (j+1)*2^p ≤ maximum) (hq : q ≤ 2^p) :
    rationalValue e p b maximum (j*2^p+q) = some ((1+(q:Rat)/(2:Rat)^p) * (2:Rat)^((j:Int)-b)) := by
  have hp := Nat.two_pow_pos p
  have hc : j*2^p+q ≤ (j+1)*2^p := by nlinarith
  have he : j*2^p+q < 2^(e+p) := by
    rw [Nat.pow_add]
    have hmul := Nat.mul_lt_mul_of_pos_right hbe hp
    omega
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




def exponentBits (f : Format) : Nat := (parameters f).1

def bias (f : Format) : Nat := (parameters f).2.2.1

def maximum (f : Format) : Nat := (parameters f).2.2.2

theorem normal_bitvalue (f : Format) (relu satfinite : Bool) (e j t r : Nat)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hb : e+bias f = j+127) (hm : (j+1)*2^(fractionBits f) ≤ maximum f)
    (ht : t < 8388608) :
    lane f relu satfinite (e*8388608+t) r =
      j*2^(fractionBits f) + roundMantissa t r (23-fractionBits f) (entropyBits f) := by
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
    norm_num [roundMantissa] <;> omega


theorem normal_code_value (f : Format) (j q : Nat) (hj : 0 < j)
    (hje : j+1 < 2^(exponentBits f))
    (hm : (j+1)*2^(fractionBits f) ≤ maximum f) (hq : q ≤ 2^(fractionBits f)) :
    finiteValue f (j*2^(fractionBits f)+q) =
      some ((1+(q:Rat)/(2:Rat)^(fractionBits f)) * (2:Rat)^((j:Int)-bias f)) := by
  change rationalValue (exponentBits f) (fractionBits f) (bias f) (maximum f) _ = _
  exact positive_normal_bin _ _ _ _ _ _ hj hje hm hq

def normalOutput (f : Format) (j t r : Nat) : Rat :=
  unitOutput f t r * (2:Rat)^((j:Int)-bias f)

theorem normal_conversion_value (f : Format) (relu satfinite : Bool) (e j t r : Nat)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hje : j+1 < 2^(exponentBits f))
    (hb : e+bias f = j+127) (hm : (j+1)*2^(fractionBits f) ≤ maximum f)
    (ht : t < 8388608) :
    finiteValue f (lane f relu satfinite (e*8388608+t) r) =
      some (normalOutput f j t r) := by
  rw [normal_bitvalue f relu satfinite e j t r he he' hj hb hm ht]
  exact normal_code_value f j _ hj hje hm (unit_round_bound f t r ht)

theorem normal_conversion_error (f : Format) (j t r : Nat) :
    |normalOutput f j t r - (1+(t:Rat)/8388608)*(2:Rat)^((j:Int)-bias f)| <
      (2:Rat)^((j:Int)-bias f) / (2:Rat)^(fractionBits f) := by
  have h := unitOutput_error f t r
  have hp : 0 < (2:Rat)^((j:Int)-bias f) := by positivity
  unfold normalOutput
  rw [← sub_mul, abs_mul, abs_of_pos hp]
  convert mul_lt_mul_of_pos_right h hp using 1; ring

def normalMean (f : Format) (j t : Nat) : Rat :=
  (∑ r ∈ Finset.range (2^(entropyBits f)), normalOutput f j t r) /
    (2:Rat)^(entropyBits f)

theorem normal_mean (f : Format) (j t : Nat) :
    normalMean f j t =
      (1+((t-t%quantum f:Nat):Rat)/8388608)*(2:Rat)^((j:Int)-bias f) := by
  have hs : normalMean f j t = unitMean f t * (2:Rat)^((j:Int)-bias f) := by
    unfold normalMean normalOutput unitMean
    rw [← Finset.sum_mul]
    ring
  rw [hs, unit_mean]

theorem normal_input_value (f : Format) (e j t : Nat) (he : 1 ≤ e)
    (hb : e+bias f = j+127) (hm : (j+1)*2^(fractionBits f) ≤ maximum f)
    (ht : t < 8388608) :
    rationalValue 8 23 127 0x7f7fffff (e*8388608+t) =
      some ((1+(t:Rat)/8388608)*(2:Rat)^((j:Int)-bias f)) := by
  have he' : e ≤ 253 := by
    cases f <;> norm_num [bias, maximum, fractionBits, parameters] at hb hm <;> omega
  have h := positive_normal_bin 8 23 127 e 0x7f7fffff t he
    (by norm_num; omega) (by norm_num; omega) (by norm_num; omega)
  have hs : (e:Int)-127 = (j:Int)-bias f := by omega
  norm_num at h
  rw [hs] at h
  exact h

theorem normal_full_error (f : Format) (relu satfinite : Bool) (e j t r : Nat)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hje : j+1 < 2^(exponentBits f))
    (hb : e+bias f = j+127) (hm : (j+1)*2^(fractionBits f) ≤ maximum f)
    (ht : t < 8388608) :
    ∃ input : Rat,
      rationalValue 8 23 127 0x7f7fffff (e*8388608+t) = some input ∧
      finiteValue f (lane f relu satfinite (e*8388608+t) r) = some (normalOutput f j t r) ∧
      |normalOutput f j t r-input| < (2:Rat)^((j:Int)-bias f)/(2:Rat)^(fractionBits f) :=
  ⟨_, normal_input_value f e j t he hb hm ht,
    normal_conversion_value f relu satfinite e j t r he he' hj hje hb hm ht,
    normal_conversion_error f j t r⟩

#print axioms normal_input_value
#print axioms normal_full_error

#print axioms normal_code_value
#print axioms normal_conversion_value
#print axioms normal_conversion_error
#print axioms normal_mean
#print axioms positive_normal_bin
#print axioms normal_bitvalue
end GB300Math.StochasticNormal
