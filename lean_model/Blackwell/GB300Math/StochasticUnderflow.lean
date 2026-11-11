import GB300Math.StochasticSubnormal

namespace GB300Math.StochasticUnderflow

open StochasticSm103 StochasticValue StochasticProbability

def sourceMantissa (e t : Nat) : Nat := t + if e = 0 then 0 else 8388608

def drop (f : Format) (e : Nat) : Nat :=
  23-fractionBits f + (128-(max e 1+(parameters f).2.2.1))

def sourceScale (e : Nat) : Rat := (2:Rat)^(((max e 1:Nat):Int)-150)

def spacing (f : Format) : Rat := (2:Rat)^((1:Int)-(parameters f).2.2.1-fractionBits f)

theorem bias_bound (f : Format) : (parameters f).2.2.1 ≤ 127 := by cases f <;> decide

theorem fraction_bound (f : Format) : fractionBits f ≤ 23 := by cases f <;> decide

theorem source_exponent_bound (f : Format) (e : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) : e ≤ 127 := by omega

theorem shifted_exponent_bound (f : Format) (e : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) : max e 1+(parameters f).2.2.1 ≤ 128 := by
  have hb := bias_bound f
  omega

theorem entropy_width (f : Format) (e : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) : entropyBits f ≤ drop f e := by
  cases f <;> norm_num [drop, entropyBits, fractionBits, parameters, isPair] at * <;> omega

theorem source_mantissa_bound (f : Format) (e t : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) (ht : t < 8388608) :
    sourceMantissa e t < 2^(drop f e+fractionBits f) := by
  by_cases hf : f = .bf16
  · subst f
    norm_num [parameters] at he
    have hz : e = 0 := by omega
    simp [hz, sourceMantissa, drop, fractionBits, parameters]
    exact ht
  · have hd : 24 ≤ drop f e+fractionBits f := by
      cases f <;> norm_num [drop, fractionBits, parameters] at * <;> omega
    have hp : 2^24 ≤ 2^(drop f e+fractionBits f) := Nat.pow_le_pow_right (by decide) hd
    have hm : sourceMantissa e t < 2^24 := by unfold sourceMantissa; split_ifs <;> omega
    omega

theorem rounded_bound (f : Format) (e t r : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) (ht : t < 8388608) :
    roundMantissa (sourceMantissa e t) r (drop f e) (entropyBits f) ≤ 2^(fractionBits f) := by
  have hm := source_mantissa_bound f e t he ht
  rw [Nat.pow_add] at hm
  have hr := (strict_one_step_error (sourceMantissa e t) r (drop f e) (entropyBits f)
    (entropy_width f e he)).1
  have hlt : roundMantissa (sourceMantissa e t) r (drop f e) (entropyBits f) * 2^(drop f e) <
      (2^(fractionBits f)+1)*2^(drop f e) := by nlinarith
  have h := (Nat.mul_lt_mul_right (Nat.two_pow_pos (drop f e))).mp hlt
  omega

theorem scale_relation (f : Format) (e : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) :
    (2:Rat)^(drop f e)*sourceScale e = spacing f := by
  have hp := fraction_bound f
  have hb := shifted_exponent_bound f e he
  have hs : ((drop f e:Nat):Int)+(((max e 1:Nat):Int)-150) =
      1-((parameters f).2.2.1:Int)-(fractionBits f:Int) := by unfold drop; omega
  unfold sourceScale spacing
  rw [← zpow_natCast, ← zpow_add₀ (by norm_num : (2:Rat) ≠ 0), hs]


theorem bitvalue (f : Format) (relu satfinite : Bool) (e t r : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) (ht : t < 8388608) :
    lane f relu satfinite (e*8388608+t) r =
      roundMantissa (sourceMantissa e t) r (drop f e) (entropyBits f) := by
  have he' := source_exponent_bound f e he
  have hshift := shifted_exponent_bound f e he
  have hx : e*8388608+t < 4294967296 := by omega
  have hxe : (e*8388608+t)/8388608%256 = e := by omega
  have hf : (e*8388608+t)%8388608 = t := by omega
  have hs : (e*8388608+t)/2147483648 = 0 := by omega
  have hzero : max 0 ((((max e 1:Nat):Int)-127+(parameters f).2.2.1)-1) = 0 := by omega
  have hdrop : (max 0 (1-(((max e 1:Nat):Int)-127+(parameters f).2.2.1))).toNat =
      128-(max e 1+(parameters f).2.2.1) := by omega
  have hne255 : e ≠ 255 := by omega
  have hq := rounded_bound f e t r he ht
  have hmax : 2^(fractionBits f) ≤ (parameters f).2.2.2 := by cases f <;> decide
  have hcap : roundMantissa (sourceMantissa e t) r (drop f e) (entropyBits f) ≤
      if isPair f && !satfinite then (parameters f).2.2.2+1 else (parameters f).2.2.2 := by
    split_ifs <;> omega
  simp only [lane, Nat.mod_eq_of_lt hx, hxe, hf, hs, hzero, hdrop, hne255,
    decide_false, Bool.false_and, Bool.and_false, Bool.false_eq_true, ↓reduceIte,
    bne_self_eq_false, zero_mul, zero_add, Int.toNat_zero]
  exact Nat.min_eq_left hcap


def output (f : Format) (e t r : Nat) : Rat :=
  (roundMantissa (sourceMantissa e t) r (drop f e) (entropyBits f):Rat)*spacing f

theorem conversion_value (f : Format) (relu satfinite : Bool) (e t r : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) (ht : t < 8388608) :
    finiteValue f (lane f relu satfinite (e*8388608+t) r) = some (output f e t r) := by
  rw [bitvalue f relu satfinite e t r he ht]
  exact StochasticSubnormal.subnormal_code_value f _ (rounded_bound f e t r he ht)

theorem input_value (e t : Nat) (he : e ≤ 127) (ht : t < 8388608) :
    rationalValue 8 23 127 0x7f7fffff (e*8388608+t) =
      some ((sourceMantissa e t:Rat)*sourceScale e) := by
  by_cases hz : e = 0
  · subst e
    have h := StochasticSubnormal.positive_initial_bin 8 23 127 0x7f7fffff t (by decide) (by decide)
      (show t ≤ 2^23 by omega)
    simpa [sourceMantissa, sourceScale] using h
  · have h := StochasticSubnormal.fp32_normal_value e t (by omega) (by omega) ht
    simpa [sourceMantissa, sourceScale, hz, show max e 1 = e by omega, add_comm] using h

theorem conversion_error (f : Format) (e t r : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) :
    |output f e t r-(sourceMantissa e t:Rat)*sourceScale e| < spacing f := by
  have h := strict_one_step_error (sourceMantissa e t) r (drop f e) (entropyBits f)
    (entropy_width f e he)
  have hl : (roundMantissa (sourceMantissa e t) r (drop f e) (entropyBits f):Rat) *
      (2:Rat)^(drop f e) < (sourceMantissa e t:Rat)+(2:Rat)^(drop f e) := by exact_mod_cast h.1
  have hr : (sourceMantissa e t:Rat) <
      (roundMantissa (sourceMantissa e t) r (drop f e) (entropyBits f):Rat) *
      (2:Rat)^(drop f e)+(2:Rat)^(drop f e) := by exact_mod_cast h.2
  have hp : 0 < sourceScale e := by unfold sourceScale; positivity
  have hleft := mul_lt_mul_of_pos_right hl hp
  have hright := mul_lt_mul_of_pos_right hr hp
  have hs := scale_relation f e he
  unfold output
  rw [abs_lt]
  constructor <;> nlinarith

theorem full_error (f : Format) (relu satfinite : Bool) (e t r : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) (ht : t < 8388608) :
    ∃ input : Rat,
      rationalValue 8 23 127 0x7f7fffff (e*8388608+t) = some input ∧
      finiteValue f (lane f relu satfinite (e*8388608+t) r) = some (output f e t r) ∧
      |output f e t r-input| < spacing f :=
  ⟨_, input_value e t (source_exponent_bound f e he) ht,
    conversion_value f relu satfinite e t r he ht, conversion_error f e t r he⟩


def quantum (f : Format) (e : Nat) : Nat := 2^(drop f e-entropyBits f)

def mean (f : Format) (e t : Nat) : Rat :=
  (∑ r ∈ Finset.range (2^(entropyBits f)), output f e t r)/(2:Rat)^(entropyBits f)

theorem quantum_scale (f : Format) (e : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) :
    (quantum f e:Rat)*(2:Rat)^(entropyBits f)*sourceScale e = spacing f := by
  have hw := entropy_width f e he
  have hp : quantum f e*2^(entropyBits f) = 2^(drop f e) := by
    unfold quantum
    rw [← Nat.pow_add]
    congr 1
    omega
  have hpr : (quantum f e:Rat)*(2:Rat)^(entropyBits f) = (2:Rat)^(drop f e) := by
    exact_mod_cast hp
  rw [hpr]
  exact scale_relation f e he

theorem exact_mean (f : Format) (e t : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) :
    mean f e t = ((sourceMantissa e t-sourceMantissa e t%quantum f e:Nat):Rat)*sourceScale e := by
  have hm := model_mean (sourceMantissa e t) (drop f e) (entropyBits f) (entropy_width f e he)
  have hmr : (quantum f e:Rat) *
      ((∑ r ∈ Finset.range (2^(entropyBits f)),
        roundMantissa (sourceMantissa e t) r (drop f e) (entropyBits f):Nat):Rat) =
      ((sourceMantissa e t-sourceMantissa e t%quantum f e:Nat):Rat) := by exact_mod_cast hm
  have hs : (∑ r ∈ Finset.range (2^(entropyBits f)), output f e t r) =
      ((∑ r ∈ Finset.range (2^(entropyBits f)),
        roundMantissa (sourceMantissa e t) r (drop f e) (entropyBits f):Nat):Rat)*spacing f := by
    simp [output, Finset.sum_mul]
  unfold mean
  rw [hs, ← quantum_scale f e he, ← hmr]
  have hn : (2:Rat)^(entropyBits f) ≠ 0 := by positivity
  field_simp

theorem mean_error (f : Format) (e t : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) :
    |mean f e t-(sourceMantissa e t:Rat)*sourceScale e| =
      (sourceMantissa e t%quantum f e:Nat)*sourceScale e := by
  rw [exact_mean f e t he, Nat.cast_sub (Nat.mod_le _ _)]
  rw [show ((sourceMantissa e t:Rat)-(sourceMantissa e t%quantum f e:Nat))*sourceScale e -
      (sourceMantissa e t:Rat)*sourceScale e =
      -((sourceMantissa e t%quantum f e:Nat)*sourceScale e) by ring, abs_neg]
  apply abs_of_nonneg
  unfold sourceScale
  positivity

theorem mean_error_bound (f : Format) (e t : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) :
    |mean f e t-(sourceMantissa e t:Rat)*sourceScale e| < spacing f/(2:Rat)^(entropyBits f) := by
  rw [mean_error f e t he]
  have hn : (2:Rat)^(entropyBits f) ≠ 0 := by positivity
  have hq : sourceMantissa e t%quantum f e < quantum f e := Nat.mod_lt _ (Nat.two_pow_pos _)
  have hqr : (sourceMantissa e t%quantum f e:Nat) < (quantum f e:Rat) := by exact_mod_cast hq
  have hp : 0 < sourceScale e := by unfold sourceScale; positivity
  have hs : spacing f/(2:Rat)^(entropyBits f) = (quantum f e:Rat)*sourceScale e := by
    rw [← quantum_scale f e he]
    field_simp
  rw [hs]
  exact mul_lt_mul_of_pos_right hqr hp

#print axioms exact_mean
#print axioms mean_error
#print axioms mean_error_bound
#print axioms conversion_value
#print axioms input_value
#print axioms conversion_error
#print axioms full_error
#print axioms bitvalue
#print axioms entropy_width
#print axioms source_mantissa_bound
#print axioms rounded_bound
#print axioms scale_relation

end GB300Math.StochasticUnderflow
