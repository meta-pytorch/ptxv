import GB300Math.StochasticProbability

namespace GB300Math.StochasticValue

open StochasticSm103 StochasticProbability

def rationalValue (exponentBits fractionBits bias maximum code : Nat) : Option Rat :=
  let magnitude := code % 2^(exponentBits+fractionBits)
  if code ≥ 2^(exponentBits+fractionBits+1) || magnitude > maximum then none
  else
    let exponent := magnitude / 2^fractionBits
    let mantissa := magnitude % 2^fractionBits + if exponent = 0 then 0 else 2^fractionBits
    let value : Rat := mantissa * (2 : Rat)^(((max exponent 1 : Nat) : Int) - bias - fractionBits)
    some (if code / 2^(exponentBits+fractionBits) = 0 then value else -value)

def finiteValue (f : Format) (code : Nat) : Option Rat :=
  let (e,p,b,m) := parameters f
  rationalValue e p b m code


theorem positive_unit_bin (e p b maximum q : Nat) (hb : 0 < b)
    (hbe : b+1 < 2^e) (hm : (b+1)*2^p ≤ maximum) (hq : q ≤ 2^p) :
    rationalValue e p b maximum (b*2^p+q) = some (1+(q:Rat)/(2:Rat)^p) := by
  have hp := Nat.two_pow_pos p
  have hc : b*2^p+q ≤ (b+1)*2^p := by nlinarith
  have he : b*2^p+q < 2^(e+p) := by
    rw [Nat.pow_add]
    have hmul := Nat.mul_lt_mul_of_pos_right hbe hp
    omega
  have hsmall : ¬ b*2^p+q ≥ 2^(e+p+1) := by
    rw [Nat.pow_succ]
    have := Nat.two_pow_pos (e+p)
    omega
  have hmax : ¬ b*2^p+q > maximum := by omega
  have hmod := Nat.mod_eq_of_lt he
  have hs := Nat.div_eq_of_lt he
  simp only [rationalValue, hmod, hsmall, hmax, decide_false, Bool.false_or,
    Bool.false_eq_true, ↓reduceIte, hs]
  by_cases heq : q = 2^p
  · subst q
    have hc : b*2^p+2^p = (b+1)*2^p := by ring
    rw [hc, Nat.mul_div_cancel _ hp, Nat.mul_mod_left]
    simp only [Nat.add_eq_zero_iff, one_ne_zero, and_false, ↓reduceIte, zero_add]
    have hmaxb : max (b+1) 1 = b+1 := by omega
    rw [hmaxb]
    have hs : ((b+1:Nat):Int) - b - p = 1 - (p:Int) := by omega
    rw [hs, zpow_sub₀ (by norm_num : (2:Rat) ≠ 0)]
    norm_num
    field_simp
  · have hlt : q < 2^p := by omega
    have hd : (b*2^p+q)/2^p = b := by
      rw [Nat.mul_comm b, Nat.mul_add_div hp, Nat.div_eq_of_lt hlt, Nat.add_zero]
    have hfrac : (b*2^p+q)%2^p = q := by simp [Nat.add_mod, Nat.mod_eq_of_lt hlt]
    rw [hd, hfrac]
    have hmaxb : max b 1 = b := by omega
    simp only [Nat.ne_of_gt hb, ↓reduceIte, hmaxb]
    have hs : (b:Int) - b - p = -(p:Int) := by omega
    rw [hs]
    norm_num [zpow_neg]
    field_simp
    ring



def fractionBits (f : Format) : Nat := (parameters f).2.1

def unitBase (f : Format) : Nat := (parameters f).2.2.1 * 2^(fractionBits f)

def entropyBits (f : Format) : Nat := if isPair f then 23-fractionBits f else 16

theorem unit_bitvalue (f : Format) (relu satfinite : Bool) (t r : Nat) (ht : t < 8388608) :
    lane f relu satfinite (0x3f800000+t) r =
      unitBase f + roundMantissa t r (23-fractionBits f) (entropyBits f) := by
  have hx : (0x3f800000+t)%4294967296 = 0x3f800000+t := by omega
  have he : (0x3f800000+t)/8388608%256 = 127 := by omega
  have hf : (0x3f800000+t)%8388608 = t := by omega
  have hs : (0x3f800000+t)/2147483648 = 0 := by omega
  cases f <;> cases relu <;> cases satfinite <;> norm_num [lane, parameters, isPair, hx, he, hf, hs, unitBase,
    fractionBits, entropyBits, roundMantissa] <;> omega



theorem unit_round_bound (f : Format) (t r : Nat) (ht : t < 8388608) :
    roundMantissa t r (23-fractionBits f) (entropyBits f) ≤ 2^(fractionBits f) := by
  cases f <;> norm_num [fractionBits, entropyBits, parameters, isPair, roundMantissa] <;> omega

theorem unit_code_value (f : Format) (q : Nat) (hq : q ≤ 2^(fractionBits f)) :
    finiteValue f (unitBase f+q) = some (1+(q:Rat)/(2:Rat)^(fractionBits f)) := by
  cases f <;> simp only [finiteValue, unitBase, fractionBits, parameters] at * <;>
    apply positive_unit_bin <;> norm_num at * <;> omega

def unitOutput (f : Format) (t r : Nat) : Rat :=
  1 + (roundMantissa t r (23-fractionBits f) (entropyBits f) : Rat) /
    (2:Rat)^(fractionBits f)

theorem unit_conversion_value (f : Format) (relu satfinite : Bool) (t r : Nat)
    (ht : t < 8388608) :
    finiteValue f (lane f relu satfinite (0x3f800000+t) r) = some (unitOutput f t r) := by
  rw [unit_bitvalue f relu satfinite t r ht]
  exact unit_code_value f _ (unit_round_bound f t r ht)

theorem input_unit_value (t : Nat) (ht : t < 8388608) :
    rationalValue 8 23 127 0x7f7fffff (0x3f800000+t) = some (1+(t:Rat)/8388608) := by
  have h := positive_unit_bin 8 23 127 0x7f7fffff t (by decide) (by decide) (by decide)
    (show t ≤ 2^23 by omega)
  norm_num at h
  exact h

theorem entropyBits_le_drop (f : Format) : entropyBits f ≤ 23-fractionBits f := by
  cases f <;> decide

theorem unitOutput_error (f : Format) (t r : Nat) :
    |unitOutput f t r - (1+(t:Rat)/8388608)| < 1/(2:Rat)^(fractionBits f) := by
  have h := strict_one_step_error t r (23-fractionBits f) (entropyBits f)
    (entropyBits_le_drop f)
  have hl : (roundMantissa t r (23-fractionBits f) (entropyBits f) : Rat) *
      (2:Rat)^(23-fractionBits f) < (t:Rat)+(2:Rat)^(23-fractionBits f) := by
    exact_mod_cast h.1
  have hr : (t:Rat) < (roundMantissa t r (23-fractionBits f) (entropyBits f) : Rat) *
      (2:Rat)^(23-fractionBits f) + (2:Rat)^(23-fractionBits f) := by
    exact_mod_cast h.2
  rw [abs_lt]
  cases f <;> norm_num [unitOutput, fractionBits, entropyBits, parameters, isPair] at * <;>
    constructor <;> linarith

theorem unit_conversion_error (f : Format) (relu satfinite : Bool) (t r : Nat)
    (ht : t < 8388608) :
    finiteValue f (lane f relu satfinite (0x3f800000+t) r) = some (unitOutput f t r) ∧
      |unitOutput f t r - (1+(t:Rat)/8388608)| < 1/(2:Rat)^(fractionBits f) :=
  ⟨unit_conversion_value f relu satfinite t r ht, unitOutput_error f t r⟩

def quantum (f : Format) : Nat := 2^(23-fractionBits f-entropyBits f)

def unitMean (f : Format) (t : Nat) : Rat :=
  (∑ r ∈ Finset.range (2^(entropyBits f)), unitOutput f t r) / (2:Rat)^(entropyBits f)

theorem unit_mean (f : Format) (t : Nat) :
    unitMean f t = 1 + ((t-t%quantum f : Nat) : Rat)/8388608 := by
  have hm := model_mean t (23-fractionBits f) (entropyBits f) (entropyBits_le_drop f)
  have hmr : (quantum f : Rat) *
      ((∑ r ∈ Finset.range (2^(entropyBits f)),
        roundMantissa t r (23-fractionBits f) (entropyBits f) : Nat) : Rat) =
      ((t-t%quantum f : Nat) : Rat) := by
    exact_mod_cast hm
  have hs : (∑ r ∈ Finset.range (2^(entropyBits f)), unitOutput f t r) =
      (2:Rat)^(entropyBits f) +
      ((∑ r ∈ Finset.range (2^(entropyBits f)),
        roundMantissa t r (23-fractionBits f) (entropyBits f) : Nat) : Rat) /
        (2:Rat)^(fractionBits f) := by
    simp [unitOutput, Finset.sum_add_distrib, Finset.sum_div]
  unfold unitMean
  rw [hs]
  have hscale : (quantum f : Rat) * (2:Rat)^(entropyBits f) *
      (2:Rat)^(fractionBits f) = 8388608 := by
    cases f <;> norm_num [quantum, fractionBits, entropyBits, parameters, isPair]
  have hn : (2:Rat)^(entropyBits f) ≠ 0 := by positivity
  have hp : (2:Rat)^(fractionBits f) ≠ 0 := by positivity
  have hq : (quantum f : Rat) ≠ 0 := by unfold quantum; positivity
  rw [← hmr, ← hscale]
  field_simp

theorem unit_mean_error (f : Format) (t : Nat) :
    |unitMean f t - (1+(t:Rat)/8388608)| = (t%quantum f : Nat)/8388608 := by
  rw [unit_mean, Nat.cast_sub (Nat.mod_le t (quantum f))]
  rw [show (1+((t:Rat)-(t%quantum f : Nat))/8388608) - (1+(t:Rat)/8388608) =
    -((t%quantum f : Nat)/8388608) by ring, abs_neg]
  exact abs_of_nonneg (by positivity)

theorem unit_mean_error_bound (f : Format) (t : Nat) :
    |unitMean f t - (1+(t:Rat)/8388608)| <
      1 / ((2:Rat)^(fractionBits f) * (2:Rat)^(entropyBits f)) := by
  rw [unit_mean_error]
  have h : t%quantum f < quantum f := Nat.mod_lt _ (Nat.two_pow_pos _)
  have hrat : (t%quantum f : Nat) < (quantum f : Rat) := by exact_mod_cast h
  have hscale : 1 / ((2:Rat)^(fractionBits f) * (2:Rat)^(entropyBits f)) =
      (quantum f : Rat)/8388608 := by
    cases f <;> norm_num [quantum, fractionBits, entropyBits, parameters, isPair]
  rw [hscale]
  exact (div_lt_div_iff_of_pos_right (by norm_num : (0:Rat) < 8388608)).mpr hrat

#print axioms unit_bitvalue
#print axioms unit_code_value
#print axioms unit_conversion_value
#print axioms input_unit_value
#print axioms unit_conversion_error
#print axioms unit_mean
#print axioms unit_mean_error
#print axioms unit_mean_error_bound

end GB300Math.StochasticValue
