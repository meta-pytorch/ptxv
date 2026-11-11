/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.StochasticNormal
namespace GB300Math.StochasticSubnormal
open StochasticSm103 StochasticProbability StochasticValue

theorem positive_initial_bin (e p b maximum q : Nat) (he : 0 < e)
    (hm : 2^p ≤ maximum) (hq : q ≤ 2^p) :
    rationalValue e p b maximum q = some ((q:Rat)*(2:Rat)^((1:Int)-b-p)) := by
  have hp := Nat.two_pow_pos p
  have hpow : 2^p < 2^(e+p) := by
    rw [Nat.pow_add]
    have hepow : 1 < 2^e := Nat.one_lt_two_pow (by omega)
    nlinarith
  have hlt : q < 2^(e+p) := by omega
  have hsmall : ¬ q ≥ 2^(e+p+1) := by rw [Nat.pow_succ]; omega
  simp only [rationalValue, Nat.mod_eq_of_lt hlt, hsmall, show ¬q>maximum by omega,
    decide_false, Bool.false_or, Bool.false_eq_true, ↓reduceIte, Nat.div_eq_of_lt hlt]
  by_cases heq : q = 2^p
  · subst q
    simp [Nat.div_self hp]
  · have hq' : q < 2^p := by omega
    simp [Nat.div_eq_of_lt hq', Nat.mod_eq_of_lt hq']


def subnormalInput (f : Format) (t : Nat) : Nat :=
  if f == .bf16 then t else (127-(parameters f).2.2.1)*8388608+t

def subnormalSignificand (f : Format) (t : Nat) : Nat :=
  if f == .bf16 then t else 8388608+t

def subnormalDrop (f : Format) : Nat :=
  if f == .bf16 then 16 else 24-fractionBits f

theorem subnormal_bitvalue (f : Format) (relu satfinite : Bool) (t r : Nat) (ht : t < 8388608) :
    lane f relu satfinite (subnormalInput f t) r =
      roundMantissa (subnormalSignificand f t) r (subnormalDrop f) (entropyBits f) := by
  have hx : subnormalInput f t < 4294967296 := by
    cases f <;> norm_num [subnormalInput, parameters] <;> omega
  have hs : subnormalInput f t / 2147483648 = 0 := by
    cases f <;> norm_num [subnormalInput, parameters] <;> omega
  have he : subnormalInput f t / 8388608 % 256 =
      if f == .bf16 then 0 else 127-(parameters f).2.2.1 := by
    cases f <;> norm_num [subnormalInput, parameters] <;> omega
  have hf : subnormalInput f t % 8388608 = t := by
    cases f <;> norm_num [subnormalInput, parameters] <;> omega
  simp only [lane, Nat.mod_eq_of_lt hx, hs, he, hf]
  clear hx hs he hf
  cases f <;> cases relu <;> cases satfinite <;>
    norm_num [subnormalSignificand, subnormalDrop, fractionBits,
      entropyBits, parameters, isPair, roundMantissa] <;> omega


def spacing (f : Format) : Rat :=
  (2:Rat)^((1:Int)-(parameters f).2.2.1-fractionBits f)

def inputScale (f : Format) : Rat :=
  if f == .bf16 then (2:Rat)^(-149:Int) else (2:Rat)^(-((parameters f).2.2.1:Int)-23)

theorem subnormal_round_bound (f : Format) (t r : Nat) (ht : t < 8388608) :
    roundMantissa (subnormalSignificand f t) r (subnormalDrop f) (entropyBits f) ≤
      2^(fractionBits f) := by
  cases f <;> norm_num [subnormalSignificand, subnormalDrop, entropyBits, fractionBits,
    parameters, isPair, roundMantissa] <;> omega

theorem subnormal_code_value (f : Format) (q : Nat) (hq : q ≤ 2^(fractionBits f)) :
    finiteValue f q = some ((q:Rat)*spacing f) := by
  change rationalValue (parameters f).1 (fractionBits f) (parameters f).2.2.1
    (parameters f).2.2.2 q = _
  apply positive_initial_bin <;> try exact hq
  all_goals cases f <;> decide

def subnormalOutput (f : Format) (t r : Nat) : Rat :=
  (roundMantissa (subnormalSignificand f t) r (subnormalDrop f) (entropyBits f):Rat) * spacing f

theorem subnormal_conversion_value (f : Format) (relu satfinite : Bool) (t r : Nat)
    (ht : t < 8388608) :
    finiteValue f (lane f relu satfinite (subnormalInput f t) r) =
      some (subnormalOutput f t r) := by
  rw [subnormal_bitvalue f relu satfinite t r ht]
  exact subnormal_code_value f _ (subnormal_round_bound f t r ht)

theorem subnormal_entropy_width (f : Format) : entropyBits f ≤ subnormalDrop f := by
  cases f <;> decide

theorem subnormal_conversion_error (f : Format) (t r : Nat) :
    |subnormalOutput f t r - (subnormalSignificand f t:Rat)*inputScale f| < spacing f := by
  have h := strict_one_step_error (subnormalSignificand f t) r (subnormalDrop f)
    (entropyBits f) (subnormal_entropy_width f)
  have hl : (roundMantissa (subnormalSignificand f t) r (subnormalDrop f) (entropyBits f):Rat) *
      (2:Rat)^(subnormalDrop f) < (subnormalSignificand f t:Rat)+(2:Rat)^(subnormalDrop f) := by
    exact_mod_cast h.1
  have hr : (subnormalSignificand f t:Rat) <
      (roundMantissa (subnormalSignificand f t) r (subnormalDrop f) (entropyBits f):Rat) *
        (2:Rat)^(subnormalDrop f)+(2:Rat)^(subnormalDrop f) := by
    exact_mod_cast h.2
  rw [abs_lt]
  cases f <;> norm_num [subnormalOutput, inputScale, spacing, subnormalSignificand,
    subnormalDrop, entropyBits, fractionBits, parameters, isPair] at * <;>
    constructor <;> linarith


def subnormalQuantum (f : Format) : Nat := 2^(subnormalDrop f-entropyBits f)

def subnormalMean (f : Format) (t : Nat) : Rat :=
  (∑ r ∈ Finset.range (2^(entropyBits f)), subnormalOutput f t r) /
    (2:Rat)^(entropyBits f)

theorem subnormal_mean (f : Format) (t : Nat) :
    subnormalMean f t =
      ((subnormalSignificand f t - subnormalSignificand f t % subnormalQuantum f : Nat):Rat) * inputScale f := by
  have hm := model_mean (subnormalSignificand f t) (subnormalDrop f) (entropyBits f)
    (subnormal_entropy_width f)
  have hmr : (subnormalQuantum f:Rat) *
      ((∑ r ∈ Finset.range (2^(entropyBits f)),
        roundMantissa (subnormalSignificand f t) r (subnormalDrop f) (entropyBits f) : Nat):Rat) =
      ((subnormalSignificand f t-subnormalSignificand f t%subnormalQuantum f:Nat):Rat) := by
    exact_mod_cast hm
  have hs : (∑ r ∈ Finset.range (2^(entropyBits f)), subnormalOutput f t r) =
      ((∑ r ∈ Finset.range (2^(entropyBits f)),
        roundMantissa (subnormalSignificand f t) r (subnormalDrop f) (entropyBits f) : Nat):Rat) * spacing f := by
    simp [subnormalOutput, Finset.sum_mul]
  have hscale : (subnormalQuantum f:Rat)*(2:Rat)^(entropyBits f)*inputScale f = spacing f := by
    cases f <;> norm_num [subnormalQuantum, subnormalDrop, entropyBits, fractionBits,
      inputScale, spacing, parameters, isPair]
  unfold subnormalMean
  rw [hs, ← hscale, ← hmr]
  have hn : (2:Rat)^(entropyBits f) ≠ 0 := by positivity
  field_simp

theorem subnormal_mean_error (f : Format) (t : Nat) :
    |subnormalMean f t - (subnormalSignificand f t:Rat)*inputScale f| =
      (subnormalSignificand f t%subnormalQuantum f:Nat)*inputScale f := by
  rw [subnormal_mean, Nat.cast_sub (Nat.mod_le _ _)]
  rw [show ((subnormalSignificand f t:Rat) - (subnormalSignificand f t%subnormalQuantum f:Nat)) * inputScale f -
      (subnormalSignificand f t:Rat)*inputScale f =
      -((subnormalSignificand f t%subnormalQuantum f:Nat)*inputScale f) by ring, abs_neg]
  apply abs_of_nonneg
  unfold inputScale
  split_ifs <;> positivity


theorem subnormal_mean_error_bound (f : Format) (t : Nat) :
    |subnormalMean f t - (subnormalSignificand f t:Rat)*inputScale f| <
      spacing f/(2:Rat)^(entropyBits f) := by
  rw [subnormal_mean_error]
  have h : subnormalSignificand f t%subnormalQuantum f < subnormalQuantum f :=
    Nat.mod_lt _ (Nat.two_pow_pos _)
  have hrat : (subnormalSignificand f t%subnormalQuantum f:Nat) < (subnormalQuantum f:Rat) := by
    exact_mod_cast h
  have hs : 0 < inputScale f := by unfold inputScale; split_ifs <;> positivity
  have hscale : (subnormalQuantum f:Rat)*inputScale f = spacing f/(2:Rat)^(entropyBits f) := by
    cases f <;> norm_num [subnormalQuantum, subnormalDrop, entropyBits, fractionBits,
      inputScale, spacing, parameters, isPair]
  rw [← hscale]
  exact mul_lt_mul_of_pos_right hrat hs


theorem fp32_normal_value (e t : Nat) (he : 0 < e) (he' : e ≤ 253) (ht : t < 8388608) :
    rationalValue 8 23 127 0x7f7fffff (e*8388608+t) =
      some ((8388608+(t:Rat))*(2:Rat)^((e:Int)-150)) := by
  have h := StochasticNormal.positive_normal_bin 8 23 127 e 0x7f7fffff t he
    (by norm_num; omega) (by norm_num; omega) (by norm_num; omega)
  norm_num at h
  rw [h]
  congr 1
  have hs : (e:Int)-150 = ((e:Int)-127)-23 := by omega
  have hpw : (2:Rat)^((e:Int)-150) = (2:Rat)^((e:Int)-127)/8388608 := by
    calc
      _ = (2:Rat)^(((e:Int)-127)-23) := congrArg (fun k : Int => (2:Rat)^k) hs
      _ = (2:Rat)^((e:Int)-127)/(2:Rat)^(23:Int) :=
        zpow_sub₀ (by norm_num : (2:Rat) ≠ 0) _ _
      _ = _ := by norm_num
  rw [hpw]
  ring

theorem subnormal_input_value (f : Format) (t : Nat) (ht : t < 8388608) :
    rationalValue 8 23 127 0x7f7fffff (subnormalInput f t) =
      some ((subnormalSignificand f t:Rat)*inputScale f) := by
  by_cases hf : f = .bf16
  · subst f
    have h := positive_initial_bin 8 23 127 0x7f7fffff t (by decide) (by decide)
      (show t ≤ 2^23 by omega)
    simpa [subnormalInput, subnormalSignificand, inputScale] using h
  · have he : 0 < 127-(parameters f).2.2.1 := by
      cases f <;> norm_num [parameters] at *
    have he' : 127-(parameters f).2.2.1 ≤ 253 := by omega
    have hb : (parameters f).2.2.1 ≤ 127 := by cases f <;> decide
    have h := fp32_normal_value (127-(parameters f).2.2.1) t he he' ht
    have hs : ((127-(parameters f).2.2.1:Nat):Int)-150 = -((parameters f).2.2.1:Int)-23 := by omega
    rw [hs] at h
    simpa [subnormalInput, subnormalSignificand, inputScale, hf, Nat.cast_add] using h

theorem subnormal_full_error (f : Format) (relu satfinite : Bool) (t r : Nat)
    (ht : t < 8388608) :
    ∃ input : Rat,
      rationalValue 8 23 127 0x7f7fffff (subnormalInput f t) = some input ∧
      finiteValue f (lane f relu satfinite (subnormalInput f t) r) = some (subnormalOutput f t r) ∧
      |subnormalOutput f t r-input| < spacing f :=
  ⟨_, subnormal_input_value f t ht, subnormal_conversion_value f relu satfinite t r ht,
    subnormal_conversion_error f t r⟩

#print axioms subnormal_input_value
#print axioms subnormal_full_error
#print axioms subnormal_mean_error_bound
#print axioms subnormal_mean
#print axioms subnormal_mean_error
#print axioms subnormal_code_value
#print axioms subnormal_conversion_value
#print axioms subnormal_conversion_error
#print axioms subnormal_bitvalue
#print axioms positive_initial_bin
end GB300Math.StochasticSubnormal
