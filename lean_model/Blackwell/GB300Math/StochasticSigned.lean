/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.StochasticFinite
import GB300Math.StochasticUnderflow
namespace GB300Math.StochasticSigned
open StochasticSm103 StochasticValue StochasticNormal

theorem rational_negative (e p b maximum code : Nat) (v : Rat)
    (hc : code ≤ maximum) (hm : maximum < 2^(e+p))
    (hv : rationalValue e p b maximum code = some v) :
    rationalValue e p b maximum (code+2^(e+p)) = some (-v) := by
  have hp := Nat.two_pow_pos (e+p)
  have hl : code < 2^(e+p) := by omega
  have hsign : (code+2^(e+p))/2^(e+p) = 1 := by
    rw [Nat.add_div_right _ hp, Nat.div_eq_of_lt hl]
  have hmod : (code+2^(e+p))%2^(e+p) = code := by simp [Nat.mod_eq_of_lt hl]
  have hwidth : ¬ code+2^(e+p) ≥ 2^(e+p+1) := by rw [Nat.pow_succ]; omega
  have hwidth0 : ¬ code ≥ 2^(e+p+1) := by rw [Nat.pow_succ]; omega
  simp only [rationalValue, Nat.mod_eq_of_lt hl, Nat.div_eq_of_lt hl, hmod, hsign,
    hwidth, hwidth0, Nat.not_lt.mpr hc, decide_false, Bool.false_or,
    Bool.false_eq_true, ↓reduceIte, one_ne_zero] at hv ⊢
  simp only [Option.some.injEq] at hv ⊢
  exact congrArg Neg.neg hv

theorem lane_negative (f : Format) (satfinite : Bool) (x r : Nat)
    (hx : x < 2139095040) :
    lane f false satfinite (x+2147483648) r =
      2^((parameters f).1+(parameters f).2.1)+lane f false satfinite x r := by
  have hx32 : x < 4294967296 := by omega
  have hx32' : x+2147483648 < 4294967296 := by omega
  have hs : x/2147483648 = 0 := by omega
  have hs' : (x+2147483648)/2147483648 = 1 := by omega
  have hfrac : (x+2147483648)%8388608 = x%8388608 := by omega
  have he : (x+2147483648)/8388608%256 = x/8388608%256 := by omega
  have hne : x/8388608%256 ≠ 255 := by omega
  simp only [lane, Nat.mod_eq_of_lt hx32, Nat.mod_eq_of_lt hx32', hs, hs', hfrac, he,
    hne, decide_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte,
    zero_mul, one_mul, zero_add]

theorem lane_rectified (f : Format) (satfinite : Bool) (x r : Nat)
    (hx : x < 2139095040) : lane f true satfinite (x+2147483648) r = 0 := by
  have hx32' : x+2147483648 < 4294967296 := by omega
  have hs' : (x+2147483648)/2147483648 = 1 := by omega
  have he : (x+2147483648)/8388608%256 = x/8388608%256 := by omega
  have hne : x/8388608%256 ≠ 255 := by omega
  simp only [lane, Nat.mod_eq_of_lt hx32', hs', he, hne, decide_false,
    Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  rfl

#print axioms rational_negative
#print axioms lane_negative
#print axioms lane_rectified


theorem finite_negative_value (f : Format) (satfinite : Bool) (x r : Nat) (v : Rat)
    (hx : x < 2139095040) (hc : lane f false satfinite x r ≤ maximum f)
    (hv : finiteValue f (lane f false satfinite x r) = some v) :
    finiteValue f (lane f false satfinite (x+2147483648) r) = some (-v) := by
  rw [lane_negative f satfinite x r hx, Nat.add_comm]
  change rationalValue (exponentBits f) (fractionBits f) (bias f) (maximum f) _ = _
  exact rational_negative _ _ _ _ _ _ hc (by cases f <;> decide) hv

theorem full_negative_error (f : Format) (satfinite : Bool) (x r : Nat) (input output bound : Rat)
    (hx : x < 2139095040) (hc : lane f false satfinite x r ≤ maximum f)
    (hi : rationalValue 8 23 127 0x7f7fffff x = some input)
    (ho : finiteValue f (lane f false satfinite x r) = some output)
    (he : |output-input| < bound) :
    rationalValue 8 23 127 0x7f7fffff (x+2147483648) = some (-input) ∧
      finiteValue f (lane f false satfinite (x+2147483648) r) = some (-output) ∧
      |(-output)-(-input)| < bound := by
  refine ⟨rational_negative 8 23 127 0x7f7fffff x input (by omega) (by decide) hi,
    finite_negative_value f satfinite x r output hx hc ho, ?_⟩
  rw [show (-output)-(-input) = -(output-input) by ring, abs_neg]
  exact he

theorem negative_normal_error (f : Format) (satfinite : Bool) (e j t r : Nat)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hb : e+bias f = j+127)
    (hbase : j*2^(fractionBits f) ≤ maximum f)
    (hcap : t ≤ (maximum f-j*2^(fractionBits f))*2^(23-fractionBits f))
    (ht : t < 8388608) :
    ∃ input : Rat,
      rationalValue 8 23 127 0x7f7fffff (e*8388608+t+2147483648) = some (-input) ∧
      finiteValue f (lane f false satfinite (e*8388608+t+2147483648) r) = some (-normalOutput f j t r) ∧
      |(-normalOutput f j t r)-(-input)| < (2:Rat)^((j:Int)-bias f)/(2:Rat)^(fractionBits f) := by
  obtain ⟨input,hi,ho,herr⟩ := StochasticFinite.finite_full_error f false satfinite e j t r
    he he' hj hb hbase hcap ht
  have hq := StochasticFinite.normal_round_cap f j t r hbase hcap
  have hc : lane f false satfinite (e*8388608+t) r ≤ maximum f := by
    rw [StochasticFinite.finite_bitvalue f false satfinite e j t r he he' hj hb hq ht]
    exact hq
  exact ⟨input,full_negative_error f satfinite _ r input _ _ (by omega) hc hi ho herr⟩

theorem negative_underflow_error (f : Format) (satfinite : Bool) (e t r : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) (ht : t < 8388608) :
    ∃ input : Rat,
      rationalValue 8 23 127 0x7f7fffff (e*8388608+t+2147483648) = some (-input) ∧
      finiteValue f (lane f false satfinite (e*8388608+t+2147483648) r) =
        some (-StochasticUnderflow.output f e t r) ∧
      |(-StochasticUnderflow.output f e t r)-(-input)| < StochasticUnderflow.spacing f := by
  obtain ⟨input,hi,ho,herr⟩ := StochasticUnderflow.full_error f false satfinite e t r he ht
  have he' := StochasticUnderflow.source_exponent_bound f e he
  have hc : lane f false satfinite (e*8388608+t) r ≤ maximum f := by
    rw [StochasticUnderflow.bitvalue f false satfinite e t r he ht]
    have hq := StochasticUnderflow.rounded_bound f e t r he ht
    have hm : 2^(fractionBits f) ≤ maximum f := by cases f <;> decide
    omega
  exact ⟨input,full_negative_error f satfinite _ r input _ _ (by omega) hc hi ho herr⟩

#print axioms finite_negative_value
#print axioms full_negative_error
#print axioms negative_normal_error
#print axioms negative_underflow_error
end GB300Math.StochasticSigned
