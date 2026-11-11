/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import RcpPtxHigh
set_option maxRecDepth 10000
namespace GB300Math.RcpPtxBounds
open RcpBounds

theorem rcp_high_ftz_word (e f : Nat) (he : 253 ≤ e ∧ e ≤ 254)
    (hf : f < 2^23) :
    Mufu.rcp (e*2^23+f) = if e = 253 ∧ f = 0 then 2^23 else 0 := by
  have hh : Mufu.exponent (e*2^23+f) = e := by dsimp [Mufu.exponent]; omega
  have hm : Mufu.fraction (e*2^23+f) = f := by dsimp [Mufu.fraction]; omega
  have hs := small_sign (e*2^23+f) (by omega)
  have hmag : ¬Mufu.magnitude (e*2^23+f) > 0x7f800000 := by dsimp [Mufu.magnitude]; omega
  simp only [Mufu.rcp,hh,hm,hs,ite_eq_right hmag]
  by_cases hf0 : f = 0
  · subst f
    have hz : (Mufu.interpolate MufuTables.rcp 0).toNat = 2^24 := by
      simp only [Mufu.interpolate,Fp64Seed.squareTruncated,Std.Legacy.Range.forIn_eq_forIn_range']
      decide +kernel
    have he0 := he.1
    have he1 := he.2
    interval_cases e <;> norm_num [hz]
  · have hz : (Mufu.interpolate MufuTables.rcp f).toNat ≠ 2^24 := by
      intro h
      exact hf0 (interpolate_top_only_zero f hf h)
    have hbe : (253 : Int)-e ≤ 0 := by omega
    norm_num at hz
    simp [hz,show e ≠ 0 by omega,show e ≠ 255 by omega,hbe,hf0]

theorem rcp_ftz_high_classification (negative : Bool) (e f : Nat)
    (he : 253 ≤ e ∧ e ≤ 254) (hf : f < 2^23) :
    Approx.evaluate "rcp" true ((if negative then 2^31 else 0)+e*2^23+f) =
      some ((if negative then 2^31 else 0)+(if e = 253 ∧ f = 0 then 2^23 else 0)) := by
  rw [rcp_ftz_adapter _ (by cases negative <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega)]
  have hp := rcp_high_ftz_word e f he hf
  cases negative
  · simpa using congrArg some hp
  · have hn := reciprocal_sign (e*2^23+f) (by omega)
    rw [hp] at hn
    simpa [Nat.add_assoc] using congrArg some hn

theorem rcp_ftz_subnormal_classification (negative : Bool) (f : Nat) (hf : f < 2^23) :
    Approx.evaluate "rcp" true ((if negative then 2^31 else 0)+f) =
      some ((if negative then 2^31 else 0)+0x7f800000) := by
  rw [rcp_ftz_adapter _ (by cases negative <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega)]
  have hp : Mufu.rcp f = 0x7f800000 := by
    have he : Mufu.exponent f = 0 := by dsimp [Mufu.exponent]; omega
    have hm : ¬Mufu.magnitude f > 0x7f800000 := by dsimp [Mufu.magnitude]; omega
    have hs := small_sign f (by omega)
    simp [Mufu.rcp,he,hm,hs]
  cases negative
  · simpa using congrArg some hp
  · have hn := reciprocal_sign f (by omega)
    rw [hp] at hn
    simpa using congrArg some hn

#print axioms rcp_ftz_high_classification
#print axioms rcp_ftz_subnormal_classification
end GB300Math.RcpPtxBounds
