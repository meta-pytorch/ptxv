/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricF64
import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Proof

set_option exponentiation.threshold 2048
namespace FabricF64Decode
open FabricF64 FloatLib.Floats.Formats.BinaryInterchange
theorem dyadic_units (d : FloatLib.Numerics.Dyadic) (he : -1074 ≤ d.exponent) :
    d.toReal =
      ((if d.negative then -(d.significand*2^(d.exponent+1074).toNat : Int)
        else d.significand*2^(d.exponent+1074).toNat : Int) : ℝ) / 2^1074 := by
  have hnat : ((d.exponent+1074).toNat : Int) = d.exponent+1074 := Int.toNat_of_nonneg (by omega)
  have hp : (2:ℝ) ^ (d.exponent+1074).toNat = (2:ℝ)^d.exponent * 2^1074 := by
    rw [← zpow_natCast, hnat, zpow_add₀ (by norm_num)]
    norm_num
  cases hs : d.negative <;>
    simp [FloatLib.Numerics.Dyadic.toReal, FloatLib.Numerics.Dyadic.signedSignificand, hs, hp] <;> ring

theorem units_value (word : Nat) (value : Int) (hu : units word = some value) :
    (Model.ofNatBits word : Model FloatFormat.binary64).toReal = (value:ℝ)/2^1074 := by
  by_cases hw : word < 18446744073709551616
  · cases hd : (Model.ofNatBits word : Model FloatFormat.binary64).toDyadic? with
    | none => simp [units, hw, hd] at hu
    | some d =>
      by_cases he : d.exponent ≥ -1074
      · simp [units, hw, hd, he] at hu
        rw [Model.toReal_eq, hd]
        simp only
        rw [← hu]
        exact dyadic_units d he
      · simp [units, hw, hd, he] at hu
  · simp [units, hw] at hu

theorem sum_units_value (words : List Nat)
    (valid : ∀ w ∈ words, ∃ v, units w = some v) :
    (words.map fun w => (Model.ofNatBits w : Model FloatFormat.binary64).toReal).sum =
      (((words.map fun w => (units w).getD 0).sum:Int):ℝ)/2^1074 := by
  induction words with
  | nil => simp [List.sum]
  | cons w ws ih =>
    obtain ⟨v,hv⟩ := valid w (by simp)
    have ht : ∀ t ∈ ws, ∃ v, units t = some v := by
      intro t ht
      exact valid t (by simp [ht])
    simp only [List.map_cons, List.sum_cons]
    rw [units_value w v hv, ih ht]
    simp [List.sum, hv, add_div]

theorem exponent_lower_bound (x : Model FloatFormat.binary64) (d : FloatLib.Numerics.Dyadic)
    (hd : x.toDyadic? = some d) : -1074 ≤ d.exponent := by
  simp only [Model.toDyadic?, show FloatFormat.binary64.isIEEE = true from rfl,
    ↓reduceIte] at hd
  unfold Model.ieeeToDyadic? at hd
  split at hd
  · contradiction
  · dsimp only at hd
    split at hd
    · split at hd
      · cases Option.some.inj hd
        change -1074 ≤ (0:Int)
        omega
      · cases Option.some.inj hd
        change -1074 ≤ (-1074:Int)
        omega
    · rename_i he
      cases Option.some.inj hd
      change -1074 ≤ (x.expField : Int)-1075
      have hpos : x.expField ≠ 0 := by simpa using he
      omega

theorem units_defined (word : Nat) (hw : word < (2^64))
    (hf : Model.isFinite (Model.ofNatBits word : Model FloatFormat.binary64) = true) :
    ∃ v, units word = some v := by
  norm_num at hw
  obtain ⟨d,hd⟩ := Model.exists_toDyadic?_of_isFinite hf
  have he := exponent_lower_bound _ d hd
  simp [units, hw, hd, he]


#print axioms dyadic_units
#print axioms units_value
#print axioms sum_units_value
#print axioms exponent_lower_bound
#print axioms units_defined
end FabricF64Decode
