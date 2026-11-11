/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricF16Bounds

set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace FabricF16ExactBounds
open FloatLib.Floats.Formats.BinaryInterchange FloatLib.Floats.Formats.Flocq
open FabricF16Bounds

def units (word : Nat) : Option Int := do
  unless word<65536 do none
  let d ← (Model.ofNatBits word : Model FloatFormat.binary16).toDyadic?
  unless d.exponent≥-24 do none
  let n:=d.significand*2^(d.exponent+24).toNat
  return if d.negative then -(n:Int) else n

theorem dyadic_units (d : FloatLib.Numerics.Dyadic) (he : -24 ≤ d.exponent) :
    d.toReal =
      ((if d.negative then -(d.significand*2^(d.exponent+24).toNat : Int)
        else d.significand*2^(d.exponent+24).toNat : Int) : ℝ) / 2^24 := by
  have hnat : ((d.exponent+24).toNat : Int) = d.exponent+24 := Int.toNat_of_nonneg (by omega)
  have hp : (2:ℝ) ^ (d.exponent+24).toNat = (2:ℝ)^d.exponent * 2^24 := by
    rw [← zpow_natCast, hnat, zpow_add₀ (by norm_num)]
    norm_num
  cases hs : d.negative <;>
    simp [FloatLib.Numerics.Dyadic.toReal, FloatLib.Numerics.Dyadic.signedSignificand, hs, hp] <;> ring

theorem units_value (word : Nat) (value : Int) (hu : units word = some value) :
    (Model.ofNatBits word : Model FloatFormat.binary16).toReal = (value:ℝ)/2^24 := by
  by_cases hw : word < 65536
  · cases hd : (Model.ofNatBits word : Model FloatFormat.binary16).toDyadic? with
    | none => simp [units, hw, hd] at hu
    | some d =>
      by_cases he : d.exponent ≥ -24
      · simp [units, hw, hd, he] at hu
        rw [Model.toReal_eq, hd]
        simp only
        rw [← hu]
        exact dyadic_units d he
      · simp [units, hw, hd, he] at hu
  · simp [units, hw] at hu

theorem sum_units_value (words : List Nat)
    (valid : ∀ w ∈ words, ∃ v, units w = some v) :
    (words.map fun w => (Model.ofNatBits w : Model FloatFormat.binary16).toReal).sum =
      (((words.map fun w => (units w).getD 0).sum:Int):ℝ)/2^24 := by
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

theorem exponent_lower_bound (x : Model FloatFormat.binary16) (d : FloatLib.Numerics.Dyadic)
    (hd : x.toDyadic? = some d) : -24 ≤ d.exponent := by
  simp only [Model.toDyadic?, show FloatFormat.binary16.isIEEE = true from rfl,
    ↓reduceIte] at hd
  unfold Model.ieeeToDyadic? at hd
  split at hd
  · contradiction
  · dsimp only at hd
    split at hd
    · split at hd
      · cases Option.some.inj hd
        change -24 ≤ (0:Int)
        omega
      · cases Option.some.inj hd
        change -24 ≤ (-24:Int)
        omega
    · rename_i he
      cases Option.some.inj hd
      change -24 ≤ (x.expField : Int)-25
      have hpos : x.expField ≠ 0 := by simpa using he
      omega

theorem units_defined (word : Nat) (hw : word < 65536)
    (hf : Model.isFinite (Model.ofNatBits word : Model FloatFormat.binary16) = true) :
    ∃ v, units word = some v := by
  obtain ⟨d,hd⟩ := Model.exists_toDyadic?_of_isFinite hf
  have he := exponent_lower_bound _ d hd
  simp [units, hw, hd, he]

theorem small_words : ∀k:Fin 1024,
    units (k:Nat)=some ((k:Nat):Int) ∧
    units ((k:Nat)+32768)=some (-((k:Nat):Int)) := by
  decide +kernel

theorem small_round_exact (n:Int) (hn:n.natAbs<1024) :
    Model.roundAt FloatFormat.binary16 ((n:ℝ)/2^24)=(n:ℝ)/2^24 := by
  let w:=if n<0 then n.natAbs+32768 else n.natAbs
  have hu : units w=some n := by
    have h:=small_words ⟨n.natAbs,hn⟩
    dsimp [w]
    split
    · have he : -(n.natAbs:Int)=n := by omega
      simpa only [Fin.val_mk,he] using h.2
    · have he : (n.natAbs:Int)=n := by omega
      simpa only [Fin.val_mk,he] using h.1
  have hr:=units_value w n hu
  have hf : (Model.ofNatBits w : Model FloatFormat.binary16).isFinite=true := by
    apply finite_raw
    dsimp [w]
    split <;> omega
  rw [←hr]
  exact round_preserves_generic nearestEven _ (Model.toReal_genericFormat_of_isFinite _ hf)

theorem sum_round_value (values:Array (Model FloatFormat.binary16))
    (hf:∀v∈values.toList,v.isFinite=true)
    (ho:(Model.sum FloatFormat.binary16 values .nearestEven).isFinite=true) :
    (Model.sum FloatFormat.binary16 values .nearestEven).toReal=
      Model.roundAt FloatFormat.binary16 (Model.Reduction.Semantics.finiteSum values.toList:ℝ) := by
  by_cases hz:Model.Reduction.Semantics.finiteSum values.toList=0
  · have hd : (values.foldl Model.Reduction.Internal.State.pushValue
        ({}:Model.Reduction.Internal.State FloatFormat.binary16)).exact.toRat=0 := by
      rw [Model.Reduction.Internal.array_foldl_pushValue_exact_toRat]
      exact hz
    have hs := (FloatLib.Numerics.Dyadic.toRat_eq_zero_iff _).mp hd
    change (Model.sumWithStatus FloatFormat.binary16 values .nearestEven).value.toReal=_
    rw [Model.Reduction.sumWithStatus_eq_zero_of_finite _ _ _ hf hs,hz]
    simp only [Model.toReal_zero,Rat.cast_zero,Model.roundAt_zero]
  · exact Model.Reduction.sumWithStatus_value_toReal_eq_roundAt_of_finite_nonzero
      FloatFormat.binary16 values (by decide) hf hz ho

theorem decoded_relative_error (words:List Nat)
    (valid:∀w∈words,w<65536 ∧ w%32768<31744)
    (hout:(Model.ofNatBits (FabricF16.predict words):Model FloatFormat.binary16).isFinite=true) :
    |(Model.ofNatBits (FabricF16.predict words):Model FloatFormat.binary16).toReal-
      (words.map fun w=>(Model.ofNatBits w:Model FloatFormat.binary16).toReal).sum|≤
      |(words.map fun w=>(Model.ofNatBits w:Model FloatFormat.binary16).toReal).sum|/2048 := by
  have hf:∀w∈words,w%32768<31744 := fun w hw=>(valid w hw).2
  by_cases hn:(2:ℝ)^(-14:Int)≤|(words.map fun w=>(Model.ofNatBits w:Model FloatFormat.binary16).toReal).sum|
  · exact decoded_normal_error words hf hout hn
  · let values:=words.toArray.map fun w=>(Model.ofNatBits w:Model FloatFormat.binary16)
    let n:=(words.map fun w=>(units w).getD 0).sum
    have hgrid : (words.map fun w=>(Model.ofNatBits w:Model FloatFormat.binary16).toReal).sum=(n:ℝ)/2^24 := by
      apply sum_units_value
      intro w hw
      exact units_defined w (valid w hw).1 (finite_raw w (valid w hw).2)
    have hsmall:n.natAbs<1024 := by
      rw [hgrid] at hn
      have hn' : |(n:ℝ)|<(1024:ℝ) := by
        rw [abs_div,abs_of_pos (by positivity : (0:ℝ)<2^24)] at hn
        norm_num at hn
        linarith
      have hn'' : (n.natAbs:ℝ)<1024 := by simpa using hn'
      exact_mod_cast hn''
    have hv:∀v∈values.toList,v.isFinite=true := by
      intro v hm
      obtain ⟨w,hw,rfl⟩ := List.mem_map.mp (by simpa [values] using hm)
      exact finite_raw w (hf w hw)
    have ho:(Model.sum FloatFormat.binary16 values .nearestEven).isFinite=true := by
      rw [finite_predict words hf] at hout
      exact (normalize_zero_finite _).symm.trans hout
    have he:(Model.Reduction.Semantics.finiteSum values.toList:ℝ)=(n:ℝ)/2^24 := by
      rw [exact_decoded_sum]
      simpa [values,List.map_map,Function.comp_def] using hgrid
    rw [predict_value words hf]
    change |(Model.sum FloatFormat.binary16 values .nearestEven).toReal - _| ≤ _
    rw [sum_round_value values hv ho,he,small_round_exact n hsmall, hgrid]
    simp only [sub_self,abs_zero]
    positivity

#print axioms dyadic_units
#print axioms units_value
#print axioms sum_units_value
#print axioms exponent_lower_bound
#print axioms units_defined
#print axioms small_words
#print axioms small_round_exact
#print axioms sum_round_value
#print axioms decoded_relative_error
end FabricF16ExactBounds
