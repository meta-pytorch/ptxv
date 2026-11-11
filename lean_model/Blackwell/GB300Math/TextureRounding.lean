/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureSeamlessIdeal
import GB300Math.TextureSeamlessBounds

set_option maxRecDepth 10000
set_option maxHeartbeats 800000

namespace GB300Math.TextureRounding

open TextureSeamless

def rounded (n : Nat) : Nat := n/4096 + if n%4096 ≥ 2048 then 1 else 0

theorem rounded_bounds (n : Nat) (hl : 2^35 ≤ n) (hu : n < 2^36) :
    2^23 ≤ rounded n ∧ rounded n ≤ 2^24 := by
  unfold rounded
  norm_num at *
  split_ifs <;> omega

theorem rounded_error (n : Nat) :
    |(rounded n : Rat)*4096 - (n:Rat)| ≤ 2048 := by
  have hm : n%4096 < 4096 := Nat.mod_lt _ (by decide)
  have hnonnegative : (0:Rat) ≤ (n%4096:Nat) := by positivity
  have hi := Nat.mod_add_div n 4096
  have hir : (n%4096:Nat) + (4096:Rat)*(n/4096:Nat) = n := by exact_mod_cast hi
  unfold rounded
  split_ifs with h
  · have hr : (2048:Rat) ≤ (n%4096:Nat) := by exact_mod_cast h
    have hr' : (n%4096:Nat) < (4096:Rat) := by exact_mod_cast hm
    rw [Nat.cast_add]
    rw [abs_le]
    constructor <;> norm_num only [Nat.cast_one] <;> linarith
  · have hr : (n%4096:Nat) < (2048:Rat) := by exact_mod_cast (show n%4096 < 2048 by omega)
    rw [Nat.add_zero, abs_le]
    constructor <;> linarith

theorem encode_unit (n : Nat) (hl : 2^35 ≤ n) (hu : n < 2^36) :
    encode (n:Int) (-35) = 0x3f800000 + (rounded n - 2^23) := by
  have hn : n ≠ 0 := by omega
  have hnpos : ¬ (n:Int) < 0 := by omega
  have hlog : n.log2 = 35 := (Nat.log2_eq_iff hn).2 ⟨hl, hu⟩
  have hb := rounded_bounds n hl hu
  simp only [encode, Int.natCast_eq_zero, hn, ↓reduceIte, hnpos,
    Int.natAbs_natCast, hlog]
  norm_num
  have hr : n/4096 + (if n%4096 ≥ 2048 then 1 else 0) = rounded n := rfl
  simp only [hr]
  have hzero : ¬ (n < 4096 ∧ n % 4096 < 2048) := by norm_num at hl; omega
  simp only [hzero, ↓reduceIte]
  by_cases hc : 16777216 ≤ rounded n
  · have hq : rounded n = 16777216 := by norm_num at hb; omega
    have hlq : Nat.log2 8388608 = 23 := by decide +kernel
    norm_num [hq, hlq]
  · have hq : (rounded n).log2 = 23 := (Nat.log2_eq_iff (by norm_num at hb; omega)).2 (by norm_num at *; omega)
    simp only [hc, ↓reduceIte, hq]
    norm_num
    norm_num at hb
    omega

theorem unit_code_value (q : Nat) (hl : 8388608 ≤ q) (hu : q ≤ 16777216) :
    TextureSeamlessIdeal.value (1065353216+(q-8388608)) = (q:Rat)/8388608 := by
  by_cases hq : q = 16777216
  · subst q
    decide +kernel
  · have hc : 1065353216+(q-8388608) < 2147483648 := by omega
    have hmod : (1065353216+(q-8388608))%2147483648 = 1065353216+(q-8388608) := by omega
    have he : (1065353216+(q-8388608))/8388608 = 127 := by omega
    have hm : (1065353216+(q-8388608))%8388608 = q-8388608 := by omega
    have hn : ¬ 1065353216+(q-8388608) ≥ 4294967296 := by omega
    have hf : ¬ 1065353216+(q-8388608) > 2139095039 := by omega
    have hs : (1065353216+(q-8388608))/2147483648 = 0 := by omega
    unfold TextureSeamlessIdeal.value StochasticValue.rationalValue
    norm_num only [Nat.reduceAdd, Nat.reducePow]
    simp only [hmod, hn, hf, decide_false, Bool.false_or, Bool.false_eq_true,
      ↓reduceIte, he, hm, hs, Option.getD_some]
    change ((q-8388608+8388608:Nat):Rat) * (2:Rat)^(-23:Int) = (q:Rat)/8388608
    rw [Nat.sub_add_cancel hl]
    norm_num [zpow_neg, div_eq_mul_inv]

theorem encode_unit_value (n : Nat) (hl : 2^35 ≤ n) (hu : n < 2^36) :
    TextureSeamlessIdeal.value (encode (n:Int) (-35)) = (rounded n:Rat)/8388608 := by
  rw [encode_unit n hl hu]
  have hb := rounded_bounds n hl hu
  exact unit_code_value (rounded n) hb.1 hb.2

theorem encode_unit_error (n : Nat) (hl : 2^35 ≤ n) (hu : n < 2^36) :
    |TextureSeamlessIdeal.value (encode (n:Int) (-35)) - (n:Rat)/2^35| ≤ 1/2^24 := by
  rw [encode_unit_value n hl hu]
  have h := rounded_error n
  have hdiv : |((rounded n:Rat)*4096-(n:Rat))/(2:Rat)^35| ≤ 2048/(2:Rat)^35 := by
    rw [abs_div, abs_of_pos (by positivity : (0:Rat) < 2^35)]
    exact div_le_div_of_nonneg_right h (by positivity)
  have heq : ((rounded n:Rat)*4096-(n:Rat))/2^35 = (rounded n:Rat)/8388608-(n:Rat)/2^35 := by ring
  rw [heq] at hdiv
  norm_num at hdiv ⊢
  exact hdiv

def weightedWith (enc : Int → Int → Nat) (ws geometric values : Array Nat) : Nat := Id.run do
  if geometric.any (fun i => magnitude values[i]! > 0x7f800000) then return 0x7fffffff
  let infs := geometric.filter (fun i => magnitude values[i]! = 0x7f800000)
  if !infs.isEmpty then
    let sign := negative values[infs[0]!]!
    if infs.any (fun i => negative values[i]! != sign) then return 0x7fffffff
    return (if sign then 2^31 else 0)+0x7f800000
  let finite := geometric.filter (fun i => ws[i]! > 0 && exponent values[i]! > 0)
  if finite.isEmpty then
    return if geometric.all (fun i => negative values[i]! && exponent values[i]! = 0)
      then 2^31 else 0
  let top := finite.foldl (fun e i => max e (exponent values[i]!)) 0
  let scale := (top : Int)-154
  let mut total : Int := 0
  for i in finite do
    let value := values[i]!
    let mantissa := value%2^23+2^23
    let shift := (exponent value : Int)-150-scale
    let aligned : Nat := if shift ≥ 0 then mantissa*2^shift.toNat else mantissa/2^(-shift).toNat
    let signed : Int := if negative value then -(aligned : Int) else aligned
    total := total + signed*(ws[i]! : Int)
  return enc total (scale-8)


theorem weighted_specialization (ws geometric values : Array Nat) :
    weighted ws geometric values = weightedWith encode ws geometric values := rfl

theorem weighted_with_unit (enc : Int → Int → Nat) (a b c : Nat) (ha : a < 8388608) (hb : b < 8388608) (hc : c < 8388608) :
    weightedWith enc #[86,85,85] #[0,1,2] #[0x3f800000+a,0x3f800000+b,0x3f800000+c] =
      enc (16*(2147483648+86*(a:Int)+85*(b:Int)+85*(c:Int))) (-35) := by
  have fields (t : Nat) (ht : t < 8388608) :
      magnitude (0x3f800000+t) = 0x3f800000+t ∧ exponent (0x3f800000+t) = 127 ∧
      (0x3f800000+t)%8388608 = t ∧ negative (0x3f800000+t) = false := by
    norm_num [magnitude, exponent, negative]
    omega
  obtain ⟨ha1,ha2,ha3,ha4⟩ := fields a ha
  obtain ⟨hb1,hb2,hb3,hb4⟩ := fields b hb
  obtain ⟨hc1,hc2,hc3,hc4⟩ := fields c hc
  have ha5 : ¬ 2139095040 < 1065353216+a := by omega
  have hb5 : ¬ 2139095040 < 1065353216+b := by omega
  have hc5 : ¬ 2139095040 < 1065353216+c := by omega
  have ha6 : 1065353216+a ≠ 2139095040 := by omega
  have hb6 : 1065353216+b ≠ 2139095040 := by omega
  have hc6 : 1065353216+c ≠ 2139095040 := by omega
  have ha7 : (1065353216+(a:Int))%8388608 = (a:Int) := by omega
  have hb7 : (1065353216+(b:Int))%8388608 = (b:Int) := by omega
  have hc7 : (1065353216+(c:Int))%8388608 = (c:Int) := by omega
  have hn : (#[0,1,2] : Array Nat).any (fun i => magnitude (#[1065353216+a,1065353216+b,1065353216+c] : Array Nat)[i]! > 2139095040) = false := by
    simp [ha1,hb1,hc1,ha5,hb5,hc5]
  have hi : (#[0,1,2] : Array Nat).filter (fun i => magnitude (#[1065353216+a,1065353216+b,1065353216+c] : Array Nat)[i]! = 2139095040) = #[] := by
    simp [ha1,hb1,hc1,ha6,hb6,hc6]
  have hf : (#[0,1,2] : Array Nat).filter (fun i => (#[86,85,85] : Array Nat)[i]! > 0 && exponent (#[1065353216+a,1065353216+b,1065353216+c] : Array Nat)[i]! > 0) = #[0,1,2] := by
    simp [ha2,hb2,hc2]
  have ht : (#[0,1,2] : Array Nat).foldl (fun e i => max e (exponent (#[1065353216+a,1065353216+b,1065353216+c] : Array Nat)[i]!)) 0 = 127 := by
    simp [ha2,hb2,hc2]
  unfold weightedWith
  simp only [hn,hi,hf,ht]
  conv_lhs =>
    simp [ha2,ha4,hb2,hb4,hc2,hc4,ha7,hb7,hc7]
  apply congrArg (fun z : Int => enc z (-35))
  ring


theorem weighted_unit (a b c : Nat) (ha : a < 8388608) (hb : b < 8388608) (hc : c < 8388608) :
    weighted #[86,85,85] #[0,1,2] #[0x3f800000+a,0x3f800000+b,0x3f800000+c] =
      encode (16*(2147483648+86*(a:Int)+85*(b:Int)+85*(c:Int))) (-35) := by
  rw [weighted_specialization]
  exact weighted_with_unit encode a b c ha hb hc

def unitNumerator (a b c : Nat) : Nat := 16*(2147483648+86*a+85*b+85*c)
def unitValue (a : Nat) : Rat := 1+(a:Rat)/8388608

theorem unit_numerator_bounds (a b c : Nat)
    (ha : a < 8388608) (hb : b < 8388608) (hc : c < 8388608) :
    2^35 ≤ unitNumerator a b c ∧ unitNumerator a b c < 2^36 := by
  unfold unitNumerator
  norm_num
  omega

theorem weighted_value_error (a b c : Nat)
    (ha : a < 8388608) (hb : b < 8388608) (hc : c < 8388608) :
    |TextureSeamlessIdeal.value (weighted #[86,85,85] #[0,1,2]
       #[0x3f800000+a,0x3f800000+b,0x3f800000+c]) -
      TextureSeamlessBounds.corner (unitValue a) (unitValue b) (unitValue c)| ≤ 1/2^24 := by
  have bounds := unit_numerator_bounds a b c ha hb hc
  have h := encode_unit_error (unitNumerator a b c) bounds.1 bounds.2
  have hn : (unitNumerator a b c : Int) = 16*(2147483648+86*(a:Int)+85*(b:Int)+85*(c:Int)) := by
    unfold unitNumerator
    push_cast
    ring
  have hv : (unitNumerator a b c:Rat)/2^35 =
      TextureSeamlessBounds.corner (unitValue a) (unitValue b) (unitValue c) := by
    unfold unitNumerator unitValue TextureSeamlessBounds.corner
    push_cast
    ring
  rw [hn,hv] at h
  rw [weighted_unit a b c ha hb hc]
  exact h

theorem unit_value_bounds (a : Nat) (ha : a < 8388608) :
    1 ≤ unitValue a ∧ unitValue a ≤ 2 := by
  have hl : (0:Rat) ≤ a := by positivity
  have hu : (a:Rat) < 8388608 := by exact_mod_cast ha
  unfold unitValue
  constructor <;> linarith

theorem weighted_equal_thirds_error (a b c : Nat)
    (ha : a < 8388608) (hb : b < 8388608) (hc : c < 8388608) :
    |TextureSeamlessIdeal.value (weighted #[86,85,85] #[0,1,2]
       #[0x3f800000+a,0x3f800000+b,0x3f800000+c]) -
      TextureSeamlessBounds.ideal (unitValue a) (unitValue b) (unitValue c)| ≤ 1/384+1/2^24 := by
  have h := weighted_value_error a b c ha hb hc
  have bnd := TextureSeamlessBounds.range_error (unitValue a) (unitValue b) (unitValue c) 1 2
    (unit_value_bounds a ha) (unit_value_bounds b hb) (unit_value_bounds c hc)
  have triangle := abs_sub_le
    (TextureSeamlessIdeal.value (weighted #[86,85,85] #[0,1,2]
      #[0x3f800000+a,0x3f800000+b,0x3f800000+c]))
    (TextureSeamlessBounds.corner (unitValue a) (unitValue b) (unitValue c))
    (TextureSeamlessBounds.ideal (unitValue a) (unitValue b) (unitValue c))
  norm_num at bnd
  linarith

theorem unit_input_decode (a : Nat) (ha : a < 8388608) :
    TextureSeamlessIdeal.value (0x3f800000+a) = unitValue a := by
  have h := unit_code_value (8388608+a) (by omega) (by omega)
  have hs : 8388608+a-8388608 = a := by omega
  rw [hs] at h
  rw [h]
  unfold unitValue
  push_cast
  ring

theorem weighted_bit_ideal_error (a b c : Nat)
    (ha : a < 8388608) (hb : b < 8388608) (hc : c < 8388608) :
    |TextureSeamlessIdeal.value (weighted #[86,85,85] #[0,1,2]
       #[0x3f800000+a,0x3f800000+b,0x3f800000+c]) -
      TextureSeamlessBounds.ideal (TextureSeamlessIdeal.value (0x3f800000+a))
        (TextureSeamlessIdeal.value (0x3f800000+b))
        (TextureSeamlessIdeal.value (0x3f800000+c))| ≤ 1/384+1/2^24 := by
  rw [unit_input_decode a ha, unit_input_decode b hb, unit_input_decode c hc]
  exact weighted_equal_thirds_error a b c ha hb hc

theorem weighted_range_error (a b c : Nat) (lo hi : Rat)
    (ha : a < 8388608) (hb : b < 8388608) (hc : c < 8388608)
    (ha' : lo ≤ unitValue a ∧ unitValue a ≤ hi)
    (hb' : lo ≤ unitValue b ∧ unitValue b ≤ hi)
    (hc' : lo ≤ unitValue c ∧ unitValue c ≤ hi) :
    |TextureSeamlessIdeal.value (weighted #[86,85,85] #[0,1,2]
       #[0x3f800000+a,0x3f800000+b,0x3f800000+c]) -
      TextureSeamlessBounds.ideal (unitValue a) (unitValue b) (unitValue c)| ≤
       (hi-lo)/384+1/2^24 := by
  have h := weighted_value_error a b c ha hb hc
  have bnd := TextureSeamlessBounds.range_error (unitValue a) (unitValue b) (unitValue c) lo hi ha' hb' hc'
  have triangle := abs_sub_le
    (TextureSeamlessIdeal.value (weighted #[86,85,85] #[0,1,2]
      #[0x3f800000+a,0x3f800000+b,0x3f800000+c]))
    (TextureSeamlessBounds.corner (unitValue a) (unitValue b) (unitValue c))
    (TextureSeamlessBounds.ideal (unitValue a) (unitValue b) (unitValue c))
  linarith

#print axioms unit_input_decode
#print axioms weighted_bit_ideal_error
#print axioms weighted_range_error

#print axioms rounded_bounds
#print axioms rounded_error
#print axioms encode_unit
#print axioms unit_code_value
#print axioms encode_unit_value
#print axioms encode_unit_error
#print axioms weighted_specialization
#print axioms weighted_with_unit
#print axioms weighted_unit
#print axioms unit_numerator_bounds
#print axioms weighted_value_error
#print axioms unit_value_bounds
#print axioms weighted_equal_thirds_error
end GB300Math.TextureRounding
