/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import RadianSigned
namespace GB300Math.RadianTiny
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 16000
set_option maxHeartbeats 6000000

theorem fields_word (negative : Bool) (e f : Nat) (he:e<255) (hf:f<2^23) :
    Approx.asWord (Model.ofFields FloatFormat.binary32 negative e f)=
      (if negative then 2^31 else 0)+e*2^23+f := by
  have hfinite:=Model.isFinite_ofFields_ieee FloatFormat.binary32 (by decide) negative e f he
  have hn:=Model.isNaN_eq_false_of_isFinite_eq_true _ hfinite
  simp only [Approx.asWord,hn,Bool.false_eq_true,ite_false]
  exact RcpPtxBounds.fields_bits negative e f (by omega) hf

theorem zero_word (negative : Bool) :
    Approx.asWord (Model.zero FloatFormat.binary32 negative)=(if negative then 2^31 else 0) := by
  cases negative <;> decide +kernel

theorem shift (n d : Nat) : Nat.shiftRight n d=n/2^d := Nat.shiftRight_eq_div_pow n d

theorem field_result (negative : Bool) (r : Nat) (hr : r<2^23) :
    ∃w:Nat,w<2^24 ∧ Approx.asWord (Model.ofFields FloatFormat.binary32 negative 0 r)=(if negative then 2^31 else 0)+w := by
  refine ⟨r,by omega,?_⟩
  simpa using fields_word negative 0 r (by decide) hr

theorem down_tiny (negative : Bool) (n : Nat) (q : Int)
    (hn : 0 < n ∧ n < 2^48) (hq : -176 ≤ q ∧ q ≤ -173) :
    ∃ w : Nat, w < 2^24 ∧ Approx.asWord (Model.roundDyadicMagnitudeDown FloatFormat.binary32 negative n q)=
      (if negative then 2^31 else 0)+w := by
  have hl : n.log2 ≤ 47 := by have h:=(Nat.log2_lt (by omega : n≠0)).mpr hn.2;omega
  have hhigh : ¬127<(n.log2:Int)+q := by omega
  have hmax : (n.log2:Int)+q≤ -126 := by omega
  unfold Model.roundDyadicMagnitudeDown
  dsimp only
  change ∃w,w<2^24 ∧ Approx.asWord (if (n.log2:Int)+q>127 then _ else if (n.log2:Int)+q< -149 then _ else if (n.log2:Int)+q< -126 then _ else _)=(if negative then 2^31 else 0)+w
  rw [ite_eq_right hhigh]
  by_cases hunder : (n.log2:Int)+q< -149
  · rw [ite_eq_left hunder]
    refine ⟨0,by norm_num,?_⟩
    simpa using zero_word negative
  rw [ite_eq_right hunder]
  by_cases hsub : (n.log2:Int)+q< -126
  · rw [ite_eq_left hsub]
    have hbound : n<2^((-q-126).toNat) := (Nat.log2_lt (by omega : n≠0)).mp (by omega)
    have hcases : q= -176 ∨ q= -175 ∨ q= -174 ∨ q= -173 := by omega
    rcases hcases with rfl|rfl|rfl|rfl
    all_goals
      change ∃w,w<2^24 ∧ Approx.asWord (if (Nat.shiftRight n _)==0 then Model.zero _ negative else Model.ofFields _ negative 0 (Nat.shiftRight n _))=(if negative then 2^31 else 0)+w
      rw [shift]
      norm_num at hbound ⊢
      split
      · refine ⟨0,by norm_num,?_⟩
        simpa using zero_word negative
      · exact field_result negative _ (by omega)
  rw [ite_eq_right hsub]
  have htotal : (n.log2:Int)+q= -126 := by omega
  have hlead : n.log2=47 := by omega
  have hq173 : q= -173 := by omega
  have hlog0:=Nat.log2_self_le (by omega : n≠0)
  have hlog1:=Nat.lt_log2_self (n:=n)
  rw [hlead] at hlog0 hlog1
  have hmant : 2^23≤n/2^24 ∧ n/2^24<2^24 := by norm_num at *;omega
  rw [hlead,hq173]
  change ∃w,w<2^24 ∧ Approx.asWord (Model.ofFields FloatFormat.binary32 negative 1 (Nat.shiftRight n 24-8388608))=(if negative then 2^31 else 0)+w
  rw [shift]
  refine ⟨n/2^24,hmant.2,?_⟩
  rw [fields_word negative 1 _ (by decide) (by omega)]
  omega

#print axioms fields_word
#print axioms zero_word
#print axioms down_tiny
end GB300Math.RadianTiny
