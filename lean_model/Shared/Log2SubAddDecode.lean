/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2SubAdd
namespace GB300Math.Log2SubAdd
open FloatLib.Floats.Formats.BinaryInterchange
open Root32FloatLib
set_option maxHeartbeats 500000
set_option maxRecDepth 16000

theorem decoded_generic (f l r out : Nat)
    (hl0 : 25≤l) (hl1 : l≤26) (hr0 : 2^23≤r) (hr1 : r<2^24)
    (hw : out=2^31+(l+108)*2^23+(r-2^23))
    (hd0 : amplitude f/2^(l-23)≤r) (hd1 : r≤amplitude f/2^(l-23)+1) :
    ∃ y : Rat,value out=some y ∧ |(y:ℝ)+((2^23+(f:ℝ))/2^17+24)|≤1/2^16 := by
  have hd:=RootPtxScale.normal_value (l+108) (r-2^23) (by omega) (by omega)
  have hm : 2^23+((r-2^23:Nat):Rat)=(r:Rat) := by exact_mod_cast (show 2^23+(r-2^23)=r by omega)
  rw [hm] at hd
  have he : ((l+108:Nat):Int)-150=(l:Int)-42 := by omega
  rw [he] at hd
  have hneg:=RcpSigned.signed_decode ((l+108)*2^23+(r-2^23)) ((r:Rat)*2^((l:Int)-42)) hd
  refine ⟨-((r:Rat)*2^((l:Int)-42)),?_,?_⟩
  · rw [hw]
    have hwadd : 2^31+(l+108)*2^23+(r-2^23)=(l+108)*2^23+(r-2^23)+2^31 := by omega
    rw [hwadd]
    exact hneg
  · have hdlo : amplitude f<r*2^(l-23)+2^(l-23) := by
      have hm:=Nat.mod_lt (amplitude f) (by positivity : 0<2^(l-23))
      have heq:=Nat.mod_add_div (amplitude f) (2^(l-23))
      have hmle:=Nat.mul_le_mul_right (2^(l-23)) hd0
      nlinarith only [hmle,hm,heq]
    have hdhi : r*2^(l-23)≤amplitude f+2^(l-23) := by
      have hmul:=Nat.div_mul_le_self (amplitude f) (2^(l-23))
      have hmle:=Nat.mul_le_mul_right (2^(l-23)) hd1
      nlinarith only [hmle,hmul]
    have hlor : (amplitude f:ℝ)<(r:ℝ)*(2^(l-23):Nat)+(2^(l-23):Nat) := by exact_mod_cast hdlo
    have hhir : (r:ℝ)*(2^(l-23):Nat)≤(amplitude f:ℝ)+(2^(l-23):Nat) := by exact_mod_cast hdhi
    dsimp [amplitude] at hlor hhir
    push_cast at hlor hhir ⊢
    rcases (show l=25 ∨ l=26 by omega) with he|he
    all_goals
      rw [he] at hlor hhir ⊢
      norm_num at hlor hhir ⊢
      apply abs_le.mpr
      constructor <;> linarith only[hlor,hhir]

theorem decoded_add (f : Nat) (hf : f<2^23) :
    ∃ y : Rat,value (result f)=some y ∧
      |(y:ℝ)+((2^23+(f:ℝ))/2^17+24)|≤1/2^16 := by
  obtain ⟨hl0,hl1,hr0,hr1,hw,hd0,hd1⟩:=rounded_data f hf
  exact decoded_generic f _ _ (result f) hl0 hl1 hr0 hr1 hw hd0 hd1

#print axioms decoded_generic
#print axioms decoded_add
end GB300Math.Log2SubAdd
