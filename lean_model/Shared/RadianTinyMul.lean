/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import RadianTiny
namespace GB300Math.RadianTinyMul
open FloatLib.Floats.Formats.BinaryInterchange
open RadianTiny
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

theorem rounded (negative : Bool) (n : Nat) (q : Int) (hn : n<2^48)
    (hq : n≠0 → -176≤q ∧ q≤ -173) :
    ∃w:Nat,w<2^24 ∧ Approx.asWord (Model.roundDyadicWithRounding FloatFormat.binary32 .towardZero ⟨negative,n,q⟩)=
      (if negative then 2^31 else 0)+w := by
  by_cases hz:n=0
  · subst n
    refine ⟨0,by norm_num,?_⟩
    simpa only [Model.roundDyadicWithRounding,Model.roundDyadicTowardZero,beq_self_eq_true,ite_true,Nat.add_zero] using zero_word negative
  · have h:=down_tiny negative n q (by omega) (hq hz)
    simpa only [Model.roundDyadicWithRounding,Model.roundDyadicTowardZero,
      show (n==0)=false from by simp[hz],Bool.false_eq_true,ite_false] using h

theorem mul_small (negative : Bool) (e f : Nat) (he:e<4) (hf:f<2^23) :
    ∃w:Nat,w<2^24 ∧ Approx.mul ((if negative then 2^31 else 0)+(e*2^23+f)) 0x3e22f983 true=
      (if negative then 2^31 else 0)+w := by
  have hleft : Approx.asFloat ((if negative then 2^31 else 0)+(e*2^23+f))=
      Model.ofFields FloatFormat.binary32 negative e f := by
    simpa [Nat.add_assoc] using (RcpPtxBounds.fields_float negative e f (by omega) hf).symm
  have hright : Approx.asFloat 0x3e22f983=Model.ofFields FloatFormat.binary32 false 124 2292099 := by
    simpa using (RcpPtxBounds.fields_float false 124 2292099 (by norm_num) (by norm_num)).symm
  change ∃w,w<2^24 ∧ Approx.asWord (Model.mulWithRounding .towardZero _ _)=(if negative then 2^31 else 0)+w
  rw [hleft,hright]
  have hdright:=RcpPtxBounds.decode_normal false 124 2292099 (by norm_num) (by norm_num)
  by_cases he0:e=0
  · subst e
    by_cases hf0:f=0
    · subst f
      have hd:=Model.toDyadic?_ofFields_of_isFinite FloatFormat.binary32 negative 0 0 (by decide) (by decide)
        (Model.isFinite_ofFields_ieee FloatFormat.binary32 (by decide) negative 0 0 (by decide))
      simp only [ite_true] at hd
      rw [Model.mulWithRounding_eq_of_toDyadic?_some .towardZero (by decide) hd hdright]
      simp only [Bool.xor_false,Nat.zero_mul,Int.zero_add]
      norm_num only [Nat.cast_ofNat,Int.reduceSub]
      exact rounded negative 0 (-26) (by norm_num) (by simp)
    · have hd:=Model.toDyadic?_ofFields_subnormal FloatFormat.binary32 negative f hf0 hf
      rw [Model.mulWithRounding_eq_of_toDyadic?_some .towardZero (by decide) hd hdright]
      simp only [Bool.xor_false,show FloatFormat.binary32.minSubnormalExponent=(-149:Int) from rfl]
      norm_num only [Nat.reducePow,Nat.reduceAdd,Nat.cast_ofNat,Int.reduceSub,Int.reduceAdd]
      change ∃w,w<2^24 ∧ Approx.asWord (Model.roundDyadicWithRounding FloatFormat.binary32 .towardZero ⟨negative,f*10680707,-175⟩)=(if negative then 2^31 else 0)+w
      exact rounded negative (f*10680707) (-175) (by omega) (by omega)
  · have hd:=RcpPtxBounds.decode_normal negative e f (by omega) hf
    rw [Model.mulWithRounding_eq_of_toDyadic?_some .towardZero (by decide) hd hdright]
    have hexp : ((e:Int)-150)+(124-150)=(e:Int)-176 := by omega
    simp only [Bool.xor_false]
    norm_num only [Nat.reducePow,Nat.reduceAdd,Nat.cast_ofNat] at hexp ⊢
    rw [hexp]
    change ∃w,w<2^24 ∧ Approx.asWord (Model.roundDyadicWithRounding FloatFormat.binary32 .towardZero ⟨negative,(2^23+f)*10680707,(e:Int)-176⟩)=(if negative then 2^31 else 0)+w
    exact rounded negative _ _ (by omega) (by omega)

#print axioms rounded
#print axioms mul_small
end GB300Math.RadianTinyMul
