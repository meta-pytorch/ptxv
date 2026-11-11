/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Exp2Uniform
namespace GB300Math.Exp2Domain
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

def finiteOutputInput (word : Nat) : Prop :=
  word<0x43000000 ∨ (2^31≤word ∧ word≤0xc2fc0000)

theorem finite_output_iff (word : Nat) (hw : word<2^32)
    (hf : Mufu.magnitude word<0x7f800000) :
    (0<Mufu.ex2 word ∧ Mufu.ex2 word<0x7f800000) ↔ finiteOutputInput word := by
  have hz:=Exp2Boundary.zero_bit_range word hw hf
  have hi:=Exp2Boundary.infinite_bit_range word hw hf
  have hc:=Exp2Boundary.no_subnormal_output word hf
  unfold finiteOutputInput
  omega

theorem uniform_error (word : Nat) (hdom : finiteOutputInput word) :
    ∃ x y:Rat,(Model.ofNatBits word:Model FloatFormat.binary32).toRat?=some x ∧
      (Model.ofNatBits (Mufu.ex2 word):Model FloatFormat.binary32).toRat?=some y ∧
      |(y:ℝ)-(2:ℝ)^(x:ℝ)|/((2:ℝ)^(x:ℝ))≤1/2^22 := by
  have hw : word<2^32 := by unfold finiteOutputInput at hdom;omega
  have hf : Mufu.magnitude word<0x7f800000 := by unfold finiteOutputInput at hdom;unfold Mufu.magnitude;omega
  exact Exp2Uniform.finite_output_error word hw hf ((finite_output_iff word hw hf).mpr hdom)

theorem flush_error (word : Nat) (hw : word<2^32)
    (hf : Mufu.magnitude word<0x7f800000) (hflush : 0xc2fc0000<word) :
    ∃ x:Rat,(Model.ofNatBits word:Model FloatFormat.binary32).toRat?=some x ∧
      (Model.ofNatBits (Mufu.ex2 word):Model FloatFormat.binary32).toRat?=some 0 ∧
      |(0:ℝ)-(2:ℝ)^(x:ℝ)|/((2:ℝ)^(x:ℝ))=1 := by
  obtain ⟨x,hx,_⟩:=Exp2Decode.finite word hw hf
  refine ⟨x,hx,?_,?_⟩
  · rw [(Exp2Boundary.zero_bit_range word hw hf).mpr hflush]
    decide +kernel
  · rw [zero_sub,abs_neg,abs_of_pos (Real.rpow_pos_of_pos (by norm_num) _)]
    exact div_self (ne_of_gt (Real.rpow_pos_of_pos (by norm_num) _))

theorem overflow (word : Nat) (hw : word<2^32)
    (hf : Mufu.magnitude word<0x7f800000)
    (hpos : 0x43000000≤word ∧ word<2^31) :
    Mufu.ex2 word=0x7f800000 ∧
    (Model.ofNatBits (Mufu.ex2 word):Model FloatFormat.binary32).toRat?=none := by
  have h:=(Exp2Boundary.infinite_bit_range word hw hf).mpr hpos
  refine ⟨h,?_⟩
  rw [h]
  decide +kernel

theorem all_finite_classified (word : Nat) (hw : word<2^32)
    (_hf : Mufu.magnitude word<0x7f800000) :
    finiteOutputInput word ∨ 0xc2fc0000<word ∨ (0x43000000≤word ∧ word<2^31) := by
  unfold finiteOutputInput
  omega

theorem finite_output_count :
    (Finset.Ico (0:Nat) 0x43000000).card+
    (Finset.Icc (2^31:Nat) 0xc2fc0000).card=2247884801 := by norm_num

theorem flush_count : (Finset.Ioo (0xc2fc0000:Nat) 0xff800000).card=1015283711 := by norm_num

theorem overflow_count : (Finset.Ico (0x43000000:Nat) 0x7f800000).card=1015021568 := by norm_num

theorem count_partition : (2247884801+1015283711+1015021568:Nat)=2*255*2^23 := by decide +kernel

theorem infinity_results : Mufu.ex2 0x7f800000=0x7f800000 ∧ Mufu.ex2 0xff800000=0 := by decide +kernel

theorem nan_result (word : Nat) (hnan : 0x7f800000<Mufu.magnitude word) :
    Mufu.ex2 word=Mufu.canonicalNaN := by simp only [Mufu.ex2,hnan,ite_true]

#print axioms finite_output_iff
#print axioms uniform_error
#print axioms flush_error
#print axioms overflow
#print axioms all_finite_classified
#print axioms finite_output_count
#print axioms flush_count
#print axioms overflow_count
#print axioms count_partition
#print axioms infinity_results
#print axioms nan_result
end GB300Math.Exp2Domain
