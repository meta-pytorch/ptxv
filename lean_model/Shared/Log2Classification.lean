/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2Uniform
import SineSubnormal
namespace GB300Math.Log2Classification
set_option maxRecDepth 10000
set_option maxHeartbeats 3000000

theorem zero_exponent (negative : Bool) (f : Fin (2^23)) :
    Mufu.lg2 ((if negative then 2^31 else 0)+f.val)=0xff800000 := by
  have hf:=f.isLt
  have he : Mufu.exponent ((if negative then 2^31 else 0)+f.val)=0 := by
    cases negative <;> simp only [Bool.false_eq_true,ite_false,ite_true,Nat.zero_add] <;> unfold Mufu.exponent <;> omega
  have hm : ¬Mufu.magnitude ((if negative then 2^31 else 0)+f.val)>0x7f800000 := by
    cases negative <;> simp only [Bool.false_eq_true,ite_false,ite_true,Nat.zero_add] <;> unfold Mufu.magnitude <;> omega
  simp only [Mufu.lg2,hm,ite_false,he,ite_true]

theorem negative_normal (e : Fin 254) (f : Fin (2^23)) :
    Mufu.lg2 (2^31+(1+e.val)*2^23+f.val)=0x7fffffff := by
  have he:=e.isLt
  have hf:=f.isLt
  have hx : Mufu.exponent (2^31+(1+e.val)*2^23+f.val)=1+e.val := by unfold Mufu.exponent;omega
  have hs : Mufu.sign (2^31+(1+e.val)*2^23+f.val)=2^31 := by
    have h := RcpSigned.sign_set ((1+e.val)*2^23+f.val) (by omega)
    simpa only [Nat.add_assoc] using h
  have hm : ¬Mufu.magnitude (2^31+(1+e.val)*2^23+f.val)>0x7f800000 := by unfold Mufu.magnitude;omega
  simp only [Mufu.lg2,hm,ite_false,hx,hs,Mufu.canonicalNaN]
  simp

theorem infinities : Mufu.lg2 0x7f800000=0x7f800000 ∧ Mufu.lg2 0xff800000=0x7fffffff := by decide +kernel

theorem nan_bits (w : Nat) (hn : Mufu.magnitude w>0x7f800000) : Mufu.lg2 w=0x7fffffff := by
  simp [Mufu.lg2,hn,Mufu.canonicalNaN]

theorem positive_subnormal_nonfinite (f : Fin (2^23)) (hf : 0<f.val) :
    ∃ x : Rat,Root32FloatLib.value f.val=some x ∧ 0<x ∧
      Root32FloatLib.value (Mufu.lg2 f.val)=none := by
  have h:=zero_exponent false f
  simp only [Bool.false_eq_true,ite_false,Nat.zero_add] at h
  refine ⟨_,SineSubnormal.decode f,?_,?_⟩
  · positivity
  · rw [h];decide +kernel

theorem counts : 254*2^23=2130706432 ∧ 2*2^23=16777216 := by decide +kernel
#print axioms zero_exponent
#print axioms negative_normal
#print axioms infinities
#print axioms nan_bits
#print axioms positive_subnormal_nonfinite
#print axioms counts
end GB300Math.Log2Classification
