/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2SubAdd
namespace GB300Math.Log2SubShape
set_option maxHeartbeats 500000
set_option maxRecDepth 16000

theorem output_shape (w : Nat) (hw : 2*2^23≤w ∧ w<25*2^23) :
    ∃ f : Nat,f<2^23 ∧ Mufu.lg2 w=2^31+133*2^23+f := by
  let e : Fin 254 := ⟨w/2^23-1,by omega⟩
  let f : Fin (2^23) := ⟨w%2^23,by omega⟩
  have hid : (1+e.val)*2^23+f.val=w := by dsimp [e,f];omega
  have he : 1≤e.val ∧ e.val≤23 := by dsimp[e];omega
  have hmodel:=Log2Uniform.model_normal e f (by omega)
  rw [hid] at hmodel
  have h := Log2Interpolation.interpolation_facts f.val f.isLt
  dsimp only at h
  let v:=Log2Uniform.fixedValue e f
  have hv : -(2^43:Int)<v ∧ v< -(2^42:Int) := by dsimp[v,Log2Uniform.fixedValue];omega
  let amp:=(-v-1).toNat
  have hm : 2^42≤amp ∧ amp<2^43 := by dsimp[amp];omega
  have hl : amp.log2=42 := by
    have h0:=(Nat.le_log2 (by omega : amp≠0)).mpr hm.1
    have h1:=(Nat.log2_lt (by omega : amp≠0)).mpr hm.2
    omega
  refine ⟨(amp/2^19)%2^23,by omega,?_⟩
  rw [hmodel]
  change Log2Encoding.encoded v=_
  rw [Log2Encoding.encoded,ite_eq_right (by omega : ¬v>0),ite_eq_right (by dsimp[amp] at hm;omega : ¬(-v-1).toNat=0)]
  change 2^31+Mufu.fixedToWord amp 36=_
  simp only [Mufu.fixedToWord,hl]
  norm_num
  omega

#print axioms output_shape
end GB300Math.Log2SubShape
