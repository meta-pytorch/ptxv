/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2Encoding
import Log2Input
namespace GB300Math.Log2Uniform
set_option maxHeartbeats 5000000
set_option maxRecDepth 10000

def fixedValue (e : Fin 254) (f : Fin (2^23)) : Int :=
  ((e.val:Int)-126)*2^36+Mufu.interpolate MufuTables.lg2 f.val

theorem model_normal (e : Fin 254) (f : Fin (2^23)) (hne : (1+e.val)*2^23+f.val≠0x3f800000) :
    Mufu.lg2 ((1+e.val)*2^23+f.val)=Log2Encoding.encoded (fixedValue e f) := by
  have he:=e.isLt
  have hf:=f.isLt
  have hexp : Mufu.exponent ((1+e.val)*2^23+f.val)=1+e.val := by unfold Mufu.exponent;omega
  have hfrac : Mufu.fraction ((1+e.val)*2^23+f.val)=f.val := by unfold Mufu.fraction;omega
  have hmag : ¬Mufu.magnitude ((1+e.val)*2^23+f.val)>0x7f800000 := by unfold Mufu.magnitude;omega
  have hsign : Mufu.sign ((1+e.val)*2^23+f.val)=0 := RcpSigned.sign_zero _ (by omega)
  simp only [Mufu.lg2,hmag,ite_false,hexp,hfrac,hsign,hne]
  have he0 : ¬1+e.val=0 := by omega
  have he255 : ¬1+e.val=255 := by omega
  simp only [he0,he255,ne_eq,not_true_eq_false,ite_false]
  have hx : ((1+e.val:Nat):Int)-127=(e.val:Int)-126 := by omega
  rw [hx]
  have hm : (-((e.val:Int)-126)-1)*2^36+(2^36-1-Mufu.interpolate MufuTables.lg2 f.val)=
      -fixedValue e f-1 := by dsimp [fixedValue];ring
  rw [hm]
  rfl

theorem fixed_range (e : Fin 254) (f : Fin (2^23)) :
    -(2^43:Int)<fixedValue e f ∧ fixedValue e f<2^43 := by
  have h:=Log2Interpolation.interpolation_facts f.val f.isLt
  dsimp only at h
  have he:=e.isLt
  dsimp [fixedValue]
  omega

theorem positive_normal_error (e : Fin 254) (f : Fin (2^23)) :
    ∃ x y : Rat,
      Root32FloatLib.value ((1+e.val)*2^23+f.val)=some x ∧
      Root32FloatLib.value (Mufu.lg2 ((1+e.val)*2^23+f.val))=some y ∧
      0<x ∧ |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|≤1/5000000+1/2^17+1/2^36 := by
  by_cases hone : (1+e.val)*2^23+f.val=0x3f800000
  · refine ⟨1,0,?_,?_,by norm_num,?_⟩
    · rw [hone];decide +kernel
    · rw [hone];decide +kernel
    · norm_num
  · obtain ⟨y,hy,hyerr⟩ := Log2Encoding.decoded_error (fixedValue e f) (fixed_range e f)
    refine ⟨Log2Input.input e f,y,Log2Input.decoded e f,?_,Log2Input.positive e f,?_⟩
    · rwa [model_normal e f hone]
    have he := Log2Table.table_error f.val f.isLt
    rw [Log2Input.ideal]
    have hid : (fixedValue e f:ℝ)/2^36=
        (e.val:ℝ)-126+(Mufu.interpolate MufuTables.lg2 f.val:ℝ)/2^36 := by
      dsimp [fixedValue]
      push_cast
      ring
    rw [hid] at hyerr
    have ht := abs_sub_le (y:ℝ)
      ((e.val:ℝ)-126+(Mufu.interpolate MufuTables.lg2 f.val:ℝ)/2^36)
      ((e.val:ℝ)-126+Real.log (1+(f.val:ℝ)/2^23)/Real.log 2)
    have he' : |((e.val:ℝ)-126+(Mufu.interpolate MufuTables.lg2 f.val:ℝ)/2^36)-
        ((e.val:ℝ)-126+Real.log (1+(f.val:ℝ)/2^23)/Real.log 2)|≤1/5000000 := by
      convert he using 1; ring
    linarith

theorem raw_positive_normal_error (w : Nat) (hw : 2^23≤w ∧ w<255*2^23) :
    ∃ x y : Rat, Root32FloatLib.value w=some x ∧ Root32FloatLib.value (Mufu.lg2 w)=some y ∧
      0<x ∧ |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|≤1/125000 := by
  let e : Fin 254 := ⟨w/2^23-1,by omega⟩
  let f : Fin (2^23) := ⟨w%2^23,by omega⟩
  have hword : (1+e.val)*2^23+f.val=w := by dsimp [e,f];omega
  obtain ⟨x,y,hx,hy,hpos,herr⟩ := positive_normal_error e f
  rw [hword] at hx hy
  refine ⟨x,y,hx,hy,hpos,?_⟩
  linarith

#print axioms model_normal
#print axioms fixed_range
#print axioms positive_normal_error
#print axioms raw_positive_normal_error
end GB300Math.Log2Uniform
