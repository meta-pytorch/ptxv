/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2ScaledEncoding
import Log2RelativeArithmetic
import Log2RelativeTable
import Log2Uniform
namespace GB300Math.Log2Relative
open Log2RelativeArithmetic
set_option maxHeartbeats 5000000
set_option maxRecDepth 12000

theorem fixed_real (e : Fin 254) (f : Fin (2^23)) :
    (Log2Uniform.fixedValue e f:ℝ)/2^36=
      (e.val:ℝ)-126+(Mufu.interpolate MufuTables.lg2 f.val:ℝ)/2^36 := by
  dsimp [Log2Uniform.fixedValue]
  push_cast
  ring

theorem outside_model_error (e : Fin 254) (f : Fin (2^23))
    (ho : e.val≤ 124 ∨ 127≤ e.val) :
    ∃ y : Rat,Root32FloatLib.value (Mufu.lg2 ((1+e.val)*2^23+f.val))=some y ∧
      |(y:ℝ)-((e.val:ℝ)-126+logMantissa f)|≤|(e.val:ℝ)-126+logMantissa f|/2^22 := by
  have he:=e.isLt
  have hf:=f.isLt
  have hne : (1+e.val)*2^23+f.val≠0x3f800000 := by omega
  have ht:=Log2Interpolation.interpolation_facts f.val f.isLt
  dsimp only at ht
  have hl:=log_range f
  have hz:=Log2OneSidedTable.table_interval f.val f.isLt
  let z : ℝ := (Mufu.interpolate MufuTables.lg2 f.val:ℝ)/2^36
  change -58/1000000000≤ z-logMantissa f ∧ z-logMantissa f≤ 177/1000000000 at hz
  by_cases hp : 127≤ e.val
  · let scale := e.val-126
    have hs : 1≤ scale ∧ scale≤ 127 := by dsimp [scale];omega
    have hsr : (scale:ℝ)=(e.val:ℝ)-126 := by
      dsimp [scale]
      rw [Nat.cast_sub (by omega)]
      norm_num
    have hv : 0< Log2Uniform.fixedValue e f ∧ Log2Uniform.fixedValue e f<(scale+1:Nat)*2^36 := by
      dsimp [Log2Uniform.fixedValue,scale]
      omega
    obtain ⟨y,hy,hlo,hhi⟩:=Log2ScaledEncoding.positive (Log2Uniform.fixedValue e f) scale hs hv
    refine ⟨y,?_,?_⟩
    · rwa [Log2Uniform.model_normal e f hne]
    rw [fixed_real,hsr] at hlo
    rw [fixed_real] at hhi
    exact Log2RelativeArithmetic.positive ((e.val:ℝ)-126) (logMantissa f) z y
      (by
        have h : (127:ℝ)≤ e.val := by exact_mod_cast hp
        linarith) hl hz ⟨hlo,hhi⟩
  · have hen : e.val≤ 124 := by omega
    let scale := 125-e.val
    have hs : 1≤ scale ∧ scale≤ 127 := by dsimp [scale];omega
    have hsr : (scale:ℝ)=125-(e.val:ℝ) := by
      dsimp [scale]
      rw [Nat.cast_sub (by omega)]
      norm_num
    have hv : Log2Uniform.fixedValue e f< 0 ∧ -Log2Uniform.fixedValue e f≤(scale+1:Nat)*2^36 := by
      dsimp [Log2Uniform.fixedValue,scale]
      omega
    obtain ⟨y,hy,hlo,hhi⟩:=Log2ScaledEncoding.negative (Log2Uniform.fixedValue e f) scale hs hv
    refine ⟨y,?_,?_⟩
    · rwa [Log2Uniform.model_normal e f hne]
    rw [fixed_real] at hlo hhi
    rw [hsr] at hhi
    by_cases heq : e.val=124
    · have hrel:=Log2RelativeTable.table_upper f.val f.isLt
      have hbound:=Log2RelativeArithmetic.negative_two (logMantissa f) z y hl hz.1 hrel
        (by
          dsimp [z] at *
          norm_num [heq] at hlo hhi
          constructor <;> linarith)
      norm_num [heq] at ⊢
      simpa only [show (2:ℝ)^22=4194304 by norm_num] using hbound
    · have hm : (3:ℝ)≤ 126-(e.val:ℝ) := by
        have h : (e.val:ℝ)≤ 123 := by exact_mod_cast (show e.val≤ 123 by omega)
        linarith
      have hy' : 0≤(y:ℝ)-(-(126-(e.val:ℝ))+z) ∧
          (y:ℝ)-(-(126-(e.val:ℝ))+z)≤((126-(e.val:ℝ))-1)/2^23+1/2^36 := by
        dsimp [z]
        constructor <;> linarith [hlo,hhi]
      have hbound:=Log2RelativeArithmetic.negative_large (126-(e.val:ℝ)) (logMantissa f) z y hm hl hz hy'
      convert hbound using 1 <;> congr 2 <;> ring

theorem decoded_relative (e : Fin 254) (f : Fin (2^23))
    (ho : e.val≤ 124 ∨ 127≤ e.val) :
    ∃ x y : Rat,Root32FloatLib.value ((1+e.val)*2^23+f.val)=some x ∧
      Root32FloatLib.value (Mufu.lg2 ((1+e.val)*2^23+f.val))=some y ∧
      0< x ∧ |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|/|Real.log (x:ℝ)/Real.log 2|≤ 1/2^22 := by
  obtain ⟨y,hy,herr⟩:=outside_model_error e f ho
  refine ⟨Log2Input.input e f,y,Log2Input.decoded e f,hy,Log2Input.positive e f,?_⟩
  rw [Log2Input.ideal]
  change |(y:ℝ)-((e.val:ℝ)-126+logMantissa f)|/|(e.val:ℝ)-126+logMantissa f|≤ 1/2^22
  have hl:=log_range f
  have hn : (e.val:ℝ)-126+logMantissa f≠0 := by
    rcases ho with ho|ho
    · have h : (e.val:ℝ)≤ 124 := by exact_mod_cast ho
      linarith [hl.2]
    · have h : (127:ℝ)≤ e.val := by exact_mod_cast ho
      linarith [hl.1]
  apply (div_le_iff₀ (abs_pos.mpr hn)).mpr
  simpa only [div_eq_mul_inv,mul_comm,one_mul] using herr

#print axioms fixed_real
#print axioms outside_model_error
#print axioms decoded_relative
end GB300Math.Log2Relative
