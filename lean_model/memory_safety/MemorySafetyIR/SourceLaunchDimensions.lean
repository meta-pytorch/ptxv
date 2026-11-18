/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.LaunchModel
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source

set_option autoImplicit false
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

theorem source_dimensions_loop (dimensions : Array U32 3#usize)
    (product : Option Usize) (index count : Usize)
    (computed : strict_memory_launch.dimensions_product_loop dimensions product index = .ok (some count)) :
    ∃ start, product = some start ∧
      count.val = start.val * ((dimensions.val.drop index.val).map (fun value => value.val)).prod ∧
      ∀ dimension ∈ dimensions.val.drop index.val, dimension.val ≠ 0 := by
  generalize fuelEq : 3 - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing product index with
  | h fuel ih =>
    have unroll : strict_memory_launch.dimensions_product_loop dimensions product index = (do
        let result ← strict_memory_launch.dimensions_product_loop.body dimensions product index
        match result with
        | .cont pair => strict_memory_launch.dimensions_product_loop dimensions pair.1 pair.2
        | .done result => .ok result) := by
      unfold strict_memory_launch.dimensions_product_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done result => rfl
    rw [unroll] at computed
    by_cases inside : index.val < 3
    · have test : index < 3#usize := by scalar_tac
      have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by scalar_tac
      let after : Usize := Usize.ofNatCore (index.val + 1) bound
      have afterVal : after.val = index.val + 1 := by simp [after]
      have increment := source_usize_increment index bound
      have arrayInside : index.val < dimensions.val.length := by simpa using inside
      let dimension := dimensions.val[index.val]
      have read : Array.index_usize dimensions index = .ok dimension := by
        simp [Array.index_usize, List.getElem?_eq_getElem arrayInside, dimension]
      simp only [strict_memory_launch.dimensions_product_loop.body, test, ite_true] at computed
      cases product with
      | none =>
        simp only [bind_ok, increment] at computed
        obtain ⟨start, impossible, _⟩ := ih (3 - after.val) (by omega) none after computed rfl
        cases impossible
      | some start =>
        simp only [read, bind_ok] at computed
        by_cases nonzero : dimension != 0#u32
        · simp only [nonzero, ite_true, lift, bind_ok, increment] at computed
          have castVal : (UScalar.cast .Usize dimension).val = dimension.val := by
            simp
          have checked := Usize.checked_mul_bv_spec start (UScalar.cast .Usize dimension)
          cases multiplied : Usize.checked_mul start (UScalar.cast .Usize dimension) with
          | none =>
            rw [multiplied] at computed
            obtain ⟨previous, impossible, _⟩ := ih (3 - after.val) (by omega) none after computed rfl
            cases impossible
          | some next =>
            rw [multiplied] at computed
            simp only [multiplied] at checked
            obtain ⟨previous, same, exactCount, nonzeroTail⟩ :=
              ih (3 - after.val) (by omega) (some next) after computed rfl
            cases same
            refine ⟨start, rfl, ?_, ?_⟩
            · rw [exactCount, checked.2.1, castVal, List.drop_eq_getElem_cons arrayInside]
              simp only [List.map_cons, List.prod_cons, afterVal, Nat.mul_assoc]
              rfl
            · rw [List.drop_eq_getElem_cons arrayInside]
              intro selected member
              simp only [List.mem_cons] at member
              rcases member with same | member
              · subst selected
                scalar_tac
              · apply nonzeroTail selected
                simpa only [afterVal] using member
        · simp only [nonzero, Bool.false_eq_true, ite_false, bind_ok, increment] at computed
          obtain ⟨previous, impossible, _⟩ := ih (3 - after.val) (by omega) none after computed rfl
          cases impossible
    · have test : ¬index < 3#usize := by scalar_tac
      simp only [strict_memory_launch.dimensions_product_loop.body, test, ite_false, bind_ok,
        ok.injEq] at computed
      refine ⟨count, computed, ?_, ?_⟩
      · simp [List.drop_eq_nil_of_le (show dimensions.val.length ≤ index.val by simpa using Nat.le_of_not_gt inside)]
      · simp [List.drop_eq_nil_of_le (show dimensions.val.length ≤ index.val by simpa using Nat.le_of_not_gt inside)]

theorem source_dimensions_product (dimensions : Array U32 3#usize) (count : Usize)
    (computed : strict_memory_launch.dimensions_product dimensions = .ok (some count)) :
    count.val = dimensionsProduct dimensions ∧ ∀ dimension ∈ dimensions.val, dimension.val ≠ 0 := by
  obtain ⟨start, same, exactCount, nonzero⟩ := source_dimensions_loop dimensions (some 1#usize) 0#usize count computed
  cases same
  simpa [dimensionsProduct] using And.intro exactCount nonzero

end MemorySafetyIR
