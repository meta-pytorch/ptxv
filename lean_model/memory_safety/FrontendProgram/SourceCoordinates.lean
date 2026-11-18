/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceDimensions
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

theorem source_coordinates (index : U64) (shape point : Array U32 3#usize)
    (computed : strict_memory_frontend_launch.coordinates index shape = .ok (some point)) :
    dimensionsModel point = PtxFrontend.coordinates (dimensionsModel shape) index.val ∧
      index.val < (dimensionsModel shape).size := by
  unfold strict_memory_frontend_launch.coordinates at computed
  obtain ⟨product, productComputed, computed⟩ := source_bind_success _ _ _ computed
  cases product with
  | none => simp at computed
  | some count =>
    dsimp only at computed
    obtain ⟨countEq, positive⟩ := source_dimensions_product shape count productComputed
    by_cases inside : index ≥ count
    · simp only [inside, ite_true] at computed
      simp at computed
    · simp only [inside, ite_false] at computed
      obtain ⟨xSize, xRead, computed⟩ := source_bind_success _ _ _ computed
      try simp only [lift, bind_ok] at computed
      obtain ⟨x, xComputed, computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨remainder, remainderComputed, computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨ySize, yRead, computed⟩ := source_bind_success _ _ _ computed
      try simp only [lift, bind_ok] at computed
      obtain ⟨y, yComputed, computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨z, zComputed, computed⟩ := source_bind_success _ _ _ computed
      simp only [lift, bind_ok, ok.injEq, Option.some.injEq] at computed
      have pointEq := computed
      obtain ⟨sx, sy, sz, shapeEq⟩ := dimensions_three shape
      have xLookup : Array.index_usize shape 0#usize = .ok sx := by simp [Array.index_usize, shapeEq]
      have yLookup : Array.index_usize shape 1#usize = .ok sy := by simp [Array.index_usize, shapeEq]
      have xEq : xSize = sx := by simpa only [xLookup, ok.injEq] using xRead.symm
      have yEq : ySize = sy := by simpa only [yLookup, ok.injEq] using yRead.symm
      subst xSize
      subst ySize
      have nonzeroX : (UScalar.cast .U64 sx).val ≠ 0 := by
        simp only [dimensionsModel, shapeEq, List.getD_cons_zero, List.getD_cons_succ,
          PtxFrontend.Dimensions.positive] at positive
        scalar_tac
      have nonzeroY : (UScalar.cast .U64 sy).val ≠ 0 := by
        simp only [dimensionsModel, shapeEq, List.getD_cons_zero, List.getD_cons_succ,
          PtxFrontend.Dimensions.positive] at positive
        scalar_tac
      have xLaw := U64.rem_spec index nonzeroX
      have remainderLaw := U64.div_spec index nonzeroX
      have yLaw := U64.rem_spec remainder nonzeroY
      have zLaw := U64.div_spec remainder nonzeroY
      rw [xComputed] at xLaw
      rw [remainderComputed] at remainderLaw
      rw [yComputed] at yLaw
      rw [zComputed] at zLaw
      simp only [WP.spec_ok] at xLaw remainderLaw yLaw zLaw
      have bound : index.val < (dimensionsModel shape).size := by scalar_tac
      refine ⟨?_, bound⟩
      rw [← pointEq]
      have pointBounds := PtxFrontend.coordinates_in_bounds (dimensionsModel shape) index.val positive bound
      simp only [PtxFrontend.Dimensions.contains] at pointBounds
      apply PtxFrontend.Dimensions.ext
      all_goals simp [dimensionsModel, Array.make, shapeEq, PtxFrontend.coordinates] at *
      all_goals scalar_tac

theorem source_coordinates_exists (index : U64) (shape : Array U32 3#usize) (count : U64)
    (productComputed : strict_memory_frontend_launch.dimensions_product shape = .ok (some count))
    (inside : index.val < count.val) :
    ∃ point, strict_memory_frontend_launch.coordinates index shape = .ok (some point) ∧
      dimensionsModel point = PtxFrontend.coordinates (dimensionsModel shape) index.val := by
  obtain ⟨countEq, positive⟩ := source_dimensions_product shape count productComputed
  obtain ⟨sx, sy, sz, shapeEq⟩ := dimensions_three shape
  have readX : Array.index_usize shape 0#usize = .ok sx := by simp [Array.index_usize, shapeEq]
  have readY : Array.index_usize shape 1#usize = .ok sy := by simp [Array.index_usize, shapeEq]
  have nonzeroX : (UScalar.cast .U64 sx).val ≠ 0 := by
    simp [dimensionsModel, shapeEq, PtxFrontend.Dimensions.positive] at positive
    scalar_tac
  have nonzeroY : (UScalar.cast .U64 sy).val ≠ 0 := by
    simp [dimensionsModel, shapeEq, PtxFrontend.Dimensions.positive] at positive
    scalar_tac
  have total : strict_memory_frontend_launch.coordinates index shape
      ⦃ result => ∃ point, result = some point ⦄ := by
    unfold strict_memory_frontend_launch.coordinates
    rw [productComputed]
    simp only [bind_ok]
    have test : ¬index ≥ count := by scalar_tac
    simp only [test, ite_false, readX, readY, lift, bind_ok]
    step
    step
    step
    step
    exact ⟨_, rfl⟩
  obtain ⟨result, computed, point, resultEq⟩ := MemoryWrite.spec_returns _ _ total
  subst result
  exact ⟨point, computed, (source_coordinates index shape point computed).1⟩

end PtxFrontendProgram
