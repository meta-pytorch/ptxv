/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Code.Funs
import Frontend.LaunchInventory
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option backward.split false

namespace PtxFrontendProgram

def dimensionsModel (shape : Array U32 3#usize) : PtxFrontend.Dimensions :=
  ⟨(shape.val.getD 0 0#u32).val, (shape.val.getD 1 0#u32).val,
    (shape.val.getD 2 0#u32).val⟩

theorem dimensions_three (shape : Array U32 3#usize) :
    ∃ x y z, shape.val = [x, y, z] := by
  have length : shape.val.length = 3 := by simp
  generalize listed : shape.val = values at length
  cases values with
  | nil => simp at length
  | cons x values =>
    cases values with
    | nil => simp at length
    | cons y values =>
      cases values with
      | nil => simp at length
      | cons z values =>
        cases values with
        | nil => exact ⟨x, y, z, rfl⟩
        | cons _ _ => simp at length

theorem source_dimensions_product (shape : Array U32 3#usize) (count : U64)
    (computed : strict_memory_frontend_launch.dimensions_product shape = .ok (some count)) :
    count.val = (dimensionsModel shape).size ∧ (dimensionsModel shape).positive := by
  obtain ⟨x, y, z, listed⟩ := dimensions_three shape
  have readX : Array.index_usize shape 0#usize = .ok x := by simp [Array.index_usize, listed]
  have readY : Array.index_usize shape 1#usize = .ok y := by simp [Array.index_usize, listed]
  have readZ : Array.index_usize shape 2#usize = .ok z := by simp [Array.index_usize, listed]
  unfold strict_memory_frontend_launch.dimensions_product at computed
  simp only [readX, readY, readZ, bind_ok, lift] at computed
  by_cases xZero : x = 0#u32
  · simp only [xZero, ite_true] at computed
    simp at computed
  · simp only [xZero, ite_false] at computed
    by_cases yZero : y = 0#u32
    · simp only [yZero, ite_true] at computed
      simp at computed
    · simp only [yZero, ite_false] at computed
      by_cases zZero : z = 0#u32
      · simp only [zZero, ite_true] at computed
        simp at computed
      · simp only [zZero, ite_false] at computed
        have firstLaw := U64.checked_mul_bv_spec (UScalar.cast .U64 x) (UScalar.cast .U64 y)
        cases firstComputed : U64.checked_mul (UScalar.cast .U64 x) (UScalar.cast .U64 y) with
        | none => simp only [firstComputed] at computed; simp at computed
        | some first =>
          simp only [firstComputed, ok.injEq] at computed
          rw [firstComputed] at firstLaw
          have lastLaw := U64.checked_mul_bv_spec first (UScalar.cast .U64 z)
          rw [computed] at lastLaw
          refine ⟨?_, ?_⟩
          · simp only [dimensionsModel, listed, List.getD_cons_zero, List.getD_cons_succ,
              PtxFrontend.Dimensions.size]
            rw [lastLaw.2.1, firstLaw.2.1]
            simp
          · simp only [dimensionsModel, listed, List.getD_cons_zero, List.getD_cons_succ,
              PtxFrontend.Dimensions.positive]
            scalar_tac

end PtxFrontendProgram
