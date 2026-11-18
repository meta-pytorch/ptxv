/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementInputResolution
import MemorySafetyExecution.Native.PlacementEvents

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option Elab.async false
namespace MemorySafetyExecution.Native

theorem placements_right_separated (left right : CoreProgram) (placements : ProgramPlacements left right) :
    SeparatedLayouts right.layouts := by
  intro firstIndex secondIndex first second firstAt secondAt different
  obtain ⟨firstInside,firstEq⟩ := List.getElem?_eq_some_iff.mp firstAt
  obtain ⟨secondInside,secondEq⟩ := List.getElem?_eq_some_iff.mp secondAt
  have pairwise:=List.pairwise_iff_getElem.mp placements.2.2
  rcases Nat.lt_or_gt_of_ne different with before|after
  · simpa only [firstEq,secondEq] using pairwise firstIndex secondIndex firstInside secondInside before
  · have separated:=pairwise secondIndex firstIndex secondInside firstInside after
    simp only [firstEq,secondEq] at separated
    exact separated.symm

end MemorySafetyExecution.Native
