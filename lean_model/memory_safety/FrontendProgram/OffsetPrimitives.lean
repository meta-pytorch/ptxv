/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Header

open Aeneas Aeneas.Std Result PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option backward.isDefEq.respectTransparency false

namespace PtxFrontendProgram

theorem offset_cast_value (value : U64) (bound : value.val<2^63) :
    (UScalar.hcast .I64 value).val=Int.ofNat value.val := by
  have meaning := UScalar.hcast_inBounds_spec .I64 value (by scalar_tac)
  simpa [lift,WP.spec_ok] using meaning

theorem offset_negate_value (value : U64) (offset : I64) (bound : value.val<2^63)
    (computed : (-. (UScalar.hcast .I64 value) : Result I64)=.ok offset) :
    offset.val= -Int.ofNat value.val := by
  have cast := offset_cast_value value bound
  have notMin : UScalar.hcast .I64 value≠IScalar.min .I64 := by scalar_tac
  have meaning := HNeg.hNeg.step (UScalar.hcast .I64 value) notMin
  rw [computed] at meaning
  simpa only [WP.spec_ok,cast] using meaning

theorem plus_not_minus (source : List U8) (tokens : List SourceToken) (position : Nat)
    (plus : TokenAt source tokens position "+") : ¬TokenAt source tokens position "-" := by
  rintro ⟨minusToken,minusFound,minusMeaning⟩
  obtain ⟨plusToken,plusFound,plusMeaning⟩ := plus
  have tokenEq : plusToken=minusToken := by
    rw [minusFound] at plusFound
    exact (Option.some.inj plusFound).symm
  subst minusToken
  have plusLiteral : utf8 "+"=[43#u8] := by decide
  have minusLiteral : utf8 "-"=[45#u8] := by decide
  simp only [TokenSpelling,plusLiteral] at plusMeaning
  simp only [TokenSpelling,minusLiteral] at minusMeaning
  have plusPoint := plusMeaning.2.2.2 0 (by decide)
  have minusPoint := minusMeaning.2.2.2 0 (by decide)
  rw [plusPoint] at minusPoint
  norm_num at minusPoint

end PtxFrontendProgram
