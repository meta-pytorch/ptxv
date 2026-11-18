/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Code.Funs
import Frontend.Semantics
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
set_option backward.isDefEq.respectTransparency false
set_option backward.split false

open PtxFrontend

namespace PtxFrontendProgram

theorem source_is_space (byte : U8) : strict_memory_frontend.is_space byte = .ok (decide (Space byte)) := by
  generalize expectedEq : decide (Space byte) = expected
  unfold strict_memory_frontend.is_space
  repeat' split
  all_goals apply congrArg (fun value : Bool => (ok value : Result Bool))
  all_goals rw [←expectedEq]
  all_goals apply Bool.eq_iff_iff.mpr
  all_goals simp_all [Space]
  all_goals scalar_tac

theorem source_is_word (byte : U8) : strict_memory_frontend.is_word byte = .ok (decide (WordByte byte)) := by
  generalize expectedEq : decide (WordByte byte) = expected
  unfold strict_memory_frontend.is_word
  repeat' split
  all_goals apply congrArg (fun value : Bool => (ok value : Result Bool))
  all_goals rw [←expectedEq]
  all_goals apply Bool.eq_iff_iff.mpr
  all_goals simp_all [WordByte]
  all_goals scalar_tac

theorem source_is_punctuation (byte : U8) :
    strict_memory_frontend.is_punctuation byte = .ok (decide (Punctuation byte)) := by
  generalize expectedEq : decide (Punctuation byte) = expected
  unfold strict_memory_frontend.is_punctuation
  repeat' split
  all_goals apply congrArg (fun value : Bool => (ok value : Result Bool))
  all_goals rw [←expectedEq]
  all_goals apply Bool.eq_iff_iff.mpr
  all_goals simp_all [Punctuation]
  all_goals scalar_tac

end PtxFrontendProgram
