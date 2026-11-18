/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionEvaluator.Model

open Aeneas Aeneas.Std Result
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction.SegEvaluator

theorem actual_modifier (bits : U32) (negative absolute : Bool) :
    PtxvSim.seg.fmod bits negative absolute = .ok (modified bits negative absolute) := by
  cases negative <;> cases absolute <;> simp [PtxvSim.seg.fmod, modified]

theorem actual_evaluator_exact_callbacks {E : Type}
    (callbacks : PtxvSim.seg.ComparisonEvaluator E) (operation : IR) (initial : E) :
    PtxvSim.seg.comparison_instruction callbacks operation initial = expected callbacks operation initial := by
  cases operation <;> try (solve | rfl)
  case Fsetp destination comparison a b negative absolute condition boolean ftz =>
    obtain ⟨n0, n1, hn⟩ := (List.length_eq_two (l := negative.val)).mp (by simp)
    obtain ⟨a0, a1, ha⟩ := (List.length_eq_two (l := absolute.val)).mp (by simp)
    simp [PtxvSim.seg.comparison_instruction, expected, Array.index_usize, hn, ha,
      actual_modifier, flags]
    apply congrArg (fun k => callbacks.read initial a >>= k)
    funext ⟨left, afterLeft⟩
    apply congrArg (fun k => callbacks.read afterLeft b >>= k)
    funext ⟨right, afterRight⟩
    apply congrArg (fun k => callbacks.compare afterRight
      ⟨comparison, modified left n0 a0, modified right n1 a1, ftz⟩ >>= k)
    funext ⟨answer, afterComparison⟩
    apply congrArg (fun k => callbacks.predicate afterComparison condition >>= k)
    funext ⟨predicate, final⟩
    cases boolean <;> cases answer <;> cases predicate <;>
      simp [combine, UScalar.cast_fromBool, lift]
    all_goals rfl
  case Sel destination a b predicate =>
    simp only [PtxvSim.seg.comparison_instruction, expected]
    congr 1
    funext result
    rcases result with ⟨selected, after⟩
    cases selected <;> rfl

end Instruction.SegEvaluator
