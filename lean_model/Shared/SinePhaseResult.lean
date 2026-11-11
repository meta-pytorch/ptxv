/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import SineTableBounds
import SinePhase
namespace GB300Math.SinePhaseResult
open FloatLib.Floats.Formats.BinaryInterchange
open SinePhase SineTableBounds
set_option maxHeartbeats 3000000

def phaseWord (q : Nat) (negative : Bool) : Nat :=
  (if negative then (if q/2^24=1 then 0 else 2^31) else (if q/2^24=1 then 2^31 else 0))+
    magnitudeWord (tableIndex q)

theorem decoded_phase (q : Nat) (hq : q<2^25) (negative : Bool) :
    ∃ y : Rat,
      (Model.ofNatBits (phaseWord q negative) : Model FloatFormat.binary32).toRat?=some y ∧
      |(y:ℝ)-(if negative then -Real.sin (Real.pi*q/2^24) else Real.sin (Real.pi*q/2^24))| ≤ 71/100000000 := by
  obtain ⟨y,hy,herr⟩ := decoded_table (tableIndex q) (index_bounds q hq).1
  have hr := reflection_error q hq
  have htri := abs_sub_le (y:ℝ) (Real.sin (Real.pi*tableIndex q/2^24)) (Real.sin (Real.pi*idealIndex q/2^24))
  rw [abs_sub_comm (Real.sin (Real.pi*tableIndex q/2^24))] at htri
  have hu : |(y:ℝ)-Real.sin (Real.pi*idealIndex q/2^24)| ≤ 71/100000000 := by
    have hp := Real.pi_lt_d4
    nlinarith only [herr,hr,htri,hp]
  have hn := RcpSigned.signed_decode (magnitudeWord (tableIndex q)) y hy
  have hn' : (Model.ofNatBits (2^31+magnitudeWord (tableIndex q)) : Model FloatFormat.binary32).toRat?=some (-y) := by simpa [Nat.add_comm] using hn
  have hf := ideal_fold q hq
  simp only [phaseSign] at hf
  cases negative <;> by_cases hs : q/2^24=1
  · simp only [phaseWord,Bool.false_eq_true,ite_false,hs,ite_true] at hf ⊢
    refine ⟨-y,hn',?_⟩
    rw [hf]
    simpa [phaseSign,hs,neg_add_eq_sub,abs_sub_comm] using hu
  · simp only [phaseWord,Bool.false_eq_true,ite_false,hs,Nat.zero_add] at hf ⊢
    refine ⟨y,hy,?_⟩
    rw [hf]
    simpa [phaseSign,hs] using hu
  · simp only [phaseWord,ite_true,hs,Nat.zero_add] at hf ⊢
    refine ⟨y,hy,?_⟩
    rw [hf]
    simpa [phaseSign,hs] using hu
  · simp only [phaseWord,ite_true,hs,ite_false] at hf ⊢
    refine ⟨-y,hn',?_⟩
    rw [hf]
    simpa [phaseSign,hs,neg_add_eq_sub,abs_sub_comm] using hu
#print axioms decoded_phase
end GB300Math.SinePhaseResult
