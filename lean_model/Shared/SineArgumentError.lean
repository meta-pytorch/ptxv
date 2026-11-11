/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import SineArgument
import SinePhase
namespace GB300Math.SineArgumentError
set_option maxHeartbeats 2000000

theorem reduced_error (x r : ℝ) (n q : Nat)
    (hx : x=(n:ℝ)+(q:ℝ)/2^25+r) (hr : 0≤r ∧ r≤1/2^25) :
    |Real.sin (2*Real.pi*x)-Real.sin (Real.pi*q/2^24)| ≤ Real.pi/2^24 ∧
    |Real.cos (2*Real.pi*x)-Real.cos (Real.pi*q/2^24)| ≤ Real.pi/2^24 := by
  have hangle : 2*Real.pi*x = (Real.pi*q/2^24+2*Real.pi*r)+(n:ℝ)*(2*Real.pi) := by rw [hx];ring
  rw [hangle,Real.sin_add_nat_mul_two_pi,Real.cos_add_nat_mul_two_pi]
  have hs := Real.abs_sin_sub_sin_le (Real.pi*q/2^24+2*Real.pi*r) (Real.pi*q/2^24)
  have hc := Real.abs_cos_sub_cos_le (Real.pi*q/2^24+2*Real.pi*r) (Real.pi*q/2^24)
  have hnonneg : 0 ≤ 2*Real.pi*r := mul_nonneg (by positivity) hr.1
  have he : Real.pi*q/2^24+2*Real.pi*r-Real.pi*q/2^24=2*Real.pi*r := by ring
  rw [he,abs_of_nonneg hnonneg] at hs hc
  constructor <;> nlinarith [Real.pi_pos,hr.2]

theorem cosine_phase (q : Nat) (_hq : q<2^25) :
    Real.sin (Real.pi*((q+2^23)%2^25:Nat)/2^24)=Real.cos (Real.pi*q/2^24) := by
  have hid := Nat.mod_add_div (q+2^23) (2^25)
  have hidr : (((q+2^23)%2^25:Nat):ℝ)+(2^25:ℝ)*(((q+2^23)/2^25:Nat):ℝ)=(q:ℝ)+2^23 := by exact_mod_cast hid
  have he : Real.pi*(((q+2^23)%2^25:Nat):ℝ)/2^24 =
      (Real.pi*q/2^24+Real.pi/2)-(((q+2^23)/2^25:Nat):ℝ)*(2*Real.pi) := by
    have hp := congrArg (fun t:ℝ => Real.pi*t) hidr
    nlinarith only [hp]
  rw [he,Real.sin_sub_nat_mul_two_pi,Real.sin_add_pi_div_two]
#print axioms reduced_error
#print axioms cosine_phase
end GB300Math.SineArgumentError
