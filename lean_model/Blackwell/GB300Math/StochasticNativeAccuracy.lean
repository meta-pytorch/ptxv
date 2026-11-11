/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.StochasticNativeContract
import GB300Math.StochasticSaturation
import GB300Math.StochasticUnderflow

namespace GB300Math.StochasticNativeAccuracy
open StochasticSm103 StochasticValue StochasticNormal StochasticNativeContract

def laneEntropy (f : Format) (c : Nat) (reversed : Bool) : Nat :=
  let r := StochasticNativeSm103.entropy f c
  if reversed then reverseBits 16 r else r

theorem normal_full_error (f : Format) (relu : Bool) (e j t c : Nat) (reversed : Bool)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hb : e+bias f = j+127)
    (hbase : j*2^(fractionBits f) ≤ maximum f)
    (hcap : t ≤ (maximum f-j*2^(fractionBits f))*2^(23-fractionBits f))
    (ht : t < 8388608) :
    ∃ input : Rat,
      rationalValue 8 23 127 0x7f7fffff (e*8388608+t) = some input ∧
      finiteValue f (StochasticNativeSm103.lane f relu true (e*8388608+t)
        (laneEntropy f c reversed)) =
        some (normalOutput f j t (laneEntropy f c reversed)) ∧
      |normalOutput f j t (laneEntropy f c reversed)-input| <
        (2:Rat)^((j:Int)-bias f)/(2:Rat)^(fractionBits f) := by
  rw [saturated_lane_agrees]
  exact StochasticFinite.finite_full_error f relu true e j t _ he he' hj hb hbase hcap ht

theorem underflow_full_error (f : Format) (relu : Bool) (e t c : Nat) (reversed : Bool)
    (he : e ≤ 127-(parameters f).2.2.1) (ht : t < 8388608) :
    ∃ input : Rat,
      rationalValue 8 23 127 0x7f7fffff (e*8388608+t) = some input ∧
      finiteValue f (StochasticNativeSm103.lane f relu true (e*8388608+t)
        (laneEntropy f c reversed)) =
        some (StochasticUnderflow.output f e t (laneEntropy f c reversed)) ∧
      |StochasticUnderflow.output f e t (laneEntropy f c reversed)-input| < StochasticUnderflow.spacing f := by
  rw [saturated_lane_agrees]
  exact StochasticUnderflow.full_error f relu true e t _ he ht

theorem overflow_saturates (f : Format) (relu : Bool) (e j t c : Nat) (reversed : Bool)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hb : e+bias f = j+127) (ht : t < 8388608)
    (hmax : maximum f ≤ j*2^(fractionBits f)+t/2^(23-fractionBits f)) :
    finiteValue f (StochasticNativeSm103.lane f relu true (e*8388608+t)
      (laneEntropy f c reversed)) = some (StochasticSaturation.maximumValue f) := by
  rw [saturated_lane_agrees]
  exact StochasticSaturation.saturated_value f relu e j t _ he he' hj hb ht hmax

#print axioms normal_full_error
#print axioms underflow_full_error
#print axioms overflow_saturates
end GB300Math.StochasticNativeAccuracy
