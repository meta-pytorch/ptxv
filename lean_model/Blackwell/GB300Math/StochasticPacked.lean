/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.StochasticValue

namespace GB300Math.StochasticPacked

open StochasticSm103 StochasticValue

def laneWidth (f : Format) : Nat := if isPair f then 16 else if f == .e2m1 then 4 else 8

theorem lane_bound (f : Format) (relu satfinite : Bool) (x r : Nat) :
    lane f relu satfinite x r < 2^(laneWidth f) := by
  have hx : x%4294967296 < 4294967296 := Nat.mod_lt x (by decide)
  cases f <;> cases relu <;> cases satfinite <;>
    norm_num [laneWidth, lane, parameters, isPair] <;>
    split_ifs <;> omega



def laneCount (f : Format) : Nat := if isPair f then 2 else 4

def inputLane (a b c d i : Nat) : Nat :=
  if i = 0 then a else if i = 1 then b else if i = 2 then c else d

theorem packed_lane (f : Format) (relu satfinite : Bool) (a b c d r i : Nat)
    (hi : i < laneCount f) :
    (packed f relu satfinite a b c d r / 2^(laneWidth f*i)) % 2^(laneWidth f) =
      lane f relu satfinite (inputLane a b c d i) (entropyLane f r i) := by
  have ha := lane_bound f relu satfinite a (entropyLane f r 0)
  have hb := lane_bound f relu satfinite b (entropyLane f r 1)
  have hc := lane_bound f relu satfinite c (entropyLane f r 2)
  have hd := lane_bound f relu satfinite d (entropyLane f r 3)
  cases f <;> norm_num [laneCount, isPair] at hi <;> interval_cases i <;>
    norm_num [packed, laneWidth, isPair, inputLane] at * <;> omega

theorem packed_bound (f : Format) (relu satfinite : Bool) (a b c d r : Nat) :
    packed f relu satfinite a b c d r < 2^(laneWidth f*laneCount f) := by
  have ha := lane_bound f relu satfinite a (entropyLane f r 0)
  have hb := lane_bound f relu satfinite b (entropyLane f r 1)
  have hc := lane_bound f relu satfinite c (entropyLane f r 2)
  have hd := lane_bound f relu satfinite d (entropyLane f r 3)
  cases f <;> norm_num [packed, laneWidth, laneCount, isPair] at * <;> omega

theorem semantic_lane_width (f : Format) (relu satfinite : Bool) (x r : Nat) :
    lane f relu satfinite x r < 2^((parameters f).1+(parameters f).2.1+1) := by
  have hx : x%4294967296 < 4294967296 := Nat.mod_lt x (by decide)
  cases f <;> cases relu <;> cases satfinite <;>
    norm_num [lane, parameters, isPair] <;> split_ifs <;> omega

theorem fp6_padding (f : Format) (hf : f = .e2m3 ∨ f = .e3m2)
    (relu satfinite : Bool) (a b c d r i : Nat) (hi : i < 4) :
    (packed f relu satfinite a b c d r / 2^(8*i)) % 256 < 64 := by
  rcases hf with rfl | rfl
  · have h := packed_lane .e2m3 relu satfinite a b c d r i (by simpa [laneCount, isPair] using hi)
    norm_num [laneWidth, isPair] at h
    rw [h]
    exact semantic_lane_width .e2m3 relu satfinite _ _
  · have h := packed_lane .e3m2 relu satfinite a b c d r i (by simpa [laneCount, isPair] using hi)
    norm_num [laneWidth, isPair] at h
    rw [h]
    exact semantic_lane_width .e3m2 relu satfinite _ _

#print axioms semantic_lane_width
#print axioms fp6_padding

#print axioms packed_bound
#print axioms lane_bound
#print axioms packed_lane


theorem packed_unit_error (f : Format) (relu satfinite : Bool) (a b c d r i t : Nat)
    (hi : i < laneCount f) (ht : t < 8388608)
    (hx : inputLane a b c d i = 0x3f800000+t) :
    finiteValue f
        ((packed f relu satfinite a b c d r / 2^(laneWidth f*i)) % 2^(laneWidth f)) =
      some (unitOutput f t (entropyLane f r i)) ∧
    |unitOutput f t (entropyLane f r i) - (1+(t:Rat)/8388608)| <
      1/(2:Rat)^(fractionBits f) := by
  rw [packed_lane f relu satfinite a b c d r i hi, hx]
  exact unit_conversion_error f relu satfinite t (entropyLane f r i) ht

#print axioms packed_unit_error

end GB300Math.StochasticPacked
