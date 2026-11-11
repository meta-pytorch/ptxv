/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.StochasticValue

namespace GB300Math.StochasticBoundaries

open StochasticSm103 StochasticValue

def floorWord : Format → Nat
  | .f16 => 0x2d000000
  | .bf16 => 1
  | .e4m3 => 0x33000000
  | .e5m2 => 0x2f800000
  | .e2m3 => 0x36000000
  | .e3m2 => 0x35800000
  | .e2m1 => 0x37000000

theorem below_entropy_floor (f : Format) (r : Nat) :
    lane f false true (floorWord f-1) r = 0 := by
  cases f <;> norm_num [floorWord, lane, parameters, isPair, roundMantissa] <;> omega

theorem at_entropy_floor (f : Format) (r : Nat) :
    lane f false true (floorWord f) r =
      if r%2^(entropyBits f) = 2^(entropyBits f)-1 then 1 else 0 := by
  cases f <;> norm_num [floorWord, lane, parameters, isPair, roundMantissa, entropyBits,
    fractionBits] <;> split_ifs <;> omega

theorem saturated_max_f32 (f : Format) (r : Nat) :
    lane f false true 0x7f7fffff r = (parameters f).2.2.2 := by
  cases f <;> norm_num [lane, parameters, isPair, roundMantissa] <;> omega

theorem bf16_overflow_chance (r : Nat) :
    lane .bf16 false false 0x7f7fffff r = if r%65536 = 0 then 32639 else 32640 := by
  norm_num [lane, parameters, isPair, roundMantissa]
  split_ifs <;> omega

theorem negative_zero (f : Format) (satfinite : Bool) (r : Nat) :
    lane f false satfinite 0x80000000 r = 2^((parameters f).1+(parameters f).2.1) := by
  cases f <;> cases satfinite <;> norm_num [lane, parameters, isPair, roundMantissa] <;> omega

theorem rectified_negative_zero (f : Format) (satfinite : Bool) (r : Nat) :
    lane f true satfinite 0x80000000 r = 0 := by
  cases f <;> cases satfinite <;> norm_num [lane, parameters, isPair, roundMantissa]


def nanCode (f : Format) : Nat :=
  if isPair f then 32767 else if f == .e4m3 || f == .e5m2 then 127 else (parameters f).2.2.2

theorem nan_payload (f : Format) (relu satfinite : Bool) (x r : Nat)
    (hx : x < 4294967296) (he : x/8388608%256 = 255) (hf : x%8388608 ≠ 0) :
    lane f relu satfinite x r = nanCode f := by
  cases f <;> cases relu <;> cases satfinite <;>
    simp [lane, parameters, isPair, Nat.mod_eq_of_lt hx, he, hf, nanCode]

#print axioms nan_payload
#print axioms below_entropy_floor
#print axioms at_entropy_floor
#print axioms saturated_max_f32
#print axioms bf16_overflow_chance
#print axioms negative_zero
#print axioms rectified_negative_zero

end GB300Math.StochasticBoundaries
