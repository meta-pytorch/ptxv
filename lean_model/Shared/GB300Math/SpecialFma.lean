/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.SpecialFma

open FloatLib.Floats.Formats.BinaryInterchange

def ordinary (fmt : FloatFormat) (relu sat : Bool) (a b c : Nat) : Nat :=
  let x : Model fmt := Model.ofNatBits a
  let y : Model fmt := Model.ofNatBits b
  let z : Model fmt := Model.ofNatBits c
  let result := Model.fmaWithRounding .nearestEven x y z
  let bits := Model.toNatBits result
  if Model.isNaN result then if sat then 0 else 0x7fff
  else if (relu || sat) && bits ≥ 0x8000 then 0
  else if sat then min bits (fmt.exponentBias * 2^fmt.fracWidth) else bits

def withOob (fmt : FloatFormat) (relu sat : Bool) (isOob : Nat → Bool)
    (a b c : Nat) : Nat :=
  if isOob (a % 65536) || isOob (b % 65536) || isOob (c % 65536) then 0
  else ordinary fmt relu sat a b c

theorem withOob_ordinary (fmt : FloatFormat) (relu sat : Bool) (isOob : Nat → Bool)
    (a b c : Nat) (ha : isOob (a % 65536) = false)
    (hb : isOob (b % 65536) = false) (hc : isOob (c % 65536) = false) :
    withOob fmt relu sat isOob a b c = ordinary fmt relu sat a b c := by
  simp [withOob, ha, hb, hc]

def evaluate (format : String) (relu sat : Bool) (a b c : Nat) : Option Nat := do
  if relu && sat then none else pure ()
  let fmt ← match format with
    | "f16" | "f16x2" => some FloatFormat.binary16
    | "bf16" | "bf16x2" => some FloatFormat.bfloat16
    | _ => none
  let lo := ordinary fmt relu sat (a % 65536) (b % 65536) (c % 65536)
  if format = "f16x2" || format = "bf16x2" then
    let hi := ordinary fmt relu sat (a / 65536 % 65536) (b / 65536 % 65536) (c / 65536 % 65536)
    return lo + 65536 * hi
  return lo

end GB300Math.SpecialFma
