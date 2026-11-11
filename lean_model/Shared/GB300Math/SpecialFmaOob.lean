/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.SpecialFma

namespace GB300Math.SpecialFmaOob
open FloatLib.Floats.Formats.BinaryInterchange

def sentinel (word : Nat) : Bool := word % 32768 = 0x7ff7

def lane (fmt : FloatFormat) (relu sat : Bool) (a b c : Nat) : Nat :=
  if sentinel a || sentinel b then 0
  else SpecialFma.ordinary fmt relu sat a b c

def evaluate (format : String) (relu sat : Bool) (a b c : Nat) : Option Nat := do
  if relu && sat then none else pure ()
  let fmt ← match format with
    | "f16" | "f16x2" => some FloatFormat.binary16
    | "bf16" | "bf16x2" => if sat then none else some FloatFormat.bfloat16
    | _ => none
  let lo := lane fmt relu sat (a % 65536) (b % 65536) (c % 65536)
  if format = "f16x2" || format = "bf16x2" then
    return lo + 65536 * lane fmt relu sat (a / 65536 % 65536)
      (b / 65536 % 65536) (c / 65536 % 65536)
  return lo

theorem negative_sentinel : sentinel 0xfff7 = true := by decide
theorem adjacent_nan_not_sentinel : sentinel 0x7ff6 = false := by decide
theorem a_forces_zero (fmt : FloatFormat) (relu sat : Bool) (b c : Nat) :
    lane fmt relu sat 0x7ff7 b c = 0 := by simp [lane, sentinel]
theorem b_forces_zero (fmt : FloatFormat) (relu sat : Bool) (a c : Nat) :
    lane fmt relu sat a 0xfff7 c = 0 := by simp [lane, sentinel]
set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem addend_not_trigger : lane .binary16 false false 0x3c00 0x3c00 0x7ff7 = 0x7fff := by decide

end GB300Math.SpecialFmaOob

#print axioms GB300Math.SpecialFmaOob.negative_sentinel
#print axioms GB300Math.SpecialFmaOob.adjacent_nan_not_sentinel
#print axioms GB300Math.SpecialFmaOob.a_forces_zero
#print axioms GB300Math.SpecialFmaOob.b_forces_zero
#print axioms GB300Math.SpecialFmaOob.addend_not_trigger
