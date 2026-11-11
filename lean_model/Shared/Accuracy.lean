/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured
import FloatLib.Numerics.Enclosure.Elementary.Proof

set_option maxRecDepth 100000
set_option maxHeartbeats 20000000

namespace GB300Math.Accuracy

open FloatLib.Numerics
open FloatLib.Floats

def decoded (word : Nat) : ℚ :=
  (ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits word : ExecFloat.Binary 8 23)).getD 0

theorem exp_output_decode : decoded 0x402df854 = 2850325 / 1048576 := by decide +kernel
theorem log_output_decode : decoded 0x3f317218 = 1453635 / 2097152 := by decide +kernel
theorem one_input_decode : decoded 0x3f800000 = 1 := by decide +kernel
theorem two_input_decode : decoded 0x40000000 = 2 := by decide +kernel

theorem error_of_enclosure {interval : RationalInterval} {actual : ℝ} {output error : ℚ}
    (hcontains : interval.Contains actual)
    (hlower : output - error ≤ interval.lo)
    (hupper : interval.hi ≤ output + error) :
    |(output : ℝ) - actual| ≤ (error : ℝ) := by
  have hlo : ((output - error : ℚ) : ℝ) ≤ (interval.lo : ℝ) := by exact_mod_cast hlower
  have hhi : (interval.hi : ℝ) ≤ ((output + error : ℚ) : ℝ) := by exact_mod_cast hupper
  push_cast at hlo hhi
  obtain ⟨hl, hu⟩ := hcontains
  exact abs_le.mpr ⟨by linarith, by linarith⟩

theorem exp_one_error :
    |(decoded 0x402df854 : ℝ) - Real.exp (decoded 0x3f800000 : ℝ)| ≤ (1 : ℝ) / 8388608 := by
  rw [one_input_decode, exp_output_decode]
  have h := error_of_enclosure (Enclosure.contains_exp 1 20)
    (output := 2850325 / 1048576) (error := 1 / 8388608)
    (by decide +kernel) (by decide +kernel)
  norm_num at h ⊢
  exact h

theorem log_two_error :
    |(decoded 0x3f317218 : ℝ) - Real.log (decoded 0x40000000 : ℝ)| ≤ (1 : ℝ) / 268435456 := by
  rw [two_input_decode, log_output_decode]
  have h := error_of_enclosure (Enclosure.contains_log 2 24 (by norm_num))
    (output := 1453635 / 2097152) (error := 1 / 268435456)
    (by decide +kernel) (by decide +kernel)
  norm_num at h ⊢
  exact h

#print axioms exp_one_error
#print axioms log_two_error

end GB300Math.Accuracy

open Lean

def main : IO Unit := do
  let cases := [
    Lean.Json.mkObj [("operation", "f32_exp"), ("input", toJson (0x3f800000 : Nat)),
      ("output", toJson (0x402df854 : Nat)), ("theorem", "GB300Math.Accuracy.exp_one_error"),
      ("absolute_error", "1/8388608"), ("ulp_error", "1/2")],
    Lean.Json.mkObj [("operation", "f32_log"), ("input", toJson (0x40000000 : Nat)),
      ("output", toJson (0x3f317218 : Nat)), ("theorem", "GB300Math.Accuracy.log_two_error"),
      ("absolute_error", "1/268435456"), ("ulp_error", "1/16")]]
  IO.println s!"GB300_ACCURACY {(toJson cases).compress}"
