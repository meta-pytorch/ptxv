/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured
import GB300Math.TextureFormatsCore
import Mathlib.Tactic.FinCases

set_option maxRecDepth 100000
set_option maxHeartbeats 0

namespace GB300Math.TextureHalfFloatLib
open FloatLib.Floats.Formats.BinaryInterchange

def sourceValue (word : Nat) : Option Rat :=
  (Model.ofNatBits word : Model FloatFormat.binary16).toRat?

def decodedValue (word : Nat) : Option Rat :=
  (Model.ofNatBits (TextureFormatsCore.halfToFloat word) : Model FloatFormat.binary32).toRat?

private theorem block_exact (high : Fin 256) :
    ∀ low : Fin 256, (high.val * 256 + low.val) % 32768 < 31744 →
      decodedValue (high.val * 256 + low.val) = sourceValue (high.val * 256 + low.val) := by
  fin_cases high <;> decide +kernel

theorem all_finite_values_exact :
    ∀ word : Fin 65536, word.val % 32768 < 31744 →
      decodedValue word.val = sourceValue word.val := by
  intro word finite
  have hw := word.isLt
  have hi : word.val / 256 < 256 := by omega
  have lo : word.val % 256 < 256 := Nat.mod_lt _ (by decide)
  have h := block_exact ⟨word.val / 256, hi⟩ ⟨word.val % 256, lo⟩
  have split : word.val / 256 * 256 + word.val % 256 = word.val := by omega
  simp only at h
  rw [split] at h
  exact h finite

end GB300Math.TextureHalfFloatLib
#print axioms GB300Math.TextureHalfFloatLib.all_finite_values_exact
