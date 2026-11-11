/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MufuReciprocalUniversal
import MufuSelectorSm103

namespace GB300Math.MufuReciprocalSelectorAccuracy

open MufuReciprocalBounds

def selected (modifier selector input : Nat) : Nat :=
  MufuSelectorSm103.modify 16 modifier (input / 2^(16*selector))

def result (bf16 : Bool) (modifier selector input : Nat) : Nat :=
  (MufuSelectorSm103.evaluate (if bf16 then "bf16" else "f16")
    "rcp" modifier selector input).getD 65536

theorem selected_bounded (modifier selector input : Nat) :
    selected modifier selector input < 65536 := by
  unfold selected MufuSelectorSm103.modify
  have hm := Nat.mod_lt (input / 2^(16*selector)) (by decide : 0 < 2^16)
  have hhalf := Nat.mod_lt (input / 2^(16*selector) % 2^16) (by decide : 0 < 2^15)
  split_ifs <;> dsimp only
  · exact Nat.xor_lt_two_pow (by omega) (by decide : 2^15 < 2^16)
  · omega
  · exact Nat.xor_lt_two_pow hm (by decide : 2^15 < 2^16)
  · omega

theorem result_eq (bf16 : Bool) (modifier selector input : Nat) :
    result bf16 modifier selector input =
      output bf16 (selected modifier selector input) := by
  cases bf16 <;>
    simp [result, selected, MufuSelectorSm103.evaluate, output]

theorem source_floatLib_relative (bf16 : Bool) (modifier : Fin 4) (selector : Fin 2)
    (input : BitVec 32)
    (inside : inDomain bf16 (selected modifier.val selector.val input.toNat)
      (result bf16 modifier.val selector.val input.toNat) = true) :
    ∃ x y : ℝ,
      (floatLibValue bf16 (selected modifier.val selector.val input.toNat)).map (fun q : ℚ => (q : ℝ)) = some x ∧
      (floatLibValue bf16 (result bf16 modifier.val selector.val input.toNat)).map (fun q : ℚ => (q : ℝ)) = some y ∧
      |y-1/x| / |1/x| ≤
        ((bound bf16 (result bf16 modifier.val selector.val input.toNat)).1 : ℝ) /
          (bound bf16 (result bf16 modifier.val selector.val input.toNat)).2 := by
  rw [result_eq] at inside ⊢
  exact MufuReciprocalUniversal.floatLib_relative_all bf16
    ⟨selected modifier.val selector.val input.toNat, selected_bounded _ _ _⟩ inside

#print axioms selected_bounded
#print axioms result_eq
#print axioms source_floatLib_relative

end GB300Math.MufuReciprocalSelectorAccuracy
