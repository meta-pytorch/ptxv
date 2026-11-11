/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import NativeIntegerModifiers

namespace GB300Math.NativeIntegerModifiers

theorem encode_bound (width : Nat) (value : Int) : encode width value < 2^width := by
  have hp : (0 : Int) < (2^width : Nat) := by exact_mod_cast Nat.two_pow_pos width
  have hn : ((2^width : Nat) : Int) ≠ 0 := by omega
  exact (Int.toNat_lt (Int.emod_nonneg value hn)).2 (Int.emod_lt_of_pos value hp)

theorem lane_bound (operation : String) (signed relu neg : Bool) (width a b : Nat) :
    lane operation signed relu neg width a b < 2^width := by
  exact encode_bound width _

theorem lower_lane_preserved (operation : String) (signed relu sat neg : Bool) (a b c selector : Nat)
    (h : operation ≠ "pack") :
    evaluate operation signed relu sat neg 16 selector a b c % 65536 =
      lane operation signed relu neg 16 a b := by
  have bound := lane_bound operation signed relu neg 16 a b
  simp only [evaluate, h, ↓reduceIte, show (16 : Nat) ≠ 32 by decide]
  omega

theorem lane_ignores_high_bits (operation : String) (signed relu neg : Bool) (a b highA highB : Nat) :
    lane operation signed relu neg 16 (a % 65536 + highA*65536) (b % 65536 + highB*65536) =
      lane operation signed relu neg 16 a b := by
  simp [lane, decode, Nat.add_mod]

theorem unsaturated_signedness_irrelevant (width word : Nat) :
    convert false false width word = convert true false width word := by rfl

theorem packing_preserves_selected_payload (signed sat : Bool) (width selector a b c : Nat)
    (hw : width ∈ [2,4,8,16]) :
    pack signed sat width selector a b c / 2^(2*width) =
      (c / (if width = 8 ∧ selector = 1 then 65536 else 1)) % 2^(32-2*width) := by
  have ha : convert signed sat width a < 2^width := encode_bound width _
  have hb : convert signed sat width b < 2^width := encode_bound width _
  simp at hw
  rcases hw with rfl | rfl | rfl | rfl <;> simp only [pack] <;> omega

#print axioms encode_bound
#print axioms lower_lane_preserved
#print axioms lane_ignores_high_bits
#print axioms unsaturated_signedness_irrelevant
#print axioms packing_preserves_selected_payload

end GB300Math.NativeIntegerModifiers
