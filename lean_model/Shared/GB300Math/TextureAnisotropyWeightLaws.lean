/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyOriented

namespace GB300Math.TextureAnisotropyWeightLaws

open TextureAnisotropyOriented

set_option maxRecDepth 100000
set_option maxHeartbeats 0

theorem groups_independent (q : Nat) (x y : Int) :
    (taps q x y).1 = (taps q 0 0).1 := by
  unfold taps
  split <;> simp_all only
  split <;> simp_all only

theorem weights_independent (q : Nat) (x y : Int) :
    ((taps q x y).2.map Tap.weight) = ((taps q 0 0).2.map Tap.weight) := by
  by_cases h : q=0
  · simp [taps,h]
  by_cases l : q<256
  · simp [taps,h,l]
  simp [taps,h,l,List.map_map,Function.comp_def]

theorem finite_weight_conservation : ∀ q : Fin 1025,
    ((taps q.val 0 0).2.map Tap.weight).sum = (taps q.val 0 0).1*256 := by
  decide +kernel

theorem weight_conservation (q : Nat) (x y : Int) (hq : q ≤ 1024) :
    ((taps q x y).2.map Tap.weight).sum = (taps q x y).1*256 := by
  rw [weights_independent, groups_independent]
  exact finite_weight_conservation ⟨q,by omega⟩

theorem finite_tap_count : ∀ q : Fin 1025,
    (taps q.val 0 0).2.length ≤ 16 := by decide +kernel

theorem sixteen_taps_attained : (taps 1024 0 0).2.length = 16 := by decide +kernel

#print axioms groups_independent
#print axioms weights_independent
#print axioms finite_weight_conservation
#print axioms weight_conservation
#print axioms finite_tap_count
#print axioms sixteen_taps_attained

end GB300Math.TextureAnisotropyWeightLaws
