/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.MufuSm103

namespace GB300Math.MufuModelTotality

theorem tanh_total (x : Nat) : ∃ y, Mufu.tanh x = some y := by
  unfold Mufu.tanh
  split <;> try { exact ⟨_, rfl⟩ }
  split <;> try { exact ⟨_, rfl⟩ }
  split <;> exact ⟨_, rfl⟩

theorem canonical_total (operation : String) (x : Nat)
    (h : operation ∈ MufuSm103.operations) :
    ∃ y, Mufu.evaluate32 operation x = some y := by
  simp [MufuSm103.operations] at h
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals first
    | exact ⟨_, rfl⟩
    | exact tanh_total _

theorem fp64_high_total (rsq : Bool) (x : Nat) :
    ∃ y, Fp64Seed.highWord rsq x = y := ⟨_, rfl⟩

#print axioms tanh_total
#print axioms canonical_total
#print axioms fp64_high_total

end GB300Math.MufuModelTotality
