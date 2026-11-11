/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.MatrixTranspose

def transposeIndex (i : Nat) : Nat := (i%8)*8+i/8

def half (words : Array Nat) (index : Nat) : Nat :=
  (words[index/2]! / 2^(16*(index%2))) % 65536

def word (words : Array Nat) (lane : Nat) : Nat :=
  if lane ≥ 32 then 0xdeadbeef
  else half words (transposeIndex (2*lane)) + 65536*half words (transposeIndex (2*lane+1))

theorem transpose_involution : ∀ i : Fin 64,
    transposeIndex (transposeIndex i) = i := by decide +kernel

theorem transpose_in_bounds : ∀ i : Fin 64, transposeIndex i < 64 := by decide +kernel

theorem half_in_bounds (words : Array Nat) (i : Nat) : half words i < 65536 := by
  exact Nat.mod_lt _ (by decide)

theorem word_in_bounds (words : Array Nat) (lane : Nat) : word words lane < 2^32 := by
  unfold word
  split
  · decide
  · have h1 := half_in_bounds words (transposeIndex (2*lane))
    have h2 := half_in_bounds words (transposeIndex (2*lane+1))
    omega

#print axioms transpose_involution
#print axioms transpose_in_bounds
#print axioms half_in_bounds
#print axioms word_in_bounds
end GB300Math.MatrixTranspose
