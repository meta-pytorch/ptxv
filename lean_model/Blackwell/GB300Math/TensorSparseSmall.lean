/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorSmallBlackwell

namespace GB300Math.TensorSparseSmall

def selectedIndex (j code : Nat) : Option Nat :=
  if code = 4 ∨ code = 6 ∨ code = 8 ∨ code = 9 ∨ code = 12 ∨ code = 13 ∨ code = 14 then
    some (j / 2 * 4 + code / 2^(2*(j%2)) % 4)
  else none

def dot (afmt bfmt : String) (a b : Array Nat) (c code : Nat) : Option Nat := do
  if a.size != 32 || b.size != 64 then none else do
    let indices ← (List.range 32).mapM (selectedIndex · code)
    TensorSmallBlackwell.dot afmt bfmt a (indices.toArray.map fun i => b[i]!) c

theorem selected_index_bounded (j code index : Nat) (hj : j < 32)
    (h : selectedIndex j code = some index) : index < 64 := by
  unfold selectedIndex at h
  split at h <;> simp_all
  omega

theorem reverse_pair : selectedIndex 0 6 = some 2 ∧ selectedIndex 1 6 = some 1 := by decide

#print axioms selected_index_bounded
#print axioms reverse_pair

end GB300Math.TensorSparseSmall
