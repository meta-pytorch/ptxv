/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TensorTF32TransposeLayout

set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

def swizzle (x : BitVec 12) : BitVec 12 :=
  x ^^^ ((x >>> 2) &&& 96)

theorem swizzle_involution : ∀ x : BitVec 12, swizzle (swizzle x) = x := by
  decide +kernel

theorem swizzle_preserves_atom_offset : ∀ x : BitVec 12,
    swizzle x &&& 31 = x &&& 31 := by
  decide +kernel

def unswizzled (rows row k : Nat) : Nat :=
  row % 32 * 4 + k % 4 * 128 + row / 32 * 512 + k / 4 * rows * 16

theorem matrixA_roundtrip (row k : Nat) (hr : row < 128) :
    unswizzled 128 row k / 512 % 4 * 32 + unswizzled 128 row k / 4 % 32 = row ∧
    unswizzled 128 row k / 2048 * 4 + unswizzled 128 row k / 128 % 4 = k := by
  unfold unswizzled
  omega

theorem matrixB_roundtrip (row k : Nat) (hr : row < 32) :
    unswizzled 32 row k / 4 % 32 = row ∧
    unswizzled 32 row k / 512 * 4 + unswizzled 32 row k / 128 % 4 = k := by
  unfold unswizzled
  omega

theorem matrixA_bounds (row k : Nat) (hr : row < 128) (hk : k < 8) :
    unswizzled 128 row k + 4 ≤ 4096 ∧ unswizzled 128 row k % 4 = 0 := by
  unfold unswizzled
  omega

theorem matrixB_bounds (row k : Nat) (hr : row < 32) (hk : k < 8) :
    unswizzled 32 row k + 4 ≤ 1024 ∧ unswizzled 32 row k % 4 = 0 := by
  unfold unswizzled
  omega

#print axioms swizzle_involution
#print axioms swizzle_preserves_atom_offset
#print axioms matrixA_roundtrip
#print axioms matrixB_roundtrip
#print axioms matrixA_bounds
#print axioms matrixB_bounds

end GB300Math.TensorTF32TransposeLayout
