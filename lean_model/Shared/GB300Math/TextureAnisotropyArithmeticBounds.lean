/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyArithmetic

namespace GB300Math.TextureAnisotropyArithmeticBounds

theorem floor_error (a d : Int) (hd : 0<d) :
    a/d*d ≤ a ∧ a < a/d*d+d := by
  have h0 := Int.emod_nonneg a (by omega : d≠0)
  have h1 := Int.emod_lt_of_pos a hd
  rw [Int.emod_def, Int.mul_comm d (a/d)] at h0 h1
  omega

theorem pair_alignment_error (accumulator a b d : Int) (hd : 0<d) :
    let rounded := (accumulator/d+a/d+b/d)*d
    rounded ≤ accumulator+a+b ∧ accumulator+a+b-rounded ≤ 3*(d-1) := by
  have h0 := floor_error accumulator d hd
  have h1 := floor_error a d hd
  have h2 := floor_error b d hd
  simp only [Int.add_mul]
  omega

theorem pair_alignment_tight :
    let d : Int := 256
    let x : Int := 255
    3*x-(x/d+x/d+x/d)*d = 3*(d-1) := by decide +kernel

theorem alignment_preserves_exact_multiple (a d : Int) (hd : d≠0) :
    (d*a)/d=a := Int.mul_ediv_cancel_left a hd

#print axioms floor_error
#print axioms pair_alignment_error
#print axioms pair_alignment_tight
#print axioms alignment_preserves_exact_multiple

end GB300Math.TextureAnisotropyArithmeticBounds
