/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2Complete
namespace GB300Math.Log2PtxContract
set_option maxHeartbeats 3000000

theorem half_relative :
    ∃ x y : Rat, Root32FloatLib.value 0x3f000000 = some x ∧
      Root32FloatLib.value (Mufu.lg2 0x3f000000) = some y ∧ 0 < x ∧
      |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|/|Real.log (x:ℝ)/Real.log 2| ≤ 1/2^22 := by
  obtain ⟨x,y,hx,hy,hp,herr⟩ := Log2PtxRange.raw_near_absolute_error 0x3f000000 (by norm_num)
  refine ⟨x,y,hx,hy,hp,?_⟩
  have hhalf : Root32FloatLib.value 0x3f000000 = some (1/2) := by decide +kernel
  rw [hhalf] at hx
  have hxval : x = 1/2 := (Option.some.inj hx).symm
  have hideal : Real.log (x:ℝ)/Real.log 2 = -1 := by
    rw [hxval]
    norm_num only [Rat.cast_div, Rat.cast_one, Rat.cast_ofNat]
    rw [one_div, Real.log_inv]
    have hlog : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
    field_simp
  rw [hideal] at herr ⊢
  simpa using herr

theorem raw_relative_outside_open (w : Nat)
    (hw : 2^23 ≤ w ∧ w < 255*2^23)
    (ho : w ≤ 126*2^23 ∨ 128*2^23 ≤ w) :
    ∃ x y : Rat, Root32FloatLib.value w = some x ∧
      Root32FloatLib.value (Mufu.lg2 w) = some y ∧ 0 < x ∧
      |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|/|Real.log (x:ℝ)/Real.log 2| ≤ 1/2^22 := by
  by_cases hh : w = 126*2^23
  · subst w
    exact half_relative
  · exact Log2Complete.raw_outside_relative w hw (by omega)

theorem documented_open_contract (w : Nat) (hw : 2^23 ≤ w ∧ w < 255*2^23) :
    ∃ x y : Rat, Root32FloatLib.value w = some x ∧
      Root32FloatLib.value (Mufu.lg2 w) = some y ∧ 0 < x ∧
      (if 126*2^23 < w ∧ w < 128*2^23
       then |(y:ℝ)-Real.log (x:ℝ)/Real.log 2| ≤ 1/2^22
       else |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|/|Real.log (x:ℝ)/Real.log 2| ≤ 1/2^22) := by
  split_ifs with hnear
  · exact Log2PtxRange.raw_near_absolute_error w (by omega)
  · exact raw_relative_outside_open w hw (by omega)

#print axioms half_relative
#print axioms raw_relative_outside_open
#print axioms documented_open_contract
end GB300Math.Log2PtxContract
