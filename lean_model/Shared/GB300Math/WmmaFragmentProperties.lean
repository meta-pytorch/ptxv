/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.WmmaFragmentMap

namespace GB300Math.WmmaFragmentMap

def capacity (layout : Layout) : Nat :=
  layout.segments.foldl (fun acc s => acc + (2^s.width-1)*2^s.target) layout.constant

theorem fold_bound (input : Array Nat) (xs : List Segment) (a b : Nat) (h : a ≤ b) :
    xs.foldl (fun acc s => acc + field input s.source s.width * 2^s.target) a ≤
      xs.foldl (fun acc s => acc + (2^s.width-1)*2^s.target) b := by
  induction xs generalizing a b with
  | nil => exact h
  | cons s xs ih =>
    simp only [List.foldl_cons]
    apply ih
    have hf := field_bound input s.source s.width
    have hm : field input s.source s.width ≤ 2^s.width-1 := by omega
    exact Nat.add_le_add h (Nat.mul_le_mul_right _ hm)

theorem word_le_capacity (input : Array Nat) (layout : Layout) :
    word input layout ≤ capacity layout := by
  have h := fold_bound input layout.segments.toList layout.constant layout.constant (by omega)
  simpa only [word, capacity, Array.foldl_toList] using h

theorem word_bound (input : Array Nat) (layout : Layout) (h : capacity layout < 2^32) :
    word input layout < 2^32 := by
  exact Nat.lt_of_le_of_lt (word_le_capacity input layout) h

#print axioms fold_bound
#print axioms word_le_capacity
#print axioms word_bound
end GB300Math.WmmaFragmentMap
