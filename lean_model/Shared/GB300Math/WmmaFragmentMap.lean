/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.WmmaFragmentMap

structure Segment where
  source : Nat
  target : Nat
  width : Nat

structure Layout where
  constant : Nat
  segments : Array Segment
  deriving Inhabited

def field (input : Array Nat) (offset width : Nat) : Nat :=
  let low := input[offset / 32]!
  let high := input[offset / 32 + 1]!
  (low + high*2^32) / 2^(offset % 32) % 2^width

def word (input : Array Nat) (layout : Layout) : Nat :=
  layout.segments.foldl (fun acc s => acc + field input s.source s.width * 2^s.target)
    layout.constant

def validSegment (s : Segment) : Prop :=
  s.width > 0 ∧ s.width ≤ 32 ∧ s.source+s.width ≤ 32768 ∧ s.target+s.width ≤ 32

theorem field_bound (input : Array Nat) (offset width : Nat) :
    field input offset width < 2^width := by
  exact Nat.mod_lt _ (Nat.two_pow_pos _)

theorem zero_width (input : Array Nat) (offset : Nat) : field input offset 0 = 0 := by
  unfold field
  simp only [Nat.pow_zero, Nat.mod_one]

theorem constant_layout (input : Array Nat) (v : Nat) : word input ⟨v,#[]⟩ = v := by
  rfl

theorem source_in_bounds (s : Segment) (h : validSegment s) : s.source < 32768 := by
  unfold validSegment at h
  omega

#print axioms field_bound
#print axioms zero_width
#print axioms constant_layout
#print axioms source_in_bounds
end GB300Math.WmmaFragmentMap
