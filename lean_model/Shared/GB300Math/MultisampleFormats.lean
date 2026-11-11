/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.MultisampleFormats

def lane (channels format component input : Nat) : Nat :=
  if component < channels then input
  else if component = 3 then (if format = 0 then 0x3f800000 else 1)
  else 0

theorem full_payload (format component input : Nat) (h : component < 4) :
    lane 4 format component input = input := by simp [lane, h]

theorem red_payload (format input : Nat) : lane 1 format 0 input = input := by
  simp [lane]

theorem absent_rgb (format : Nat) : lane 1 format 1 0xffffffff = 0 ∧
    lane 1 format 2 0xffffffff = 0 := by simp [lane]

theorem alpha_float : lane 1 0 3 0 = 0x3f800000 := by decide +kernel

theorem alpha_integer : lane 1 1 3 0 = 1 ∧ lane 1 2 3 0 = 1 := by decide +kernel

theorem signaling_nan_preserved : lane 1 0 0 0x7f800001 = 0x7f800001 := by decide +kernel

theorem negative_zero_preserved : lane 1 0 0 0x80000000 = 0x80000000 := by decide +kernel

end GB300Math.MultisampleFormats
