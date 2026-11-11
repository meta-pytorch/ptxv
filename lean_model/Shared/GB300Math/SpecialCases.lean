/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Basic

namespace GB300Math

def magnitude (x : Word32) : Word32 := x &&& 0x7fffffff

def signBit (x : Word32) : Word32 := x &&& 0x80000000

def flushSubnormal (x : Word32) : Word32 :=
  if magnitude x < 0x00800000 then signBit x else x

/-- Candidate partial models. `none` means the result is not modeled. -/
def ex2Special (x : Word32) : Option Word32 :=
  if magnitude x ≤ 0x00800000 then some 0x3f800000
  else if magnitude x > 0x7f800000 then some 0x7fffffff
  else if x = 0x7f800000 then some 0x7f800000
  else if x = 0xff800000 then some 0
  else none

def lg2Special (ftz : Bool) (input : Word32) : Option Word32 :=
  let x := if ftz then flushSubnormal input else input
  if magnitude x = 0 then some 0xff800000
  else if magnitude x > 0x7f800000 ∨ signBit x ≠ 0 then some 0x7fffffff
  else if x = 0x7f800000 then some 0x7f800000
  else if x = 0x3f800000 then some 0
  else none

def tanhSpecial (x : Word32) : Option Word32 :=
  if magnitude x ≤ 0x00800000 then some x
  else if magnitude x > 0x7f800000 then some 0x7fffffff
  else if magnitude x = 0x7f800000 then some (signBit x ||| 0x3f800000)
  else none

theorem tanh_preserves_tiny_words (x : Word32)
    (h : magnitude x ≤ 0x00800000) : tanhSpecial x = some x := by
  unfold tanhSpecial
  rw [ite_eq_left h]

theorem ex2_tiny_words (x : Word32)
    (h : magnitude x ≤ 0x00800000) : ex2Special x = some 0x3f800000 := by
  unfold ex2Special
  rw [ite_eq_left h]

theorem tanh_negative_zero : tanhSpecial 0x80000000 = some 0x80000000 := by decide

theorem lg2_negative_subnormal_ftz : lg2Special true 0x80000001 = some 0xff800000 := by
  decide

theorem lg2_negative_subnormal_default : lg2Special false 0x80000001 = some 0x7fffffff := by
  decide

theorem lg2_modes_differ : lg2Special true 0x80000001 ≠ lg2Special false 0x80000001 := by
  decide

end GB300Math
