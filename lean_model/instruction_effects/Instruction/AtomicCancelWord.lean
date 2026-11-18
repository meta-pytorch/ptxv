/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Aeneas
open Aeneas
namespace AtomicCancel

def evalCancel (exclusiveOr : Bool) (left right : BitVec 32) : BitVec 32 :=
  if exclusiveOr then left ^^^ right else left - right

theorem word_self_subtract (value : BitVec 32) : value - value = 0 := by simp

theorem word_self_xor (value : BitVec 32) : value ^^^ value = 0 := by simp

theorem accepted_word_is_zero (exclusiveOr : Bool) (value : BitVec 32) :
    evalCancel exclusiveOr value value = 0 := by
  cases exclusiveOr <;> simp [evalCancel]

theorem accepted_word_is_return_independent (exclusiveOr : Bool) (left right : BitVec 32) :
    evalCancel exclusiveOr left left = evalCancel exclusiveOr right right := by
  simp [accepted_word_is_zero]
end AtomicCancel
