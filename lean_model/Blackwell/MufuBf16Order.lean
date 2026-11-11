/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.MufuSm103

namespace GB300Math.MufuBf16Order

set_option maxRecDepth 100000

def magnitude (word : Nat) : Nat := word % 32768

def numericKey (word : Nat) : Int :=
  if word / 32768 % 2 = 1 then -(magnitude word : Int) else magnitude word

def code (domain : String) (index : Nat) : Nat :=
  if domain = "finite" then
    if index < 32640 then 0xff7f-index else index-32640
  else if domain = "negative" then 0xff7f-index
  else if domain = "positive" then index+1
  else index

def validOrder (op domain : String) (decreasing : Bool) (index : Nat) : Bool :=
  let a := (MufuSm103.evaluateBf16 op (code domain index)).getD 0x7fff
  let b := (MufuSm103.evaluateBf16 op (code domain (index+1))).getD 0x7fff
  magnitude a ≤ 0x7f80 && magnitude b ≤ 0x7f80 &&
    if decreasing then numericKey b ≤ numericKey a else numericKey a ≤ numericKey b

theorem zero_keys_equal : numericKey 0x8000 = numericKey 0 := by decide +kernel

theorem finite_domain_bounds (index : Nat) (h : index < 65280) :
    magnitude (code "finite" index) < 0x7f80 := by
  change (if index < 32640 then 65407-index else index-32640) % 32768 < 32640
  split <;> omega

theorem positive_domain_bounds (index : Nat) (h : index < 32639) :
    0 < code "positive" index ∧ code "positive" index < 0x7f80 := by
  change 0 < index+1 ∧ index+1 < 32640
  omega

#print axioms zero_keys_equal
#print axioms finite_domain_bounds
#print axioms positive_domain_bounds

end GB300Math.MufuBf16Order
