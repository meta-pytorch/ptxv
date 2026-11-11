/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorCore

namespace GB300Math.TensorScaledIdeal
open GB300Math.Tensor

def fp4Twice (word : Nat) : Int :=
  let magnitude := (#[0,1,2,3,4,6,8,12] : Array Nat)[word % 8]!
  if word < 8 then magnitude else -(magnitude : Int)

def scale (nv : Bool) (word : Nat) : Option GB300Math.Tensor.Dyadic :=
  if word ≥ 256 then none
  else if nv then
    if word % 128 = 127 then none else some (decode word 3 4)
  else if word = 255 then none else some ⟨1, (word : Int)-127⟩

def exactDot (nv : Bool) (a b sa sb : Array Nat) (c : Nat) : Option GB300Math.Tensor.Dyadic := Id.run do
  let block := if nv then 16 else 32
  if a.size != b.size || a.size % block != 0 || sa.size != a.size/block ||
      sb.size != sa.size || !(a ++ b).all (· < 16) || c ≥ 2^32 ||
      isNaN c 23 8 || isInf c 23 8 then return none
  let mut terms := #[decode c 23 8]
  for j in [:sa.size] do
    let some x := scale nv sa[j]! | return none
    let some y := scale nv sb[j]! | return none
    let mut quarterSum : Int := 0
    for i in [:block] do
      quarterSum := quarterSum + fp4Twice a[j*block+i]! * fp4Twice b[j*block+i]!
    terms := terms.push ⟨quarterSum*x.significand*y.significand, x.exponent+y.exponent-2⟩
  let exponent := terms.foldl (fun e t => min e t.exponent) 0
  let coefficient := terms.foldl
    (fun v t => v + t.significand * (2^((t.exponent-exponent).toNat) : Int)) 0
  return some ⟨coefficient, exponent⟩

def idealRz (nv : Bool) (a b sa sb : Array Nat) (c : Nat) : Option Nat := do
  let block := if nv then 16 else 32
  if a.size != b.size || !(a.size = 64 || a.size = 96) || c ≥ 2^32 ||
      sa.size != a.size/block || sb.size != sa.size ||
      !(a ++ b).all (· < 16) || !(sa ++ sb).all (· < 256) then none else
  if isNaN c 23 8 || (sa ++ sb).any (fun x => (scale nv x).isNone) then some 0x7fffffff
  else if isInf c 23 8 then some c
  else some (encodeRz (← exactDot nv a b sa sb c))

def physicalColumn (logicalRow block : Nat) : Nat := logicalRow/32 + 4*(block/4)
def physicalByte (block : Nat) : Nat := block%4

theorem physical_column_in_range (row block : Nat) (hr : row < 128) (hb : block < 16) :
    physicalColumn row block < 16 := by unfold physicalColumn; omega

theorem scale_slot_roundtrip (row block : Nat) (hr : row < 128) :
    physicalColumn row block % 4 = row/32 ∧
    (physicalColumn row block / 4)*4 + physicalByte block = block := by
  unfold physicalColumn physicalByte
  omega

theorem fp4_maximum : fp4Twice 7 = 12 ∧ fp4Twice 15 = -12 := by decide

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem unit_scale_integer_dot :
    idealRz false (Array.replicate 64 2) (Array.replicate 64 2) #[127,127] #[127,127] 0 =
      some 0x42800000 := by decide +kernel

#print axioms physical_column_in_range
#print axioms scale_slot_roundtrip
#print axioms unit_scale_integer_dot
end GB300Math.TensorScaledIdeal
