/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorScaledIdeal

namespace GB300Math.TensorScaledSelected
open GB300Math.Tensor GB300Math.TensorScaledIdeal

/-- Selected alignment hypothesis; hardware agreement remains corpus-bounded. -/
def evaluateCore (nv : Bool) (block : Nat) (a b sa sb : Array Nat) (c : Nat)
    (minimumStep : Int) : Option Nat := Id.run do
  if !(block = 16 || block = 32) || (nv && block != 16) then return none
  if a.size != b.size || !(a.size = 64 || a.size = 96) || c ≥ 2^32 ||
      sa.size != a.size/block || sb.size != sa.size ||
      !(a ++ b).all (· < 16) || !(sa ++ sb).all (· < (if nv then 128 else 256)) then return none
  if isNaN c 23 8 || (sa ++ sb).any (fun x => (scale nv x).isNone) then
    return some 0x7fffffff
  if isInf c 23 8 then return some c
  let accumulator := decode c 23 8
  let mut top := if accumulator.significand = 0 then -10000
    else max (-126) (inputTop accumulator)
  let mut terms := #[accumulator]
  for j in [:sa.size] do
    let some x := scale nv sa[j]! | return none
    let some y := scale nv sb[j]! | return none
    if x.significand != 0 && y.significand != 0 &&
        (Array.range block).any (fun i => fp4Twice a[j*block+i]! != 0 &&
          fp4Twice b[j*block+i]! != 0) then
      top := max top (x.exponent+y.exponent+(if nv then 6 else 0))
    for half in [:block/16] do
      let mut quarterSum : Int := 0
      for i in [:16] do
        let index := j*block+half*16+i
        quarterSum := quarterSum + fp4Twice a[index]! * fp4Twice b[index]!
      terms := terms.push ⟨quarterSum*x.significand*y.significand,
        x.exponent+y.exponent-2⟩
  let unit := max minimumStep (top-35)
  let total := terms.foldl
    (fun s t => s + shiftTruncate t.significand (t.exponent-unit)) 0
  return some (encodeRz ⟨total, unit⟩)

def evaluate (nv : Bool) (a b sa sb : Array Nat) (c : Nat) : Option Nat :=
  evaluateCore nv (if nv then 16 else 32) a b sa sb c (-174)

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem unit_scale_dot :
    evaluate false (Array.replicate 64 2) (Array.replicate 64 2)
      #[127,127] #[127,127] 0 = some 0x42800000 := by decide +kernel

theorem malformed_scale_extent_rejected :
    evaluate false (Array.replicate 64 2) (Array.replicate 64 2)
      #[127] #[127,127] 0 = none := by decide +kernel

theorem nan_scale_overrides_infinite_accumulator :
    evaluate false (Array.replicate 64 2) (Array.replicate 64 2)
      #[255,127] #[127,127] 0x7f800000 = some 0x7fffffff := by decide +kernel

theorem nonzero_unsigned_scale_padding_rejected :
    evaluate true (Array.replicate 64 2) (Array.replicate 64 2)
      #[128,56,56,56] #[56,56,56,56] 0 = none := by decide +kernel

#print axioms nonzero_unsigned_scale_padding_rejected
#print axioms unit_scale_dot
#print axioms malformed_scale_extent_rejected
#print axioms nan_scale_overrides_infinite_accumulator
end GB300Math.TensorScaledSelected
