/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorCore

namespace GB300Math.TensorSmallBlackwell
open GB300Math.Tensor

def format (name : String) : Option (Nat × Nat) :=
  match name with
  | "e4m3" => some (3,4)
  | "e5m2" => some (2,5)
  | "e2m3" => some (3,2)
  | "e3m2" => some (2,3)
  | "e2m1" => some (1,2)
  | _ => none

def nan (name : String) (x : Nat) : Bool :=
  if name = "e4m3" then x%128 == 127
  else if name = "e5m2" then isNaN x 2 5 else false

def infinity (name : String) (x : Nat) : Bool :=
  name == "e5m2" && isInf x 2 5

def dot (afmt bfmt : String) (a b : Array Nat) (c : Nat) : Option Nat := do
  let (af, ae) ← format afmt
  let (bf, be) ← format bfmt
  if a.size != 32 || b.size != 32 || c ≥ 2^32 ||
      !a.all (· < 2^(af+ae+1)) || !b.all (· < 2^(bf+be+1)) then none else
  some (Id.run do
    let accumulator := decode c 23 8
    let mut isNan := isNaN c 23 8
    let mut positiveInf := isInf c 23 8 && c < 2^31
    let mut negativeInf := isInf c 23 8 && c ≥ 2^31
    let ctop := if accumulator.significand = 0 then -10000 else max (-126) (inputTop accumulator)
    let mut terms : Array (GB300Math.Tensor.Dyadic × Int) := #[(accumulator,ctop)]
    for i in [:32] do
      let ax := a[i]!
      let bx := b[i]!
      let ai := infinity afmt ax
      let bi := infinity bfmt bx
      isNan := isNan || nan afmt ax || nan bfmt bx ||
        (ai && isZero bx bf be) || (bi && isZero ax af ae)
      if ai || bi then
        let negative := ax / 2^(af+ae) != bx / 2^(bf+be)
        positiveInf := positiveInf || !negative
        negativeInf := negativeInf || negative
      let x := decode ax af ae
      let y := decode bx bf be
      let top := if x.significand = 0 || y.significand = 0 then -10000
        else x.exponent+y.exponent+af+bf
      terms := terms.push (⟨x.significand*y.significand,x.exponent+y.exponent⟩,top)
    if isNan || (positiveInf && negativeInf) then return 0x7fffffff
    if positiveInf || negativeInf then return (if negativeInf then 2^31 else 0)+0x7f800000
    let top := terms.foldl (fun s t => max s t.2) (-10000)
    let exponent := top-25
    let coefficient := terms.foldl
      (fun s t => s + shiftTruncate t.1.significand (t.1.exponent-exponent)) 0
    return encodeRz ⟨coefficient,exponent⟩)

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem e4m3_maximum_is_finite : nan "e4m3" 126 = false ∧ infinity "e4m3" 126 = false := by decide

theorem e5m2_infinity_times_zero :
    dot "e5m2" "e2m1" (Array.replicate 32 124) (Array.replicate 32 0) 0 =
      some 0x7fffffff := by decide +kernel

theorem mixed_unit_dot :
    dot "e4m3" "e2m1" (Array.replicate 32 56) (Array.replicate 32 2) 0 =
      some 0x42000000 := by decide +kernel

#print axioms e4m3_maximum_is_finite
#print axioms e5m2_infinity_times_zero
#print axioms mixed_unit_dot
end GB300Math.TensorSmallBlackwell
