/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorCore

namespace GB300Math.TensorAccumulatorBlackwell
open GB300Math.Tensor

def alignedDot (a b : Array Nat) (c fw ew : Nat)
    (shift mode : Nat) : Nat := Id.run do
  let original := decode c 23 8
  let exactScaled : GB300Math.Tensor.Dyadic := ⟨original.significand,original.exponent-(shift:Int)⟩
  let accumulator := if mode = 2 then decode (encodeRz exactScaled) 23 8 else exactScaled
  let mut nan := isNaN c 23 8
  let mut positiveInf := isInf c 23 8 && c < 2^31
  let mut negativeInf := isInf c 23 8 && c ≥ 2^31
  let accumulatorTop := if accumulator.significand = 0 then -10000
    else if mode = 1 then max (-126) (inputTop original) - (shift:Int)
    else max (-126) (inputTop accumulator)
  let mut terms : Array (GB300Math.Tensor.Dyadic × Int) := #[(accumulator, accumulatorTop)]
  for i in [:a.size] do
    let ax := a[i]!
    let bx := b[i]!
    nan := nan || isNaN ax fw ew || isNaN bx fw ew ||
      (isInf ax fw ew && isZero bx fw ew) || (isZero ax fw ew && isInf bx fw ew)
    if isInf ax fw ew || isInf bx fw ew then
      let negative := ax / 2^(fw+ew) % 2 != bx / 2^(fw+ew) % 2
      positiveInf := positiveInf || !negative
      negativeInf := negativeInf || negative
    let x := decode ax fw ew
    let y := decode bx fw ew
    terms := terms.push (⟨x.significand*y.significand, x.exponent+y.exponent⟩,
      if x.significand = 0 ∨ y.significand = 0 then -10000
      else x.exponent+y.exponent+2*fw)
  if nan || (positiveInf && negativeInf) then return 0x7fffffff
  if positiveInf || negativeInf then
    return (if negativeInf then 2^31 else 0) + 0x7f800000
  let top := terms.foldl (fun s t => max s t.2) (-10000)
  let scale := max (-158) (top - 25)
  let total := terms.foldl
    (fun s t => s + shiftTruncate t.1.significand (t.1.exponent-scale)) 0
  return encodeRz ⟨total, scale⟩

def evaluate (fmt : String) (a b : Array Nat) (c : Nat)
    (shift mode : Nat) : Option Nat := do
  let (fw, ew, k) ← match fmt with
    | "f16" => some (10,5,16)
    | "bf16" => some (7,8,16)
    | "tf32" => some (23,8,8)
    | _ => none
  if a.size != k || b.size != k || c ≥ 2^32 || shift > 15 || mode > 2 then none
  else if !(a ++ b).all (fun x => x < 2^(fw+ew+1)) then none
  else if fmt = "tf32" && !(a ++ b).all (fun x => x % 2^13 = 0) then none
  else some (alignedDot a b c fw ew shift mode)

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem clamp_order_discriminator :
    alignedDot #[0x9a20] #[0x1a00] 3 7 8 1 0 = 1 ∧
    alignedDot #[0x9a20] #[0x1a00] 3 7 8 1 1 = 0 := by decide +kernel

theorem intermediate_rounding_discriminator :
    alignedDot #[0x1a00] #[0x1a00] 1 7 8 1 1 = 1 ∧
    alignedDot #[0x1a00] #[0x1a00] 1 7 8 1 2 = 0 := by decide +kernel

#print axioms clamp_order_discriminator
#print axioms intermediate_rounding_discriminator
end GB300Math.TensorAccumulatorBlackwell
