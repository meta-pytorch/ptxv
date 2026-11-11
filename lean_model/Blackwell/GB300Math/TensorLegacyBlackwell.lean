/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorBlackwell
import GB300Math.SparseTensorExtra

namespace GB300Math.TensorLegacyBlackwell
open GB300Math.Tensor
open GB300Math.SparseTensorExtra

def halfDot (a b : Array Nat) (c : Nat) : Nat := Id.run do
  let fw := 10
  let ew := 5
  let accumulator := decode c 10 5
  let mut nan := isNaN c 10 5
  let mut positiveInf := isInf c 10 5 && c < 2^15
  let mut negativeInf := isInf c 10 5 && c ≥ 2^15
  let accumulatorTop := if accumulator.significand = 0 then -10000
    else max (-14) (inputTop accumulator)
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
  if nan || (positiveInf && negativeInf) then return 0x7fff
  if positiveInf || negativeInf then
    return (if negativeInf then 2^15 else 0) + 0x7c00
  let top := terms.foldl (fun s t => max s t.2) (-10000)
  let scale := top - 25
  let total := terms.foldl
    (fun s t => s + shiftTruncate t.1.significand (t.1.exponent-scale)) 0
  return GB300Math.SparseTensorExtra.roundHalf total scale


def dot (kind acc : String) (a b : Array Nat) (c : Nat) : Nat :=
  if acc == "f16" then halfDot a b c
  else if kind == "tf32" then TensorBlackwell.alignedDot
    (a.map canonicalTF32) (b.map canonicalTF32) c 23 8 true
  else if kind == "bf16" then TensorBlackwell.alignedDot a b c 7 8 true
  else TensorBlackwell.alignedDot a b c 10 5 true

def flip (bits mode value : Nat) : Nat :=
  if mode % 2 = 1 then value ^^^ 2^(bits-1) else value

def dense (kind acc : String) (k mods : Nat) (a b c : Array Nat) : Option (Array Nat) := do
  let bits ← if kind == "tf32" then some 32
    else if kind == "f16" || kind == "bf16" then some 16 else none
  if a.size != 16*k || b.size != 8*k || c.size != 128 || mods ≥ 4 ||
      !(a++b).all (· < 2^bits) || !c.all (· < if acc == "f16" then 65536 else 2^32) then none
  else if !(k == (if kind == "tf32" then 4 else 8) || k == (if kind == "tf32" then 8 else 16)) ||
      !(acc == "f32" || (acc == "f16" && kind == "f16")) then none
  else
    let mut result := #[]
    for row in [:16] do
      let av := (Array.range k).map fun j => flip bits mods a[row*k+j]!
      for col in [:8] do
        let bv := (Array.range k).map fun j => flip bits (mods/2) b[col*k+j]!
        result := result.push (dot kind acc av bv c[row*8+col]!)
    return result

def sparse (m : Mode) (mods : Nat) (a b c e : Array Nat) : Option (Array Nat) := do
  if mods ≥ 4 || m.bits < 16 || !validInput m a b c e then none else
    let mut out := #[]
    for row in [:16] do
      let av := (Array.range (m.k/2)).map fun j => flip m.bits mods a[row*(m.k/2)+j]!
      for col in [:8] do
        let bv := (Array.range (m.k/2)).map fun j =>
          flip m.bits (mods/2) b[col*m.k+position m e row j]!
        out := out.push (dot m.a m.acc av bv c[row*8+col]!)
    return out

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem full_group_avoids_intermediate_half_overflow :
    halfDot (Array.replicate 8 0x5c00 ++ Array.replicate 8 0xdc00)
      (Array.replicate 16 0x5c00) 0x3c00 = 0x3c00 ∧
    GB300Math.SparseTensorExtra.halfDot
      (Array.replicate 8 0x5c00 ++ Array.replicate 8 0xdc00)
      (Array.replicate 16 0x5c00) 0x3c00 = 0x7c00 := by
  decide +kernel

#print axioms full_group_avoids_intermediate_half_overflow
end GB300Math.TensorLegacyBlackwell
