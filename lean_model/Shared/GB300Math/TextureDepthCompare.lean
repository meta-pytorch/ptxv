/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TextureDepthCompare

def magnitude (word : Nat) : Nat := word % 2^31

def isNaN (word : Nat) : Bool := magnitude word > 0x7f800000

def flush (word : Nat) : Nat :=
  if magnitude word < 0x00800000 then 0 else word

def eq (a b : Nat) : Bool :=
  let a := flush a
  let b := flush b
  !isNaN a && !isNaN b && a == b

def lt (a b : Nat) : Bool :=
  let a := flush a
  let b := flush b
  if isNaN a || isNaN b then false
  else if a / 2^31 != b / 2^31 then a / 2^31 == 1
  else if a / 2^31 == 0 then a < b else b < a

def compare (function reference depth : Nat) : Bool :=
  match function with
  | 0 => false
  | 1 => lt reference depth
  | 2 => eq reference depth
  | 3 => lt reference depth || eq reference depth
  | 4 => lt depth reference
  | 5 => !eq reference depth
  | 6 => lt depth reference || eq reference depth
  | _ => true

def lane (value : Bool) : Nat := if value then 1 else 0

def blend (x y : Nat) (a b c d : Bool) : Nat :=
  (4-x)*(4-y)*lane a + x*(4-y)*lane b +
  (4-x)*y*lane c + x*y*lane d

def encodeSixteenths (value : Nat) : Nat :=
  if value = 0 then 0
  else (value.log2 + 123)*2^23 + (value*2^(23-value.log2)) % 2^23

def encodeSixtyFourths (value : Nat) : Nat :=
  if value = 0 then 0
  else (value.log2 + 121)*2^23 + (value*2^(23-value.log2)) % 2^23

def coordinate (word : Nat) : Option Nat :=
  (List.range 66).find? fun value => encodeSixtyFourths value == word

def clamp (value : Int) : Nat := (max 0 (min 15 value)).toNat

def sample (depths : Array Nat) (x y function reference : Nat) (linear : Bool) : Nat :=
  let fetch :=  fun (ix iy : Int) =>
    compare function reference depths[clamp iy * 16 + clamp ix]!
  if linear then
    let baseX := ((x : Int)-2) / 4
    let baseY := ((y : Int)-2) / 4
    let fx := (((x : Int)-2) % 4).toNat
    let fy := (((y : Int)-2) % 4).toNat
    encodeSixteenths (blend fx fy (fetch baseX baseY) (fetch (baseX+1) baseY)
      (fetch baseX (baseY+1)) (fetch (baseX+1) (baseY+1)))
  else encodeSixteenths (16*lane (fetch (x/4) (y/4)))

theorem blend_bounded (x y : Fin 5) (a b c d : Bool) :
    blend x.val y.val a b c d ≤ 16 := by
  revert x y a b c d
  decide +kernel

theorem uniform_blend (x y : Fin 5) (value : Bool) :
    blend x.val y.val value value value value = 16*lane value := by
  revert x y value
  decide +kernel

theorem negzero_eq_poszero : compare 2 0x80000000 0 = true := by decide +kernel

theorem subnormal_reference_eq_zero : compare 2 1 0 = true := by decide +kernel

theorem subnormal_reference_not_greater : compare 4 1 0 = false := by decide +kernel

theorem nan_unordered :
    compare 1 0x7fc00000 0x3f000000 = false ∧
    compare 2 0x7fc00000 0x3f000000 = false ∧
    compare 3 0x7fc00000 0x3f000000 = false ∧
    compare 4 0x7fc00000 0x3f000000 = false ∧
    compare 5 0x7fc00000 0x3f000000 = true ∧
    compare 6 0x7fc00000 0x3f000000 = false := by decide +kernel

theorem representable_blends :
    (List.range 17).map encodeSixteenths =
      [0, 0x3d800000, 0x3e000000, 0x3e400000, 0x3e800000, 0x3ea00000,
       0x3ec00000, 0x3ee00000, 0x3f000000, 0x3f100000, 0x3f200000,
       0x3f300000, 0x3f400000, 0x3f500000, 0x3f600000, 0x3f700000,
       0x3f800000] := by decide +kernel

end GB300Math.TextureDepthCompare
